"""
Theory Forge — FastAPI server.

Endpoints:
    GET  /theories                                seed theories
    GET  /domains                                 unique domain list (for UI dropdowns)
    GET  /ledger                                  known collisions (tried/failed)
    GET  /pairs?limit=&novel_only=&alpha=         ranked cross-domain pairs
    POST /collide          { a_id, b_id }         synthesize new framework
    POST /collide_domains  { domain_a, domain_b, novel_only? }
    POST /batch_collide    { count, domain_filter? }  batch N collisions
    GET  /history                                 all saved batch outputs
    GET  /rankings?limit=&min_confidence=&domain= ranked frameworks
    GET  /                                        one-page UI

Embeddings load lazily on first request that needs them (alpha > 0).
The first call may take ~10s downloading the model; subsequent calls are fast.

Auth: ANTHROPIC_API_KEY must be set in the environment. Never hardcoded.
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import forge

app = FastAPI(title="Theory Forge", version="0.3.0")

_theories = forge.load_theories()
_by_id = {t.id: t for t in _theories}
_ledger = forge.load_ledger()
_tried = forge.tried_keys(_ledger)
_vecs: dict | None = None  # lazy


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set in environment.")
    try:
        import anthropic
    except ImportError as e:
        raise HTTPException(500, f"anthropic SDK not installed: {e}") from e
    return anthropic.Anthropic(api_key=key)


def _ensure_vecs() -> dict | None:
    """Lazy-load embeddings. Returns None if sentence-transformers isn't installed."""
    global _vecs
    if _vecs is None:
        _vecs = forge.embed_theories(_theories)  # may stay None on ImportError
    return _vecs


class CollideIn(BaseModel):
    a_id: str
    b_id: str


class CollideDomainsIn(BaseModel):
    domain_a: str
    domain_b: str
    novel_only: bool = True


class BatchCollideIn(BaseModel):
    count: int = 5
    domain_filter: str | None = None


@app.get("/theories")
def list_theories():
    return [t.to_dict() for t in _theories]


@app.get("/domains")
def list_domains():
    return sorted({t.domain for t in _theories})


@app.get("/ledger")
def get_ledger():
    return [
        {"a_id": tc.a_id, "b_id": tc.b_id, "status": tc.status,
         "label": tc.label, "note": tc.note}
        for tc in _ledger
    ]


@app.get("/pairs")
def list_pairs(
    limit: int = 20,
    min_sim: float = 0.25,
    max_sim: float = 0.92,
    novel_only: bool = True,
    alpha: float = 0.0,
):
    vecs = _ensure_vecs() if alpha > 0 else None
    rows = forge.candidate_pairs(
        _theories,
        min_sim=min_sim, max_sim=max_sim,
        vecs=vecs, alpha=alpha,
        exclude_tried=_tried if novel_only else set(),
    )
    return [
        {"a": a.id, "b": b.id, "a_name": a.name, "b_name": b.name,
         "a_domain": a.domain, "b_domain": b.domain,
         "similarity": round(s, 3),
         "embeddings": vecs is not None}
        for a, b, s in rows[:limit]
    ]


@app.post("/collide")
def collide(body: CollideIn):
    a, b = _by_id.get(body.a_id), _by_id.get(body.b_id)
    if not a or not b:
        raise HTTPException(404, "Unknown theory id.")
    prior = forge.find_prior(a, b, _ledger)
    framework = forge.collide(a, b, _client(), prior=prior)
    return {
        "a": a.to_dict(), "b": b.to_dict(),
        "framework": framework,
        "prior": prior.__dict__ if prior else None,
    }


@app.post("/collide_domains")
def collide_domains(body: CollideDomainsIn):
    vecs = _ensure_vecs()
    pick = forge.best_pair_between(
        _theories, body.domain_a, body.domain_b,
        vecs=vecs, alpha=0.5 if vecs else 0.0,
        exclude_tried=_tried if body.novel_only else set(),
    )
    if not pick:
        raise HTTPException(404, "No novel pair found between those domains "
                                 "(try novel_only=false).")
    a, b, s = pick
    prior = forge.find_prior(a, b, _ledger)
    framework = forge.collide(a, b, _client(), prior=prior)
    return {
        "a": a.to_dict(), "b": b.to_dict(),
        "similarity": round(s, 3),
        "framework": framework,
        "prior": prior.__dict__ if prior else None,
    }


@app.post("/batch_collide")
def batch_collide(body: BatchCollideIn):
    count = max(1, min(body.count, 10))
    vecs = _ensure_vecs()
    alpha = 0.5 if vecs else 0.0
    pairs = forge.candidate_pairs(
        _theories, vecs=vecs, alpha=alpha, exclude_tried=_tried,
    )
    if body.domain_filter:
        filt = body.domain_filter.lower()
        pairs = [
            (a, b, s) for a, b, s in pairs
            if filt in a.domain.lower() or filt in b.domain.lower()
        ]
    if not pairs:
        raise HTTPException(404, "No novel pairs found matching filters.")
    selected = random.sample(pairs[:50], min(count, len(pairs[:50])))
    client = _client()
    results = []

    def _do_collide(a, b):
        prior = forge.find_prior(a, b, _ledger)
        fw = forge.collide(a, b, client, prior=prior)
        return {"a": a.to_dict(), "b": b.to_dict(), "framework": fw,
                "prior": prior.__dict__ if prior else None}

    with ThreadPoolExecutor(max_workers=min(count, 4)) as pool:
        futures = {pool.submit(_do_collide, a, b): (a, b) for a, b, _ in selected}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                a, b = futures[fut]
                results.append({"error": str(e), "a": a.id, "b": b.id})
    return {"count": len(results), "results": results}


@app.get("/history")
def history():
    out_dir = Path(__file__).parent / "outputs"
    if not out_dir.exists():
        return []
    batches = []
    for f in sorted(out_dir.glob("batch-*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            batches.append({
                "file": f.name,
                "count": len(data),
                "frameworks": [
                    {"name": fw.get("name", "?"),
                     "confidence": fw.get("confidence", 0),
                     "source_a": fw.get("source_a", ""),
                     "source_b": fw.get("source_b", "")}
                    for fw in data
                ],
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return batches


@app.get("/rankings")
def rankings(limit: int = 50, min_confidence: float = 0.0, domain: str = ""):
    out_dir = Path(__file__).parent / "outputs"
    all_fw = []
    for f in sorted(out_dir.glob("batch-*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for fw in data:
                fw["_batch"] = f.name
                all_fw.append(fw)
        except (json.JSONDecodeError, KeyError):
            continue
    if min_confidence > 0:
        all_fw = [fw for fw in all_fw if fw.get("confidence", 0) >= min_confidence]
    if domain:
        d = domain.lower()
        all_fw = [fw for fw in all_fw
                  if d in fw.get("domain_applied_to", "").lower()
                  or d in fw.get("source_a", "").lower()
                  or d in fw.get("source_b", "").lower()
                  or d in fw.get("mechanism_borrowed_from", "").lower()]
    all_fw.sort(key=lambda fw: -fw.get("confidence", 0))
    return all_fw[:limit]


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
