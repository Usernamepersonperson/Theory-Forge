"""
Theory Forge — FastAPI server.

Endpoints:
    GET  /theories                                seed theories
    GET  /theories/{theory_id}                    single theory by id
    GET  /domains                                 unique domain list (for UI dropdowns)
    GET  /ledger                                  known collisions (tried/failed)
    GET  /pairs?limit=&novel_only=&alpha=         ranked cross-domain pairs
    POST /collide          { a_id, b_id }         synthesize new framework
    POST /collide_domains  { domain_a, domain_b, novel_only? }
    POST /batch_collide    { count, domain_filter? }  batch N collisions
    GET  /history                                 all saved batch outputs
    GET  /rankings?limit=&min_confidence=&domain= ranked frameworks
    GET  /deep-dives                              list deep-dive analyses
    GET  /deep-dives/{slug}                       single deep-dive
    GET  /stats                                   aggregate statistics
    GET  /search?q=&limit=                        full-text framework search
    GET  /gaps?limit=                             unexplored high-potential domain pairs
    GET  /framework/{name}                        single framework with related frameworks
    GET  /export?format=json|csv&min_confidence=  export all frameworks
    GET  /random?min_confidence=                  random framework above threshold
    GET  /domain-network                          graph data (nodes + weighted edges)
    GET  /semantic-search?q=&limit=               embedding-based semantic search
    GET  /recommend?limit=&strategy=              collision recommendations (coverage/diversity/bridge)
    GET  /chain/{name}?depth=                     framework exploration chain
    GET  /synthesis?domain=&top_n=&format=        synthesis report (markdown/html)
    GET  /tag-analysis                            tag frequency and domain spread
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

import csv
import io
import re
from collections import Counter, defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import forge

app = FastAPI(title="Theory Forge", version="0.7.0")

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
def list_theories(domain: str = ""):
    if domain:
        d = domain.lower()
        return [t.to_dict() for t in _theories if d in t.domain.lower()]
    return [t.to_dict() for t in _theories]


@app.get("/theories/{theory_id}")
def get_theory(theory_id: str):
    t = _by_id.get(theory_id)
    if not t:
        raise HTTPException(404, f"Theory '{theory_id}' not found.")
    return t.to_dict()


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
def rankings(limit: int = 50, offset: int = 0, min_confidence: float = 0.0, domain: str = ""):
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
    return all_fw[offset:offset + limit]


@app.get("/deep-dives")
def list_deep_dives():
    dd_dir = Path(__file__).parent / "outputs" / "deep-dives"
    if not dd_dir.exists():
        return []
    results = []
    for f in sorted(dd_dir.glob("deep-dive-*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({
                "file": f.name,
                "slug": f.stem,
                "name": data.get("name", "?"),
                "rank": data.get("rank", 0),
                "confidence": data.get("confidence", 0),
                "executive_summary": data.get("executive_summary", ""),
                "predictions_count": len(data.get("falsifiable_predictions", [])),
                "experiments_count": len(data.get("experimental_designs", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return results


@app.get("/deep-dives/{slug}")
def get_deep_dive(slug: str):
    dd_dir = Path(__file__).parent / "outputs" / "deep-dives"
    target = dd_dir / f"{slug}.json"
    if not target.exists():
        raise HTTPException(404, f"Deep dive '{slug}' not found.")
    return json.loads(target.read_text(encoding="utf-8"))


@app.get("/stats")
def stats():
    out_dir = Path(__file__).parent / "outputs"
    all_fw = []
    batch_count = 0
    for f in sorted(out_dir.glob("batch-*.json")):
        batch_count += 1
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for fw in data:
                fw["_batch"] = f.name
                all_fw.append(fw)
        except (json.JSONDecodeError, KeyError):
            continue
    seen = set()
    unique = []
    for fw in all_fw:
        n = fw.get("name", "").strip().lower()
        if n and n not in seen:
            seen.add(n)
            unique.append(fw)
    confs = [fw.get("confidence", 0) for fw in unique]
    viability = {}
    for fw in unique:
        v = fw.get("viability", "unknown")
        viability[v] = viability.get(v, 0) + 1
    domains_from = {}
    domains_to = {}
    for fw in unique:
        src = fw.get("mechanism_borrowed_from", "")
        dst = fw.get("domain_applied_to", "")
        if src:
            domains_from[src] = domains_from.get(src, 0) + 1
        if dst:
            domains_to[dst] = domains_to.get(dst, 0) + 1
    top_sources = sorted(domains_from.items(), key=lambda x: -x[1])[:15]
    top_targets = sorted(domains_to.items(), key=lambda x: -x[1])[:15]
    dd_dir = out_dir / "deep-dives"
    deep_dive_count = len(list(dd_dir.glob("deep-dive-*.json"))) if dd_dir.exists() else 0
    return {
        "theory_count": len(_theories),
        "domain_count": len({t.domain for t in _theories}),
        "batch_count": batch_count,
        "total_frameworks": len(unique),
        "deep_dive_count": deep_dive_count,
        "viability_breakdown": viability,
        "confidence_stats": {
            "mean": round(sum(confs) / len(confs), 3) if confs else 0,
            "max": round(max(confs), 3) if confs else 0,
            "min": round(min(confs), 3) if confs else 0,
        },
        "top_mechanism_sources": top_sources,
        "top_target_domains": top_targets,
    }


@app.get("/search")
def search_frameworks(q: str = "", limit: int = 20):
    if not q or len(q) < 2:
        return []
    out_dir = Path(__file__).parent / "outputs"
    q_lower = q.lower()
    matches = []
    for f in sorted(out_dir.glob("batch-*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for fw in data:
                fw["_batch"] = f.name
                text = json.dumps(fw).lower()
                if q_lower in text:
                    matches.append(fw)
        except (json.JSONDecodeError, KeyError):
            continue
    matches.sort(key=lambda fw: -fw.get("confidence", 0))
    return matches[:limit]


@app.get("/gaps")
def find_gaps(limit: int = 20):
    out_dir = Path(__file__).parent / "outputs"
    collided_domains: set[frozenset[str]] = set()
    for f in sorted(out_dir.glob("batch-*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for fw in data:
                sa = fw.get("source_a", "").strip()
                sb = fw.get("source_b", "").strip()
                if sa and sb:
                    collided_domains.add(frozenset({sa.lower(), sb.lower()}))
        except (json.JSONDecodeError, KeyError):
            continue
    theory_domains = sorted({t.domain for t in _theories})
    domain_theories = {}
    for t in _theories:
        domain_theories.setdefault(t.domain, []).append(t)
    gaps = []
    for i, da in enumerate(theory_domains):
        for db in theory_domains[i + 1:]:
            if frozenset({da.lower(), db.lower()}) not in collided_domains:
                max_sim = 0.0
                for ta in domain_theories[da]:
                    for tb in domain_theories[db]:
                        s = forge.tag_similarity(ta, tb)
                        if s > max_sim:
                            max_sim = s
                if max_sim >= 0.15:
                    gaps.append({
                        "domain_a": da, "domain_b": db,
                        "max_tag_similarity": round(max_sim, 3),
                        "theories_a": len(domain_theories[da]),
                        "theories_b": len(domain_theories[db]),
                    })
    gaps.sort(key=lambda g: -g["max_tag_similarity"])
    return gaps[:limit]


@app.get("/framework/{name}")
def get_framework(name: str):
    out_dir = Path(__file__).parent / "outputs"
    name_lower = name.lower().strip()
    for f in sorted(out_dir.glob("batch-*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for fw in data:
                if fw.get("name", "").lower().strip() == name_lower:
                    fw["_batch"] = f.name
                    mech = fw.get("mechanism_borrowed_from", "").lower()
                    domain = fw.get("domain_applied_to", "").lower()
                    related = []
                    for f2 in sorted(out_dir.glob("batch-*.json")):
                        try:
                            d2 = json.loads(f2.read_text(encoding="utf-8"))
                            for fw2 in d2:
                                if fw2.get("name", "").lower().strip() == name_lower:
                                    continue
                                m2 = fw2.get("mechanism_borrowed_from", "").lower()
                                d2v = fw2.get("domain_applied_to", "").lower()
                                if (mech and mech in m2) or (domain and domain in d2v):
                                    related.append({
                                        "name": fw2.get("name", "?"),
                                        "confidence": fw2.get("confidence", 0),
                                        "shared": "mechanism" if mech and mech in m2 else "domain",
                                    })
                        except (json.JSONDecodeError, KeyError):
                            continue
                    related.sort(key=lambda x: -x["confidence"])
                    fw["related_frameworks"] = related[:10]
                    return fw
        except (json.JSONDecodeError, KeyError):
            continue
    raise HTTPException(404, f"Framework '{name}' not found.")


@app.get("/domain-matrix")
def domain_matrix():
    out_dir = Path(__file__).parent / "outputs"
    collided_pairs: dict[str, set[str]] = {}
    for f in sorted(out_dir.glob("batch-*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for fw in data:
                src = fw.get("source_a", "").lower()
                tgt = fw.get("source_b", "").lower()
                if src and tgt:
                    collided_pairs.setdefault(src, set()).add(tgt)
                    collided_pairs.setdefault(tgt, set()).add(src)
        except (json.JSONDecodeError, KeyError):
            continue
    theory_domains = sorted({t.domain for t in _theories})
    coverage = []
    for d in theory_domains:
        partners = collided_pairs.get(d.lower(), set())
        coverage.append({
            "domain": d,
            "collision_count": len(partners),
            "partners": sorted(partners),
        })
    coverage.sort(key=lambda x: -x["collision_count"])
    total_possible = len(theory_domains) * (len(theory_domains) - 1) // 2
    covered = sum(len(c["partners"]) for c in coverage) // 2
    return {
        "total_domains": len(theory_domains),
        "total_possible_pairs": total_possible,
        "pairs_with_collisions": covered,
        "coverage_pct": round(covered / total_possible * 100, 1) if total_possible else 0,
        "domains": coverage,
    }


@app.get("/export")
def export_frameworks(format: str = "json", min_confidence: float = 0.0):
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
    seen = set()
    unique = []
    for fw in all_fw:
        n = fw.get("name", "").strip().lower()
        if n and n not in seen:
            seen.add(n)
            unique.append(fw)
    if min_confidence > 0:
        unique = [fw for fw in unique if fw.get("confidence", 0) >= min_confidence]
    unique.sort(key=lambda fw: -fw.get("confidence", 0))
    if format == "csv":
        buf = io.StringIO()
        cols = ["name", "confidence", "viability", "core_claim",
                "mechanism_borrowed_from", "domain_applied_to",
                "source_a", "source_b", "_batch"]
        writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(unique)
        buf.seek(0)
        return StreamingResponse(
            buf, media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=theory-forge-export.csv"},
        )
    return unique


@app.get("/tag-analysis")
def tag_analysis():
    tag_counts: dict[str, int] = {}
    tag_domains: dict[str, set[str]] = {}
    for t in _theories:
        for tag in t.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            tag_domains.setdefault(tag, set()).add(t.domain)
    tags_sorted = sorted(tag_counts.items(), key=lambda x: -x[1])
    return {
        "total_unique_tags": len(tag_counts),
        "top_tags": [{"tag": t, "count": c, "domains": len(tag_domains[t])} for t, c in tags_sorted[:30]],
        "rare_tags": [{"tag": t, "count": c, "domains": len(tag_domains[t])} for t, c in tags_sorted if c <= 3],
        "tag_domain_spread": [
            {"tag": t, "count": c, "domain_count": len(tag_domains[t]),
             "domains": sorted(tag_domains[t])}
            for t, c in tags_sorted[:50]
        ],
    }


@app.get("/random")
def random_framework(min_confidence: float = 0.6):
    out_dir = Path(__file__).parent / "outputs"
    all_fw = []
    for f in sorted(out_dir.glob("batch-*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for fw in data:
                if fw.get("confidence", 0) >= min_confidence:
                    fw["_batch"] = f.name
                    all_fw.append(fw)
        except (json.JSONDecodeError, KeyError):
            continue
    if not all_fw:
        raise HTTPException(404, "No frameworks match that confidence threshold.")
    return random.choice(all_fw)


@app.get("/domain-network")
def domain_network():
    out_dir = Path(__file__).parent / "outputs"
    edges: dict[tuple[str, str], int] = {}
    for f in sorted(out_dir.glob("batch-*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for fw in data:
                src = fw.get("mechanism_source", fw.get("mechanism_borrowed_from", fw.get("source_a", ""))).strip()
                tgt = fw.get("target_domain", fw.get("domain_applied_to", fw.get("source_b", ""))).strip()
                if src and tgt:
                    key = tuple(sorted([src.lower(), tgt.lower()]))
                    edges[key] = edges.get(key, 0) + 1
        except (json.JSONDecodeError, KeyError):
            continue
    nodes = set()
    edge_list = []
    for (a, b), weight in sorted(edges.items(), key=lambda x: -x[1]):
        nodes.add(a)
        nodes.add(b)
        edge_list.append({"source": a, "target": b, "weight": weight})
    return {
        "nodes": sorted(nodes),
        "edges": edge_list,
        "node_count": len(nodes),
        "edge_count": len(edge_list),
    }


def _load_all_frameworks() -> list[dict]:
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
    return all_fw


def _unique_frameworks(all_fw: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for fw in all_fw:
        n = fw.get("name", "").strip().lower()
        if n and n not in seen:
            seen.add(n)
            unique.append(fw)
    return unique


# ---- Research Assistant: Semantic Search ----

_fw_embeddings = None

def _ensure_fw_embeddings():
    global _fw_embeddings
    if _fw_embeddings is not None:
        return _fw_embeddings
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return None
    all_fw = _unique_frameworks(_load_all_frameworks())
    texts = []
    for fw in all_fw:
        text = " ".join(filter(None, [
            fw.get("name", ""),
            fw.get("core_claim", ""),
            fw.get("mechanism", ""),
            fw.get("application", ""),
            fw.get("prediction", ""),
            fw.get("mechanism_borrowed_from", fw.get("mechanism_source", "")),
            fw.get("domain_applied_to", fw.get("target_domain", "")),
        ]))
        texts.append(text)
    model = SentenceTransformer(forge.EMBED_MODEL)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    _fw_embeddings = {"frameworks": all_fw, "vectors": vecs, "model": model}
    return _fw_embeddings


@app.get("/semantic-search")
def semantic_search(q: str = "", limit: int = 20):
    if not q or len(q) < 2:
        return []
    emb = _ensure_fw_embeddings()
    if emb is None:
        return search_frameworks(q=q, limit=limit)
    import numpy as np
    q_vec = emb["model"].encode([q], normalize_embeddings=True)
    scores = (emb["vectors"] @ q_vec.T).flatten()
    top_idx = np.argsort(-scores)[:limit]
    results = []
    for i in top_idx:
        if scores[i] < 0.15:
            break
        fw = dict(emb["frameworks"][i])
        fw["_similarity"] = round(float(scores[i]), 3)
        results.append(fw)
    return results


# ---- Research Assistant: Recommendation Engine ----

@app.get("/recommend")
def recommend_collisions(limit: int = 20, strategy: str = "coverage"):
    """Suggest which domain pairs to collide next.

    Strategies:
    - coverage: prioritize domains with fewest collisions
    - diversity: prioritize pairs with high tag diversity (different structural patterns)
    - bridge: find domains that could bridge disconnected clusters
    """
    all_fw = _load_all_frameworks()
    collided_pairs: set[frozenset[str]] = set()
    domain_collision_count: Counter = Counter()
    for fw in all_fw:
        src = fw.get("mechanism_source", fw.get("mechanism_borrowed_from", fw.get("source_a", ""))).strip().lower()
        tgt = fw.get("target_domain", fw.get("domain_applied_to", fw.get("source_b", ""))).strip().lower()
        if src and tgt:
            collided_pairs.add(frozenset({src, tgt}))
            domain_collision_count[src] += 1
            domain_collision_count[tgt] += 1

    theory_domains = sorted({t.domain for t in _theories})
    domain_theories = defaultdict(list)
    for t in _theories:
        domain_theories[t.domain].append(t)
    domain_tags = {}
    for d in theory_domains:
        tags = set()
        for t in domain_theories[d]:
            tags.update(t.tags)
        domain_tags[d] = tags

    recommendations = []

    if strategy == "coverage":
        underexplored = [(d, domain_collision_count.get(d.lower(), 0)) for d in theory_domains]
        underexplored.sort(key=lambda x: x[1])
        cold_domains = [d for d, c in underexplored[:30]]
        for i, da in enumerate(cold_domains):
            for db in cold_domains[i+1:]:
                if frozenset({da.lower(), db.lower()}) not in collided_pairs:
                    overlap = len(domain_tags.get(da, set()) & domain_tags.get(db, set()))
                    recommendations.append({
                        "domain_a": da, "domain_b": db,
                        "strategy": "coverage",
                        "reason": f"Both under-explored ({domain_collision_count.get(da.lower(),0)} and {domain_collision_count.get(db.lower(),0)} collisions)",
                        "tag_overlap": overlap,
                        "score": 100 - domain_collision_count.get(da.lower(), 0) - domain_collision_count.get(db.lower(), 0) + overlap * 2,
                    })

    elif strategy == "diversity":
        for i, da in enumerate(theory_domains):
            for db in theory_domains[i+1:]:
                if frozenset({da.lower(), db.lower()}) in collided_pairs:
                    continue
                tags_a = domain_tags.get(da, set())
                tags_b = domain_tags.get(db, set())
                if not tags_a or not tags_b:
                    continue
                union = len(tags_a | tags_b)
                intersection = len(tags_a & tags_b)
                diversity = (union - intersection) / union if union else 0
                if intersection >= 1 and diversity >= 0.5:
                    recommendations.append({
                        "domain_a": da, "domain_b": db,
                        "strategy": "diversity",
                        "reason": f"High tag diversity ({diversity:.0%}) with {intersection} shared tags",
                        "tag_overlap": intersection,
                        "tag_diversity": round(diversity, 3),
                        "score": diversity * 100 + intersection * 5,
                    })

    elif strategy == "bridge":
        adjacency: dict[str, set[str]] = defaultdict(set)
        for fw in all_fw:
            src = fw.get("mechanism_source", fw.get("mechanism_borrowed_from", fw.get("source_a", ""))).strip().lower()
            tgt = fw.get("target_domain", fw.get("domain_applied_to", fw.get("source_b", ""))).strip().lower()
            if src and tgt:
                adjacency[src].add(tgt)
                adjacency[tgt].add(src)
        for i, da in enumerate(theory_domains):
            neighbors_a = adjacency.get(da.lower(), set())
            for db in theory_domains[i+1:]:
                if frozenset({da.lower(), db.lower()}) in collided_pairs:
                    continue
                neighbors_b = adjacency.get(db.lower(), set())
                shared_neighbors = neighbors_a & neighbors_b
                if not shared_neighbors and (neighbors_a or neighbors_b):
                    overlap = len(domain_tags.get(da, set()) & domain_tags.get(db, set()))
                    if overlap >= 1:
                        recommendations.append({
                            "domain_a": da, "domain_b": db,
                            "strategy": "bridge",
                            "reason": f"Would bridge disconnected clusters ({len(neighbors_a)} and {len(neighbors_b)} neighbors, 0 shared)",
                            "tag_overlap": overlap,
                            "score": (len(neighbors_a) + len(neighbors_b)) * 2 + overlap * 10,
                        })

    recommendations.sort(key=lambda x: -x.get("score", 0))
    return recommendations[:limit]


# ---- Research Assistant: Framework Chain/Exploration ----

@app.get("/chain/{name}")
def framework_chain(name: str, depth: int = 3):
    """Follow a chain of related frameworks across domains.
    Starting from a named framework, find related frameworks by shared
    mechanism source or target domain, building a multi-hop exploration path."""
    all_fw = _load_all_frameworks()
    name_lower = name.lower().strip()
    start = None
    for fw in all_fw:
        if fw.get("name", "").lower().strip() == name_lower:
            start = fw
            break
    if not start:
        raise HTTPException(404, f"Framework '{name}' not found.")

    chain = [start]
    visited = {name_lower}

    for _ in range(depth):
        current = chain[-1]
        mech = current.get("mechanism_borrowed_from", current.get("mechanism_source", "")).lower()
        dom = current.get("domain_applied_to", current.get("target_domain", "")).lower()
        best = None
        best_score = -1
        for fw in all_fw:
            fn = fw.get("name", "").lower().strip()
            if fn in visited:
                continue
            fw_mech = fw.get("mechanism_borrowed_from", fw.get("mechanism_source", "")).lower()
            fw_dom = fw.get("domain_applied_to", fw.get("target_domain", "")).lower()
            score = 0
            link_type = ""
            if dom and dom in fw_mech:
                score = fw.get("confidence", 0) + 0.3
                link_type = "domain→mechanism"
            elif mech and mech in fw_dom:
                score = fw.get("confidence", 0) + 0.2
                link_type = "mechanism→domain"
            elif dom and dom in fw_dom:
                score = fw.get("confidence", 0) + 0.1
                link_type = "shared domain"
            if score > best_score:
                best_score = score
                best = fw
                best["_link_type"] = link_type
        if best:
            visited.add(best.get("name", "").lower().strip())
            chain.append(best)
        else:
            break

    return {
        "start": name,
        "chain_length": len(chain),
        "frameworks": [
            {
                "name": fw.get("name", "?"),
                "confidence": fw.get("confidence", 0),
                "mechanism_source": fw.get("mechanism_borrowed_from", fw.get("mechanism_source", "")),
                "target_domain": fw.get("domain_applied_to", fw.get("target_domain", "")),
                "core_claim": fw.get("core_claim", fw.get("application", "")),
                "link_type": fw.get("_link_type", "origin"),
                "_batch": fw.get("_batch", ""),
            }
            for fw in chain
        ],
    }


# ---- Research Assistant: Synthesis Report ----

@app.get("/synthesis")
def synthesis_report(domain: str = "", top_n: int = 10, format: str = "markdown"):
    """Generate a synthesis report summarizing the top N frameworks
    in a chosen domain (or across all domains)."""
    all_fw = _unique_frameworks(_load_all_frameworks())
    if domain:
        d = domain.lower()
        filtered = [fw for fw in all_fw
                    if d in fw.get("domain_applied_to", "").lower()
                    or d in fw.get("target_domain", "").lower()
                    or d in fw.get("mechanism_borrowed_from", "").lower()
                    or d in fw.get("mechanism_source", "").lower()]
    else:
        filtered = all_fw

    filtered.sort(key=lambda fw: -fw.get("confidence", 0))
    top = filtered[:top_n]
    if not top:
        raise HTTPException(404, f"No frameworks found for domain '{domain}'.")

    viability_counts = Counter(fw.get("viability", "unknown") for fw in top)
    avg_conf = sum(fw.get("confidence", 0) for fw in top) / len(top) if top else 0
    domains_involved = set()
    for fw in top:
        domains_involved.add(fw.get("mechanism_borrowed_from", fw.get("mechanism_source", "")))
        domains_involved.add(fw.get("domain_applied_to", fw.get("target_domain", "")))
    domains_involved.discard("")

    report_lines = [
        f"# Theory Forge Synthesis Report{': ' + domain.title() if domain else ''}",
        "",
        f"**Generated from {len(filtered)} frameworks** | Top {len(top)} shown",
        f"**Average confidence:** {avg_conf:.3f} | "
        f"**Promising:** {viability_counts.get('promising', 0)} | "
        f"**Speculative:** {viability_counts.get('speculative', 0)}",
        f"**Domains involved:** {', '.join(sorted(domains_involved))}",
        "",
        "---",
        "",
    ]

    for i, fw in enumerate(top, 1):
        name = fw.get("name", "Unnamed")
        conf = fw.get("confidence", 0)
        viab = fw.get("viability", "?")
        mech = fw.get("mechanism_borrowed_from", fw.get("mechanism_source", "?"))
        tgt = fw.get("domain_applied_to", fw.get("target_domain", "?"))
        claim = fw.get("core_claim", fw.get("application", ""))
        preds = fw.get("falsifiable_predictions", [])
        if isinstance(preds, list) and preds:
            pred_block = "\n".join(f"   - {p}" for p in preds[:3])
        else:
            pred_block = f"   - {fw.get('prediction', 'N/A')}"

        report_lines.extend([
            f"## {i}. {name}",
            f"**Confidence:** {conf:.2f} | **Viability:** {viab} | "
            f"**{mech}** → **{tgt}**",
            "",
            claim,
            "",
            f"**Key predictions:**",
            pred_block,
            "",
        ])

    report_lines.extend([
        "---",
        "",
        f"*Report generated by Theory Forge v0.6 | "
        f"{len(_theories)} seed theories across {len({t.domain for t in _theories})} domains*",
    ])

    report = "\n".join(report_lines)

    if format == "html":
        viab_data = json.dumps(dict(viability_counts))
        conf_values = json.dumps([round(fw.get("confidence", 0), 3) for fw in top])
        names_data = json.dumps([fw.get("name", "?")[:30] for fw in top])
        html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Synthesis Report{': ' + domain.title() if domain else ''}</title>
<style>
body {{ font-family: ui-monospace, Consolas, monospace; background:#0e1116; color:#e6edf3; padding:32px; max-width:900px; margin:auto; }}
h1 {{ color:#7ee787; }} h2 {{ color:#79c0ff; border-bottom:1px solid #30363d; padding-bottom:6px; }}
.stats {{ display:flex; gap:16px; flex-wrap:wrap; margin:16px 0; }}
.stat {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; text-align:center; flex:1; min-width:120px; }}
.stat-val {{ font-size:28px; font-weight:bold; color:#7ee787; }}
.stat-lbl {{ font-size:11px; color:#8b949e; }}
canvas {{ background:#161b22; border:1px solid #30363d; border-radius:8px; }}
.fw {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin:12px 0; }}
.fw h3 {{ color:#7ee787; margin:0 0 8px; }}
.fw .meta {{ color:#8b949e; font-size:12px; }}
.fw .claim {{ margin:8px 0; line-height:1.6; }}
.badge {{ display:inline-block; padding:2px 6px; border-radius:4px; font-size:10px; font-weight:bold; }}
.badge-promising {{ background:#1a3a1a; color:#7ee787; }}
.badge-speculative {{ background:#3a2a1a; color:#f0b84a; }}
.badge-incoherent {{ background:#3a1a1a; color:#ff7b72; }}
.pred {{ color:#d2a8ff; font-size:13px; }}
</style></head><body>
<h1>Theory Forge Synthesis Report{': ' + domain.title() if domain else ''}</h1>
<div class="stats">
  <div class="stat"><div class="stat-val">{len(filtered)}</div><div class="stat-lbl">frameworks analyzed</div></div>
  <div class="stat"><div class="stat-val">{len(top)}</div><div class="stat-lbl">top shown</div></div>
  <div class="stat"><div class="stat-val">{avg_conf:.3f}</div><div class="stat-lbl">avg confidence</div></div>
  <div class="stat"><div class="stat-val">{len(domains_involved)}</div><div class="stat-lbl">domains involved</div></div>
</div>
<div style="display:flex;gap:16px;flex-wrap:wrap;margin:24px 0">
  <div><canvas id="viabChart" width="250" height="180"></canvas></div>
  <div><canvas id="confChart" width="500" height="180"></canvas></div>
</div>
"""
        for i, fw in enumerate(top, 1):
            name = fw.get("name", "Unnamed")
            conf = fw.get("confidence", 0)
            viab = fw.get("viability", "?")
            badge_cls = "badge-promising" if viab == "promising" else "badge-speculative" if viab == "speculative" else "badge-incoherent"
            mech = fw.get("mechanism_borrowed_from", fw.get("mechanism_source", "?"))
            tgt = fw.get("domain_applied_to", fw.get("target_domain", "?"))
            claim = fw.get("core_claim", fw.get("application", ""))
            preds = fw.get("falsifiable_predictions", [])
            pred_html = ""
            if isinstance(preds, list) and preds:
                pred_html = "".join(f'<div class="pred">• {p}</div>' for p in preds[:3])
            else:
                pred_html = f'<div class="pred">• {fw.get("prediction", "N/A")}</div>'
            html += f"""
<div class="fw">
  <h3>{i}. {name} <span class="badge {badge_cls}">{viab}</span></h3>
  <div class="meta">Confidence: <strong>{conf:.2f}</strong> | {mech} → {tgt}</div>
  <div class="claim">{claim}</div>
  <div style="margin-top:8px"><strong style="color:#79c0ff;font-size:12px">KEY PREDICTIONS</strong></div>
  {pred_html}
</div>"""

        html += f"""
<script>
const viabData = {viab_data};
const confValues = {conf_values};
const names = {names_data};
// Viability pie chart
const vc = document.getElementById('viabChart').getContext('2d');
const colors = {{'promising':'#7ee787','speculative':'#f0b84a','incoherent':'#ff7b72','unknown':'#8b949e'}};
const total = Object.values(viabData).reduce((a,b)=>a+b,0);
let startAngle = -Math.PI/2;
Object.entries(viabData).forEach(([k,v]) => {{
  const slice = (v/total)*Math.PI*2;
  vc.beginPath(); vc.moveTo(90,90); vc.arc(90,90,75,startAngle,startAngle+slice);
  vc.fillStyle = colors[k]||'#8b949e'; vc.fill();
  const mid = startAngle+slice/2;
  vc.fillStyle='#e6edf3'; vc.font='11px monospace'; vc.textAlign='center';
  vc.fillText(k+' '+v, 90+Math.cos(mid)*50, 90+Math.sin(mid)*50);
  startAngle += slice;
}});
// Confidence bar chart
const cc = document.getElementById('confChart').getContext('2d');
const barW = Math.min(40, (480/confValues.length)-4);
confValues.forEach((v,i) => {{
  const h = v*150; const x = 10+i*(barW+4);
  cc.fillStyle = v>=0.7?'#7ee787':v>=0.5?'#f0b84a':'#ff7b72';
  cc.fillRect(x,170-h,barW,h);
  cc.fillStyle='#8b949e'; cc.font='8px monospace'; cc.textAlign='center';
  cc.save(); cc.translate(x+barW/2,172); cc.rotate(-Math.PI/4);
  cc.fillText(names[i],0,0); cc.restore();
}});
</script>
<div style="margin-top:32px;color:#8b949e;font-size:11px">
Report generated by Theory Forge v0.7
</div></body></html>"""
        return HTMLResponse(html)

    return StreamingResponse(
        io.StringIO(report),
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=synthesis-{domain or 'all'}.md"},
    )


# ---- Visualization: Domain Heatmap ----

@app.get("/domain-heatmap")
def domain_heatmap():
    """Jaccard similarity between all domain tag sets — data for a heatmap."""
    theory_domains = sorted({t.domain for t in _theories})
    domain_tags: dict[str, set[str]] = {}
    for t in _theories:
        domain_tags.setdefault(t.domain, set()).update(t.tags)

    rows = []
    for da in theory_domains:
        row = []
        tags_a = domain_tags.get(da, set())
        for db in theory_domains:
            tags_b = domain_tags.get(db, set())
            union = len(tags_a | tags_b)
            row.append(round(len(tags_a & tags_b) / union, 3) if union else 0)
        rows.append(row)
    return {"domains": theory_domains, "matrix": rows}


# ---- Visualization: Theory Genealogy ----

@app.get("/genealogy")
def genealogy():
    """Map which seed theories produced which frameworks."""
    all_fw = _load_all_frameworks()
    theory_names = {t.name.lower(): t.to_dict() for t in _theories}
    tree: dict[str, list] = {}
    for fw in all_fw:
        for src_key in ("source_a", "source_b"):
            src = fw.get(src_key, "").strip()
            if src and src.lower() in theory_names:
                tree.setdefault(src, []).append({
                    "name": fw.get("name", "?"),
                    "confidence": fw.get("confidence", 0),
                    "viability": fw.get("viability", "?"),
                    "partner": fw.get("source_b" if src_key == "source_a" else "source_a", ""),
                })
    top_parents = sorted(tree.items(), key=lambda x: -len(x[1]))[:60]
    return {
        "total_parent_theories": len(tree),
        "parents": [
            {"theory": name, "offspring_count": len(children),
             "offspring": sorted(children, key=lambda c: -c["confidence"])[:10]}
            for name, children in top_parents
        ],
    }


# ---- Visualization: Discovery Timeline ----

@app.get("/timeline")
def discovery_timeline():
    """When each domain was first explored across collision batches."""
    out_dir = Path(__file__).parent / "outputs"
    domain_first: dict[str, dict] = {}
    for f in sorted(out_dir.glob("batch-*.json")):
        batch_num_match = re.search(r"batch-(\d+)", f.name)
        if not batch_num_match:
            continue
        batch_num = int(batch_num_match.group(1))
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            continue
        for fw in data:
            for key in ("mechanism_borrowed_from", "mechanism_source", "domain_applied_to", "target_domain", "source_a", "source_b"):
                d = fw.get(key, "").strip()
                if d and d.lower() not in domain_first:
                    domain_first[d.lower()] = {
                        "domain": d,
                        "first_batch": batch_num,
                        "first_framework": fw.get("name", "?"),
                        "batch_file": f.name,
                    }
    timeline = sorted(domain_first.values(), key=lambda x: x["first_batch"])
    batches_with_new = defaultdict(list)
    for entry in timeline:
        batches_with_new[entry["first_batch"]].append(entry["domain"])
    return {
        "total_domains_discovered": len(timeline),
        "timeline": timeline,
        "by_batch": [
            {"batch": b, "new_domains": ds}
            for b, ds in sorted(batches_with_new.items())
        ],
    }


# ---- Compare Frameworks ----

@app.get("/compare")
def compare_frameworks(names: str = ""):
    """Compare 2-3 frameworks side by side."""
    if not names:
        raise HTTPException(400, "Provide comma-separated framework names via ?names=")
    name_list = [n.strip() for n in names.split(",") if n.strip()][:3]
    if len(name_list) < 2:
        raise HTTPException(400, "Need at least 2 framework names to compare.")
    all_fw = _load_all_frameworks()
    fw_by_name: dict[str, dict] = {}
    for fw in all_fw:
        fn = fw.get("name", "").strip().lower()
        if fn and fn not in fw_by_name:
            fw_by_name[fn] = fw
    results = []
    for name in name_list:
        fw = fw_by_name.get(name.lower())
        if not fw:
            raise HTTPException(404, f"Framework '{name}' not found.")
        results.append(fw)

    all_tags = set()
    fw_tags = []
    for fw in results:
        tags = set()
        mech = fw.get("mechanism_borrowed_from", fw.get("mechanism_source", ""))
        tgt = fw.get("domain_applied_to", fw.get("target_domain", ""))
        for t in _theories:
            if t.name.lower() in fw.get("source_a", "").lower() or t.name.lower() in fw.get("source_b", "").lower():
                tags.update(t.tags)
            if t.domain.lower() == mech.lower() or t.domain.lower() == tgt.lower():
                tags.update(t.tags)
        fw_tags.append(tags)
        all_tags.update(tags)

    shared_tags = fw_tags[0]
    for t in fw_tags[1:]:
        shared_tags = shared_tags & t
    unique_tags = [t - shared_tags for t in fw_tags]

    return {
        "frameworks": results,
        "shared_tags": sorted(shared_tags),
        "unique_tags": [sorted(u) for u in unique_tags],
        "overlap_ratio": round(len(shared_tags) / len(all_tags), 3) if all_tags else 0,
        "structural_comparison": {
            "fields": ["name", "confidence", "viability", "mechanism_borrowed_from",
                        "domain_applied_to", "core_claim", "prediction"],
            "values": [
                [fw.get(f, fw.get({"mechanism_borrowed_from": "mechanism_source",
                                    "domain_applied_to": "target_domain"}.get(f, f), ""))
                 for f in ["name", "confidence", "viability", "mechanism_borrowed_from",
                           "domain_applied_to", "core_claim", "prediction"]]
                for fw in results
            ],
        },
    }


# ---- Surprise Chain ----

@app.get("/surprise-chain")
def surprise_chain(depth: int = 10):
    """Pick a random high-confidence framework and build a 10-hop exploration path."""
    all_fw = _unique_frameworks(_load_all_frameworks())
    high_conf = [fw for fw in all_fw if fw.get("confidence", 0) >= 0.65]
    if not high_conf:
        high_conf = all_fw
    start = random.choice(high_conf)
    start_name = start.get("name", "")
    chain = [start]
    visited = {start_name.lower().strip()}

    for _ in range(depth):
        current = chain[-1]
        mech = current.get("mechanism_borrowed_from", current.get("mechanism_source", "")).lower()
        dom = current.get("domain_applied_to", current.get("target_domain", "")).lower()
        candidates = []
        for fw in all_fw:
            fn = fw.get("name", "").lower().strip()
            if fn in visited:
                continue
            fw_mech = fw.get("mechanism_borrowed_from", fw.get("mechanism_source", "")).lower()
            fw_dom = fw.get("domain_applied_to", fw.get("target_domain", "")).lower()
            score = 0
            link = ""
            if dom and dom in fw_mech:
                score = fw.get("confidence", 0) + 0.3
                link = "domain→mechanism"
            elif mech and mech in fw_dom:
                score = fw.get("confidence", 0) + 0.2
                link = "mechanism→domain"
            elif dom and dom in fw_dom:
                score = fw.get("confidence", 0) + 0.1
                link = "shared domain"
            elif mech and mech in fw_mech:
                score = fw.get("confidence", 0) + 0.05
                link = "shared mechanism"
            if score > 0:
                candidates.append((fw, score, link))
        if not candidates:
            break
        candidates.sort(key=lambda x: -x[1])
        top_picks = candidates[:5]
        pick, _, link_type = random.choice(top_picks)
        pick["_link_type"] = link_type
        visited.add(pick.get("name", "").lower().strip())
        chain.append(pick)

    return {
        "chain_length": len(chain),
        "frameworks": [
            {
                "name": fw.get("name", "?"),
                "confidence": fw.get("confidence", 0),
                "viability": fw.get("viability", "?"),
                "mechanism_source": fw.get("mechanism_borrowed_from", fw.get("mechanism_source", "")),
                "target_domain": fw.get("domain_applied_to", fw.get("target_domain", "")),
                "core_claim": fw.get("core_claim", fw.get("application", "")),
                "link_type": fw.get("_link_type", "origin"),
            }
            for fw in chain
        ],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
