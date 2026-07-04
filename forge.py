"""
Theory Forge — core engine.

Karpathy-style: one file, readable top-to-bottom.

Pipeline:
    load_theories()                -> list[Theory]
    load_ledger()                  -> list[TriedCollision]
    fingerprint(theory)            -> frozenset[str]   (structural tag set)
    tag_similarity(a, b)           -> float in [0,1]   (Jaccard over tags)
    embed_theories(theories)       -> dict[id -> np.ndarray]  (optional)
    embed_similarity(a, b, vecs)   -> float in [-1,1]  (cosine over embeddings)
    structural_similarity(a, b, *) -> float            (blended or tag-only)
    candidate_pairs(...)           -> ranked novel cross-domain pairs
    collide(a, b, client)          -> NewFramework via Claude

A "collision" takes the *mechanism* of one theory and applies it in the
*domain* of another. Pairs are scored by structural overlap (tags + optional
embeddings) and filtered against `tried.json` so we surface NOVEL combinations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).parent
THEORIES_PATH = ROOT / "theories.json"
LEDGER_PATH = ROOT / "tried.json"
MODEL_DEFAULT = os.environ.get("THEORY_FORGE_MODEL", "claude-sonnet-4-6")
EMBED_MODEL = os.environ.get("THEORY_FORGE_EMBED_MODEL", "all-MiniLM-L6-v2")


@dataclass(frozen=True)
class Theory:
    id: str
    name: str
    domain: str
    unit: str
    variation: str
    selection: str
    fitness: str
    boundary: str
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, d: dict) -> "Theory":
        return cls(
            id=d["id"], name=d["name"], domain=d["domain"],
            unit=d["unit"], variation=d["variation"], selection=d["selection"],
            fitness=d["fitness"], boundary=d["boundary"],
            tags=tuple(d.get("tags", [])),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d

    def structural_text(self) -> str:
        """Concatenation of mechanism slots — what embeddings should see."""
        return (
            f"unit: {self.unit}. variation: {self.variation}. "
            f"selection: {self.selection}. fitness: {self.fitness}. "
            f"boundary: {self.boundary}. tags: {', '.join(self.tags)}."
        )


@dataclass(frozen=True)
class TriedCollision:
    a_id: str
    b_id: str
    status: str          # "published" | "failed" | "speculative"
    label: str
    note: str = ""

    def key(self) -> frozenset[str]:
        return frozenset({self.a_id, self.b_id})


def load_theories(path: Path = THEORIES_PATH) -> list[Theory]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Theory.from_dict(t) for t in raw]


def load_ledger(path: Path = LEDGER_PATH) -> list[TriedCollision]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TriedCollision(**r) for r in raw]


# ---------- similarity ----------

def fingerprint(t: Theory) -> frozenset[str]:
    return frozenset(tag.lower().strip() for tag in t.tags)


def tag_similarity(a: Theory, b: Theory) -> float:
    fa, fb = fingerprint(a), fingerprint(b)
    if not fa or not fb:
        return 0.0
    return len(fa & fb) / len(fa | fb)


def embed_theories(theories: list[Theory]) -> dict[str, "object"] | None:
    """
    Return {id: vector} or None if sentence-transformers isn't installed.
    The model downloads on first call (~90MB for all-MiniLM-L6-v2).
    Call this once and reuse the dict — embedding is the expensive part.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np  # noqa: F401  (used by ST)
    except ImportError:
        return None
    model = SentenceTransformer(EMBED_MODEL)
    texts = [t.structural_text() for t in theories]
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return {t.id: v for t, v in zip(theories, vecs)}


def embed_similarity(a: Theory, b: Theory, vecs: dict) -> float:
    va, vb = vecs.get(a.id), vecs.get(b.id)
    if va is None or vb is None:
        return 0.0
    # vectors are L2-normalized -> dot = cosine
    return float((va * vb).sum())


def structural_similarity(
    a: Theory, b: Theory, vecs: dict | None = None, alpha: float = 0.5
) -> float:
    """
    Blend tag Jaccard and embedding cosine. alpha=0 -> tags only.
    Embedding cosine is mapped from [-1,1] to [0,1] before blending.
    """
    s_tag = tag_similarity(a, b)
    if vecs is None or alpha == 0.0:
        return s_tag
    s_emb = (embed_similarity(a, b, vecs) + 1.0) / 2.0
    return (1 - alpha) * s_tag + alpha * s_emb


# ---------- pair search + novelty ----------

def tried_keys(ledger: list[TriedCollision]) -> set[frozenset[str]]:
    return {tc.key() for tc in ledger}


def candidate_pairs(
    theories: Iterable[Theory],
    min_sim: float = 0.25,
    max_sim: float = 0.92,
    cross_domain_only: bool = True,
    vecs: dict | None = None,
    alpha: float = 0.5,
    exclude_tried: set[frozenset[str]] | None = None,
) -> list[tuple[Theory, Theory, float]]:
    """
    Cross-domain pairs with shared structure, ranked by similarity, with
    historical collisions excluded so the engine surfaces NOVEL pairs.
    """
    exclude_tried = exclude_tried or set()
    out = []
    for a, b in combinations(theories, 2):
        if cross_domain_only and a.domain == b.domain:
            continue
        if frozenset({a.id, b.id}) in exclude_tried:
            continue
        s = structural_similarity(a, b, vecs=vecs, alpha=alpha)
        if min_sim <= s <= max_sim:
            out.append((a, b, s))
    out.sort(key=lambda r: -r[2])
    return out


def best_pair_between(
    theories: list[Theory], domain_a: str, domain_b: str,
    vecs: dict | None = None, alpha: float = 0.5,
    exclude_tried: set[frozenset[str]] | None = None,
) -> tuple[Theory, Theory, float] | None:
    da, db = domain_a.lower(), domain_b.lower()
    pool_a = [t for t in theories if t.domain.lower() == da]
    pool_b = [t for t in theories if t.domain.lower() == db]
    if not pool_a or not pool_b:
        return None
    exclude_tried = exclude_tried or set()
    best = None
    for a in pool_a:
        for b in pool_b:
            if frozenset({a.id, b.id}) in exclude_tried:
                continue
            s = structural_similarity(a, b, vecs=vecs, alpha=alpha)
            if best is None or s > best[2]:
                best = (a, b, s)
    return best


# ---------- LLM synthesis ----------

COLLIDE_SYSTEM = """You are Theory Forge. You take the *mechanism* of one theory \
and apply it in the *domain* of another to propose a new framework.

Rules:
- The mechanism must transfer structurally (unit / variation / selection / fitness).
- The output must include at least one falsifiable prediction.
- If the collision is incoherent, say so in `viability` and lower confidence.
- No hedging prose. Return JSON only, matching the schema in the user message."""


def collide_prompt(a: Theory, b: Theory, prior: TriedCollision | None = None) -> str:
    schema = {
        "name": "string — punchy title for the new framework",
        "core_claim": "string — one sentence",
        "mechanism_borrowed_from": a.name,
        "domain_applied_to": b.domain,
        "mapped_components": {
            "unit": "what plays the role of the unit in the new domain",
            "variation": "where variation comes from",
            "selection": "what selection pressure looks like",
            "fitness": "what is being maximized/minimized",
            "boundary": "scope and assumptions",
        },
        "falsifiable_predictions": ["string", "string"],
        "viability": "promising | speculative | incoherent",
        "confidence": "float 0..1",
        "notes": "string — anything the schema didn't capture (optional, <=200 chars)",
    }
    prior_block = ""
    if prior is not None:
        prior_block = (
            "\n\nNOTE: A related collision has been attempted before:\n"
            f"  {prior.label} [{prior.status}] — {prior.note}\n"
            "Propose something distinct from that, or explicitly explain how "
            "yours differs in `notes`.\n"
        )
    return (
        "Borrow the mechanism of THEORY A and apply it in the domain of THEORY B.\n\n"
        f"THEORY A ({a.domain}) — {a.name}\n"
        f"  unit: {a.unit}\n  variation: {a.variation}\n"
        f"  selection: {a.selection}\n  fitness: {a.fitness}\n"
        f"  boundary: {a.boundary}\n\n"
        f"THEORY B ({b.domain}) — {b.name}\n"
        f"  unit: {b.unit}\n  variation: {b.variation}\n"
        f"  selection: {b.selection}\n  fitness: {b.fitness}\n"
        f"  boundary: {b.boundary}\n"
        f"{prior_block}\n"
        "Return ONLY JSON matching this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def collide(
    a: Theory, b: Theory, client, model: str = MODEL_DEFAULT,
    prior: TriedCollision | None = None,
) -> dict:
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=COLLIDE_SYSTEM,
        messages=[{"role": "user", "content": collide_prompt(a, b, prior=prior)}],
    )
    text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
    return _extract_json(text)


def collide3_prompt(a: Theory, b: Theory, c: Theory) -> str:
    schema = {
        "name": "string — punchy title for the new framework",
        "core_claim": "string — one sentence",
        "mechanism_borrowed_from": f"{a.name} + {b.name}",
        "domain_applied_to": c.domain,
        "mapped_components": {
            "unit": "what plays the role of the unit in the target domain",
            "variation": "where variation comes from",
            "selection": "what selection pressure looks like",
            "fitness": "what is being maximized/minimized",
            "boundary": "scope and assumptions",
        },
        "falsifiable_predictions": ["string", "string"],
        "viability": "promising | speculative | incoherent",
        "confidence": "float 0..1",
        "notes": "string — how the two mechanisms interact (optional, <=200 chars)",
    }

    def slots(t: Theory) -> str:
        return (
            f"  unit: {t.unit}\n  variation: {t.variation}\n"
            f"  selection: {t.selection}\n  fitness: {t.fitness}\n"
            f"  boundary: {t.boundary}\n"
        )

    return (
        "Fuse the mechanisms of THEORY A and THEORY B, then apply the combined "
        "mechanism in the domain of THEORY C. The two borrowed mechanisms should "
        "interact — one may supply variation while the other supplies selection, "
        "or they may operate at different scales. Name that interaction.\n\n"
        f"THEORY A ({a.domain}) — {a.name}\n{slots(a)}\n"
        f"THEORY B ({b.domain}) — {b.name}\n{slots(b)}\n"
        f"TARGET DOMAIN — THEORY C ({c.domain}) — {c.name}\n{slots(c)}\n"
        "Return ONLY JSON matching this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def collide3(a: Theory, b: Theory, c: Theory, client,
             model: str = MODEL_DEFAULT) -> dict:
    msg = client.messages.create(
        model=model,
        max_tokens=1200,
        system=COLLIDE_SYSTEM,
        messages=[{"role": "user", "content": collide3_prompt(a, b, c)}],
    )
    text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
    return _extract_json(text)


def find_prior(a: Theory, b: Theory, ledger: list[TriedCollision]) -> TriedCollision | None:
    key = frozenset({a.id, b.id})
    for tc in ledger:
        if tc.key() == key:
            return tc
    return None


def _extract_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip().rstrip("`").strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model output: {text[:200]!r}")
    return json.loads(s[start : end + 1])


if __name__ == "__main__":
    ths = load_theories()
    ledger = load_ledger()
    tried = tried_keys(ledger)
    print(f"Loaded {len(ths)} theories, {len(ledger)} known collisions.")
    print("\nTop NOVEL cross-domain pairs (tag similarity only):")
    for a, b, s in candidate_pairs(ths, exclude_tried=tried)[:15]:
        print(f"  {s:.2f}  {a.name} ({a.domain})  ×  {b.name} ({b.domain})")
