# Theory Forge

A combinatorial theory engine. Borrow a mechanism from one field, apply it in the domain of another, see what falls out.

The white space (per the PRD): no one has built a tool that maps the *structure* of theories (not their content) and then collides them across domains. This is the smallest thing that does that.

## How it works

1. **Decompose** — every theory is broken into `unit / variation / selection / fitness / boundary / tags`. See [theories.json](theories.json) for the seed (~110 theories across biology, physics, theology, mycology, ML, theology, finance, music theory, mathematics, etc).
2. **Fingerprint** — tag-set Jaccard *plus* optional sentence-transformer embeddings of the mechanism slots. Blend controlled by `alpha`.
3. **Filter for novelty** — [tried.json](tried.json) is a ledger of collisions already published (e.g. Darwin × markets → evolutionary economics) or known to fail. The engine excludes these so what you see is white space.
4. **Collide** — Claude maps the mechanism into the new domain and emits a falsifiable prediction. If a prior collision exists on the same pair, it's surfaced as context so the model proposes something distinct.

That's the whole loop. ~300 lines.

## Files

- [forge.py](forge.py) — core: loading, fingerprinting, tag + embedding similarity, ledger-aware novelty filter, LLM synthesis. Single file, runnable.
- [server.py](server.py) — FastAPI: `/theories`, `/domains`, `/ledger`, `/pairs`, `/collide`, `/collide_domains`, `/`.
- [theories.json](theories.json) — hand-decomposed seed dataset (~110 entries).
- [tried.json](tried.json) — known-prior collisions (published / failed / speculative).
- [web/index.html](web/index.html) — one-page UI with domain autocomplete and a "novel only" toggle.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# secrets via env — never hardcoded
copy .env.example .env
# edit .env, paste your key
$env:ANTHROPIC_API_KEY = (Get-Content .env | Select-String 'ANTHROPIC_API_KEY=(.+)').Matches.Groups[1].Value

uvicorn server:app --reload
```

Open <http://127.0.0.1:8000>.

Smoke-test the engine without an API call:

```powershell
python forge.py
```

prints the top candidate cross-domain pairs ranked by structural overlap.

## API

```bash
curl http://127.0.0.1:8000/theories
curl http://127.0.0.1:8000/pairs?limit=10

curl -X POST http://127.0.0.1:8000/collide \
  -H "Content-Type: application/json" \
  -d '{"a_id":"natural-selection","b_id":"mycelial-network"}'

curl -X POST http://127.0.0.1:8000/collide_domains \
  -H "Content-Type: application/json" \
  -d '{"domain_a":"mycology","domain_b":"economics"}'
```

Response shape:

```json
{
  "a": { "...seed theory..." },
  "b": { "...seed theory..." },
  "framework": {
    "name": "...",
    "core_claim": "...",
    "mechanism_borrowed_from": "...",
    "domain_applied_to": "...",
    "mapped_components": { "unit": "...", "variation": "...", "selection": "...", "fitness": "...", "boundary": "..." },
    "falsifiable_predictions": ["...", "..."],
    "viability": "promising | speculative | incoherent",
    "confidence": 0.62
  }
}
```

## Roadmap

Already in: ~110-theory seed, embedding similarity (`alpha` blend with tag Jaccard), tried/failed ledger with novelty filter and prior-aware prompting.

Next, when this stops being enough:

- **Graph store** — Neo4j once the seed passes ~500 and lineage queries matter.
- **Sources** — pull from SEP, arXiv, Semantic Scholar instead of hand curation.
- **Falsifiability filter** — a second LLM pass rejects predictions that aren't testable.
- **Verdict loop** — let the user mark generated frameworks as `promising`/`failed`, write back to `tried.json`, and the ledger compounds.

## Why this shape

Karpathy-style: one short core module you can read in a sitting, a thin server, a flat JSON seed, no premature databases. The interesting part is the decomposition schema — get that right and the rest is plumbing.
