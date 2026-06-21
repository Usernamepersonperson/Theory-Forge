# Theory Forge

A combinatorial theory engine. Borrow a mechanism from one field, apply it in the domain of another, see what falls out.

The white space (per the PRD): no one has built a tool that maps the *structure* of theories (not their content) and then collides them across domains. This is the smallest thing that does that.

## Current scale

- **853 seed theories** across **205 domains** — from mycology to metallurgy, glaciology to glassblowing, kintsugi to cryptography
- **2400+ generated frameworks** across 160 collision batches
- **81 deep-dive analyses** with mapped components, falsifiable predictions, and experimental designs
- **2341 unique ranked frameworks** — mean confidence 0.660, top confidence 0.83

## How it works

1. **Decompose** — every theory is broken into `unit / variation / selection / fitness / boundary / tags`. See [theories.json](theories.json) for the seed dataset.
2. **Fingerprint** — tag-set Jaccard *plus* optional sentence-transformer embeddings of the mechanism slots. Blend controlled by `alpha`.
3. **Filter for novelty** — [tried.json](tried.json) is a ledger of collisions already published (e.g. Darwin × markets → evolutionary economics) or known to fail. The engine excludes these so what you see is white space.
4. **Collide** — Claude maps the mechanism into the new domain and emits a falsifiable prediction. If a prior collision exists on the same pair, it's surfaced as context so the model proposes something distinct.

That's the whole loop. ~300 lines.

## Files

- [forge.py](forge.py) — core: loading, fingerprinting, tag + embedding similarity, ledger-aware novelty filter, LLM synthesis. Single file, runnable.
- [server.py](server.py) — FastAPI v0.7: 30 endpoints including Research Assistant, visualizations, batch forge, rankings, deep dives, stats, search, export, and domain network graph.
- [theories.json](theories.json) — decomposed seed dataset (853 entries across 205 domains).
- [tried.json](tried.json) — known-prior collisions (published / failed / speculative).
- [web/index.html](web/index.html) — 12-tab UI: forge, research assistant, batch forge, rankings, deep dives, theories, stats, network, heatmap, timeline, genealogy, history.
- [outputs/](outputs/) — generated collision batches, rankings, and deep-dive analyses.

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

```
GET  /theories                                seed theories (optional ?domain= filter)
GET  /theories/{id}                           single theory
GET  /domains                                 unique domain list
GET  /ledger                                  known collisions
GET  /pairs?limit=&novel_only=&alpha=         ranked cross-domain pairs
POST /collide          { a_id, b_id }         synthesize new framework
POST /collide_domains  { domain_a, domain_b }
POST /batch_collide    { count, domain_filter }  batch N collisions
GET  /history                                 all saved batch outputs
GET  /rankings?limit=&min_confidence=&domain= ranked frameworks
GET  /deep-dives                              list deep-dive analyses
GET  /deep-dives/{slug}                       single deep-dive detail
GET  /stats                                   aggregate statistics
GET  /search?q=&limit=                        full-text framework search
GET  /domain-matrix                           domain coverage analysis
GET  /export?format=json|csv&min_confidence=  export all frameworks
GET  /random?min_confidence=                  random framework above threshold
GET  /domain-network                          graph data (nodes + weighted edges)
GET  /semantic-search?q=&limit=               embedding-based semantic search
GET  /recommend?limit=&strategy=              collision recommendations (coverage/diversity/bridge)
GET  /chain/{name}?depth=                     framework exploration chain
GET  /synthesis?domain=&top_n=&format=        synthesis report (markdown/html)
GET  /tag-analysis                            tag frequency and domain spread
GET  /domain-heatmap                          Jaccard similarity matrix for all domains
GET  /genealogy                               seed theory → framework lineage tree
GET  /timeline                                when each domain was first explored
GET  /compare?names=a,b,c                     side-by-side framework comparison
GET  /surprise-chain?depth=                   random high-confidence 10-hop chain
```

## UI tabs

| Tab | What it does |
|-----|-------------|
| **forge** | Pick two domains or click "surprise me" to collide random theories. Search box for existing frameworks. |
| **research** | Research Assistant: semantic search, collision recommendations (3 strategies), chain explorer, surprise chain, compare frameworks, synthesis reports (markdown + rich HTML). |
| **batch forge** | Generate 3-10 collisions at once with optional domain filter. |
| **rankings** | Browse all frameworks ranked by confidence with viability badges. Filter by min confidence and domain. |
| **deep dives** | Read detailed analyses of top frameworks — mapped components, predictions, experiments, limitations. |
| **theories** | Browse all 853 seed theories grouped by domain, with full decomposition and tags. |
| **stats** | Dashboard with counts, viability breakdown, confidence stats, domain coverage, mechanism source charts, and export buttons (JSON/CSV). |
| **network** | Interactive force-directed graph showing domain connections. Drag nodes, hover for details. |
| **heatmap** | Interactive domain similarity heatmap — Jaccard overlap between all 205 domains on a color-coded canvas matrix. |
| **timeline** | Discovery timeline — cumulative chart showing when each domain was first explored, batch-by-batch domain list. |
| **genealogy** | Theory genealogy tree — which seed theories produced the most frameworks, expandable offspring list. |
| **history** | Expandable list of all collision batches with framework details. |

## Why this shape

Karpathy-style: one short core module you can read in a sitting, a thin server, a flat JSON seed, no premature databases. The interesting part is the decomposition schema — get that right and the rest is plumbing.
