# AGENTS.md

Guidance for AI coding assistants (Claude Code, Cursor, Aider, Copilot Workspace, etc.) working on the Wanamaker open-source MMM project.

This file is short by design. The strategic and product context lives in `docs/wanamaker_brd_prd.md` (the BRD/PRD). This file covers *how to behave* while building, not *what to build*.

---

## Project context

Wanamaker is an open-source Bayesian Marketing Mix Model targeting mid-market companies that lack a deep data science bench. The core differentiators — the things that distinguish this project from Robyn, Meridian, LightweightMMM, and Recast — are:

- **Refresh accountability:** light posterior anchoring + comprehensive diff reports, so historical estimates don't silently rewrite themselves.
- **Trustable usability:** data readiness diagnostic, model trust card, plain-English executive summary, conservative scenario comparison.
- **Local-first processing:** no network calls in any core operation, no telemetry.
- **Decision-oriented outputs:** forecasting and scenario comparison, not just descriptive analysis.

The full rationale is in the BRD/PRD. **Read it before making non-trivial decisions.** Path: `docs/wanamaker_brd_prd.md`.

---

## Hard rules (do not violate without explicit approval)

These are architectural invariants. Violating them means rebuilding the project's credibility from scratch.

1. **No network calls in core operations.** The commands `diagnose`, `fit`, `report`, `forecast`, `compare-scenarios`, and `refresh` must not import or invoke any HTTP client, telemetry library, crash reporter, or remote service. There is a CI test for this. If you find yourself adding `requests`, `httpx`, `urllib`, telemetry SDKs, or any "phone home" library to these code paths, **stop and ask**. Documentation builds and example downloads are separate code paths and may use the network.

2. **No LLM calls for output generation.** The plain-English executive summary, the trust card, the experiment advisor recommendations, and any other user-facing report text must be generated from deterministic Jinja2 templates driven by posterior statistics. Do not add OpenAI, Anthropic, or any other LLM API call to generate output text. If a template feels limiting, expand the template logic — don't reach for an LLM. (LLMs may be used in dev tooling, never in product output.)

3. **No tree-based modeling for ROI / saturation / adstock parameters.** xgboost is in the project for two narrow purposes only: a quick-mode forecast preview (clearly labeled as such) and a validation cross-check against the Bayesian model. Tree models do not produce ROI estimates, saturation curves, adstock parameters, or any other channel-level inference output. If a feature request seems to want xgboost for these, you've misread the request — re-read the BRD/PRD.

4. **Statistical code is load-bearing.** Adstock formulas, Hill saturation, mixture priors, posterior anchoring, credible interval computations — these are the product. Bugs here produce silently wrong ROI numbers, which destroys the entire trust thesis. Statistical functions must:
   - Have a docstring referencing the canonical source (paper, textbook, or PRD section)
   - Have unit tests against worked examples with known outputs
   - Be reviewed by a human before merge, even if all tests pass
   - Have any default parameter values cross-checked against the PRD

   **Before implementing or modifying any adstock or saturation transformation, read `docs/references/adstock_and_saturation.md`.** It contains the canonical formulas, default priors per channel category, estimation method tradeoffs, and diagnostic patterns that should govern all transformation code.

5. **Reproducibility is a contract.** Per NFR-2, given the same input data, configuration, and random seed, results must be bit-for-bit identical. This means:
   - Use a single, documented seeding discipline. The seed is configured at the top level and passed down explicitly. Do not call `np.random.seed()` or `random.seed()` inside library functions.
   - Sampler operations must use the explicit seed, not the global state.
   - There is a CI test that runs the same fit twice and compares posteriors bit-for-bit.

6. **The CSV interface is the user contract.** Don't redesign input/output formats without revisiting the PRD. If a field seems redundant or missing, it's likely intentional — check `docs/wanamaker_brd_prd.md` Section 5.1 before refactoring.

---

## Engine choice — currently undecided

The Bayesian engine choice (PyMC vs. NumPyro vs. Stan) is the output of Phase -1 and is **not yet decided**. Until Phase -1 concludes:

- Write modeling code against a thin abstraction layer (`wanamaker.engine.*`) that hides the specific library
- Don't import PyMC or NumPyro directly from feature code
- Don't add either to the production dependencies in `pyproject.toml` until the decision is made
- Tests can use whatever engine is convenient for prototyping; mark them clearly

The current leading candidate is PyMC. The current rejected option for the modeling role is xgboost (see BRD/PRD Section 9 for the reasoning).

---

## Coding conventions

- **Python 3.11+.** Use modern type hints (`list[str]`, not `List[str]`; `X | None`, not `Optional[X]`).
- **Type hints required on all public functions.** Internal helpers may omit them but it's preferred to include them.
- **Docstrings:** Google style. Required on all public functions, classes, and modules. Include "Raises" and "Examples" sections where relevant.
- **Formatting:** `ruff format` (or `black` if ruff isn't set up yet). 100-character line limit.
- **Linting:** `ruff check`. All checks must pass before commit.
- **Testing:** `pytest`. Aim for 80%+ coverage on statistical and core logic; UI/CLI helper coverage matters less. Statistical functions must have explicit test cases against known outputs.
- **Imports:** absolute imports within the package. No wildcard imports.
- **Mutability:** prefer immutable data structures. Use `dataclasses` (frozen where reasonable) or pydantic models for structured data. Avoid mutating function arguments.
- **Errors:** raise specific exception types. Don't catch broad `Exception` unless re-raising. User-facing errors must be actionable ("the date column 'week' has 3 missing values at rows 12, 47, 89" — not "ValueError: bad data").

---

## Project structure (target)

```
wanamaker/
├── src/wanamaker/
│   ├── __init__.py
│   ├── cli.py              # typer/click CLI entry points
│   ├── config.py           # YAML config loading + pydantic validation
│   ├── data/               # CSV I/O, schema validation
│   ├── diagnose/           # data readiness diagnostic
│   ├── engine/             # Bayesian engine abstraction (PyMC/NumPyro/etc.)
│   ├── transforms/         # adstock, saturation
│   ├── model/              # model specification, priors, fit logic
│   ├── refresh/            # versioned runs, diff reports, anchoring
│   ├── forecast/           # posterior predictive, scenario comparison
│   ├── trust_card/         # credibility dimensions, classification
│   ├── advisor/            # experiment advisor (minimal v1)
│   ├── reports/            # Jinja2 templates, HTML/Markdown rendering
│   ├── benchmarks/         # benchmark dataset loaders
│   └── _xgboost_aux/       # xgboost in supporting role only
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/         # benchmark-driven acceptance tests
├── docs/
│   ├── wanamaker_brd_prd.md
│   ├── analyst_guide.md
│   ├── cmo_guide.md        # "What Wanamaker Tells Your CMO"
│   ├── comparison.md       # vs. Robyn/Meridian/Recast/PyMC-Marketing
│   └── references/
│       └── adstock_and_saturation.md  # canonical reference for transforms
├── examples/
├── benchmark_data/         # synthetic + public datasets
├── pyproject.toml
├── README.md
├── AGENTS.md               # this file
└── LICENSE                 # MIT
```

If the project hasn't reached this structure yet, propose changes incrementally — don't restructure everything at once.

---

## Common commands

These will be filled in as the project matures. Current state:

```bash
# Install for development
pip install -e ".[dev]"

# Run tests
pytest

# Run linters and formatters
ruff check .
ruff format .

# Run benchmarks
pytest tests/benchmarks/

# Build docs locally (once docs are set up)
mkdocs serve

# Run the CLI on the public benchmark dataset
wanamaker diagnose benchmark_data/public_example.csv
wanamaker fit --config benchmark_data/public_example.yaml
wanamaker report --run-id <run_id>
```

---

## When in doubt

1. **Check the BRD/PRD.** Most architectural questions are answered there.
2. **Match existing patterns.** If you're adding a new feature, look at how a similar existing feature was structured.
3. **Ask, don't assume.** If a request is ambiguous in a way that affects architecture, the API surface, or any of the hard rules above, ask before implementing.
4. **Prefer simplicity.** This project's whole thesis is "trustable usability over methodological novelty." When choosing between a clever solution and a boring one that works, choose boring.
5. **Default to scope discipline.** v1 is deliberately narrow. If a request feels like it's expanding scope into v1.1 or v2 territory, flag it rather than just implementing it.

---

## Things that are *not* hard rules but are strong preferences

- **Avoid abstractions until you have three concrete cases.** Premature generalization is more dangerous than mild duplication in a research-flavored codebase like this one.
- **Avoid the temptation to make things "production-grade enterprise-y."** No dependency injection frameworks, no event buses, no plugin systems unless explicitly justified by a v1 requirement.
- **Don't add config options without a user.** Every YAML field is documentation overhead. Add config options when a real user asks, not speculatively.
- **Avoid breaking changes after v1.0.** Use deprecation warnings and a clear migration path. The mid-market analyst persona will not tolerate frequent breaking changes.

---

## Statistical correctness checklist

When implementing or modifying any statistical function, verify:

- [ ] The mathematical form matches the PRD or a cited reference
- [ ] Default parameter values match the PRD
- [ ] Edge cases are handled: zero spend, constant spend, missing data, single-row data, all-zero target
- [ ] Numerical stability is considered (e.g., log-space computations where needed)
- [ ] Unit tests include at least one worked example with known input → known output
- [ ] If the function is called during fitting, it interacts correctly with the engine abstraction
- [ ] If the function affects user-facing output, it's covered by an integration test that runs through to the report

---

## Privacy and security

Per BRD/PRD Section 6, this project has explicit privacy constraints. When working on any feature:

- **No network calls** in core commands (see Hard Rules above).
- **No PII handling.** This tool processes aggregate marketing data, not user-level data. If you find yourself implementing PII masking, anonymization, or user-level joins, you've misunderstood the scope — re-read the BRD/PRD.
- **No telemetry, ever** without explicit user opt-in. If a future telemetry system is built, it must be opt-in (not opt-out), disabled by default, transparently documented, and easy to disable per-run and per-installation.
- **Local-only artifact storage.** Model artifacts, run logs, and reports go in a project-local `.wanamaker/` directory. Don't write to the user's home directory or system locations without explicit configuration.

---

## A note on the project's voice

Wanamaker is a Blue Ocean play — its differentiation comes from honesty about uncertainty, not from sounding sophisticated. When writing user-facing strings, error messages, or documentation:

- Be dry and direct, not breathless.
- Don't hide complexity, but don't manufacture it either.
- When the model can't tell the user something, say so plainly. "TV spend was constant during the model period — we can't estimate its saturation from data; the curve shown is from prior knowledge only." Not "Insufficient variance for parameter identifiability."
- Avoid AI-sparkle phrases like "powered by," "intelligent," "smart," "advanced." This project's credibility comes from being unflashy.
- The project name comes from the apocryphal John Wanamaker quote about half of advertising spend being wasted. The misattribution is part of the story — a quote about measurement uncertainty whose own provenance is uncertain.
