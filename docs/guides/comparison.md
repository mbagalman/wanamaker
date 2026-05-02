# Comparison to Other MMM Tools

This page is a candid comparison of Wanamaker against the four tools most commonly
mentioned in the same breath: **Robyn**, **Meridian**, **Recast**, and **PyMC-Marketing**.

The goal is to give you enough information to choose the right tool for your situation —
including cases where the right choice is not Wanamaker. Senior data scientists will
fact-check these claims, and they should.

---

## How to Read This Page

Each section covers:

- What the other tool does well (honest)
- Who it is best for
- Where Wanamaker is likely a better fit
- Where the other tool is clearly the better choice

We do not assign scores or declare overall winners. The right tool depends almost entirely
on your team's technical depth, budget, scale, and what you actually need to do with the
results.

---

## Robyn (Meta)

**What it is:** Robyn is Meta's open-source MMM library, the most widely deployed
open-source MMM tool as of 2026. It is available as an R package with a Python companion
(`Robyn-PY`). Robyn uses gradient-free optimization (Nevergrad) rather than full Bayesian
MCMC sampling, which makes it substantially faster than Bayesian alternatives at the cost
of not producing proper posterior distributions.

### What Robyn does well

- **Accessible to R users.** Robyn has a deep community in R, thorough documentation, and
  a large library of worked examples. If your team lives in R, Robyn's ecosystem is hard
  to match.
- **Budget optimizer as a first-class feature.** Robyn's Pareto-front budget optimizer is
  mature, well-documented, and directly surfaced in the workflow. It is arguably the best
  open-source budget allocation interface available.
- **Meta backing and active maintenance.** Meta funds the core team. The library is
  actively maintained, and community support is substantial.
- **Speed.** Nevergrad optimization is much faster than MCMC sampling, making iterative
  model development more practical on modest hardware.

### Who Robyn is best for

Teams with existing R workflows, teams that prioritize fast iteration over posterior
uncertainty quantification, and teams where budget allocation recommendations (rather than
uncertainty-aware forecasting) are the primary output.

### Where Wanamaker is likely a better fit

- **Uncertainty quantification.** Robyn's optimization-based approach does not produce a
  posterior distribution. You get point estimates and calibration-derived confidence ranges,
  not credible intervals backed by MCMC. If your decision-makers ask "how confident are
  we in this ROI?" and the answer matters, a full Bayesian approach (Wanamaker, Meridian,
  or PyMC-Marketing) is more honest.
- **Refresh stability.** "Refresh nightmare" is a well-known Robyn community complaint —
  re-running the model on new data can produce substantially different historical estimates
  with no explanation. Wanamaker's refresh diff and movement classification (FR-4.3) are
  designed specifically for this problem.
- **Python-native teams.** While Robyn-PY exists, the R package is the primary surface.
  Wanamaker is Python-only and integrates naturally into Python data stacks.
- **Plain-English outputs for non-statistical stakeholders.** Wanamaker's executive
  summary and Trust Card are designed for CMOs who will not read a Robyn output file.

### Where Robyn is the better choice

- Your team already uses R and Robyn for other work.
- Budget allocation (not forecasting or trust reporting) is your primary output.
- You need fast iteration on model specifications and cannot wait for MCMC chains.
- Your data scale or channel count requires the speed of optimization-based fitting.

---

## Meridian (Google)

**What it is:** Meridian is Google's second-generation open-source Bayesian MMM library,
released in 2024 as the successor to LightweightMMM (which was deprecated January 2025 and
archived January 2026). Meridian uses JAX and HMC-based MCMC sampling via NumPyro.
It produces full posterior distributions and supports geo-level hierarchical modeling
as a first-class feature.

### What Meridian does well

- **Full Bayesian inference with GPU acceleration.** Meridian uses JAX and can leverage
  GPU hardware for significant sampling speedups on large models. For teams with 50+
  channels or geo-level data, this matters.
- **Geo-level hierarchical modeling.** If you have national-level spend data but want
  state- or DMA-level inference, Meridian's hierarchical structure handles this natively.
  Wanamaker v1 is national-only.
- **Google-specific features.** Google Query Volume and YouTube reach/frequency
  covariates are integrated first-class features. If you run significant Google media and
  have access to these signals, Meridian is designed around them.
- **Statistical rigor and research backing.** Meridian ships alongside Google research
  publications. The underlying methodology is well-documented and peer-reviewed.
- **Posterior distributions on everything.** Like Wanamaker, Meridian produces full MCMC
  posteriors with credible intervals, convergence diagnostics, and proper uncertainty
  quantification.

### Who Meridian is best for

Large organizations with 50+ channels, geo-level data requirements, GPU infrastructure,
data science teams comfortable with JAX and NumPyro internals, and companies with
substantial Google media spend who want to use Google's native signals.

### Where Wanamaker is likely a better fit

- **Operational simplicity.** Meridian requires JAX, NumPyro, and GPU infrastructure for
  practical use at scale. Wanamaker runs on CPU with a standard pip install. For a
  single analyst at a mid-market brand, Meridian's operational overhead is often
  disproportionate.
- **Diagnostics before modeling.** Wanamaker's `diagnose` command runs a structured
  pre-flight check and gives a readiness verdict before any model is fit. Meridian has
  no equivalent.
- **Refresh accountability.** Meridian does not have a built-in mechanism for comparing
  two model runs, classifying parameter movement, or generating a refresh diff report.
  This is a gap for organizations that run MMM on a recurring cadence.
- **Non-Google data stacks.** Meridian's first-class features assume Google data. If your
  media is primarily Meta, TikTok, CTV, or linear TV, Meridian's design philosophy is
  misaligned with your data.
- **Decision-oriented outputs.** Meridian produces posterior summaries and
  contributions. Wanamaker adds forecasting, scenario comparison, Trust Card, Experiment
  Advisor, and an executive summary template on top of the same Bayesian foundation.

### Where Meridian is the better choice

- You have geo-level data and need hierarchical inference across markets.
- You have 50+ channels or data scale that requires GPU-accelerated sampling.
- You have substantial Google media spend and want Google Query Volume or YouTube
  reach/frequency as covariates.
- You have a data science team with JAX and NumPyro experience who want maximum
  flexibility in model construction.
- You need to publish methodology with Google-backed research citations.

---

## Recast

**What it is:** Recast is a commercial SaaS MMM platform. It is not open-source.
As of 2026, pricing is typically in the $80,000–$120,000/year range. Recast handles
data ingestion, model fitting, reporting, and ongoing support as a fully managed service.

### What Recast does well

- **Managed operations.** Installation, infrastructure, model maintenance, and
  interpretation support are handled by Recast's team. There is no software to install or
  maintain.
- **Fast time-to-insight.** A typical Recast engagement goes from data submission to
  first results in days, not weeks. The operational friction that makes open-source MMM
  hard is eliminated by design.
- **Client-facing reporting.** Recast's outputs are designed for marketing executives.
  The reporting interface is polished and built for stakeholder communication.
- **Ongoing support.** You are buying a service, not software. Questions get answered by
  people, not documentation.
- **Proprietary methodology advantages.** Recast has invested significantly in
  out-of-sample validation techniques and calibration methodology. Their published
  approaches to holdout testing and lift test integration are well-regarded.

### Who Recast is best for

Mid-market and enterprise brands that can justify $80K+/year, need fast onboarding, want
a vendor accountable for the results, and do not have the internal capacity to operate
open-source tooling.

### Where Wanamaker is likely a better fit

- **Budget.** $80K–$120K/year is a significant threshold. For companies spending less than
  $20–30M/year on media, the cost of Recast may exceed the value recoverable from better
  measurement.
- **Auditability.** Recast's methodology is proprietary. You cannot inspect the model,
  audit the priors, or verify the math. For teams where auditability is a requirement
  (regulated industries, agencies needing methodology defensibility), open-source is the
  only option.
- **Ownership and reproducibility.** Wanamaker artifacts are local files you own. If
  Recast's pricing changes, the vendor relationship changes, or you want to reproduce a
  result two years later, you have everything you need. With Recast, you have reports.

### Where Recast is the better choice

- You can justify the cost and want a vendor-managed service with an SLA.
- You do not have internal capacity to install, configure, and operate open-source
  tooling reliably.
- Time-to-first-result is more important than methodology ownership.
- You want human support when results are surprising or the model needs reinterpretation.

---

## PyMC-Marketing

**What it is:** PyMC-Marketing is an open-source Python library from PyMC Labs built on
top of the PyMC probabilistic programming library. It provides MMM components (adstock
transforms, Hill saturation, Bayesian model construction) as well as customer lifetime
value and attribution modeling. Wanamaker uses PyMC as its underlying inference engine,
so these two tools share a common probabilistic foundation.

### What PyMC-Marketing does well

- **Flexible model construction.** PyMC-Marketing exposes a lower-level API that gives
  you fine-grained control over every component of the model. Custom adstock variants,
  novel saturation functions, and non-standard priors are straightforward to implement.
- **Broader modeling scope.** PyMC-Marketing covers CLV, attribution, and MMM in one
  library. If your team wants a single PyMC-based toolkit for multiple modeling tasks,
  PyMC-Marketing's breadth is an advantage.
- **Direct PyMC integration.** Because PyMC-Marketing is built directly on PyMC, every
  modeling choice is transparent and the full PyMC ecosystem (ArviZ diagnostics,
  posterior predictive checks, custom samplers) is immediately available.
- **Active development and community.** PyMC Labs maintains the library actively and
  the PyMC community is large and responsive.

### Who PyMC-Marketing is best for

Senior data scientists with Bayesian experience who want maximum modeling flexibility,
teams that already use PyMC for other work, and organizations that need to build custom
model variants (e.g., hierarchical channel structures, novel transforms) that a
purpose-built tool like Wanamaker does not support.

### Where Wanamaker is likely a better fit

- **Operational workflow.** PyMC-Marketing is a modeling toolkit. It does not include
  a data readiness diagnostic, a Trust Card, a refresh diff workflow, or executive summary
  templates. To go from raw CSV to a stakeholder-ready report, you build that workflow
  yourself. Wanamaker provides that workflow out of the box.
- **Non-statisticians as users.** The target Wanamaker user is a marketing analyst who
  can run a CLI command. PyMC-Marketing's user is a data scientist comfortable writing
  probabilistic model definitions in Python. These are different populations.
- **Refresh accountability.** PyMC-Marketing has no built-in mechanism for comparing
  successive model runs, classifying estimate movements, or generating refresh reports.
  If you run MMM on a monthly or quarterly cadence, this gap compounds quickly.
- **CLI-first workflow.** `wanamaker diagnose data.csv` → `wanamaker fit` →
  `wanamaker report` is a complete workflow runnable from a terminal by someone who has
  never written a PyMC model. PyMC-Marketing requires Python code to drive every step.

### Where PyMC-Marketing is the better choice

- You need modeling flexibility beyond what Wanamaker's standard feature set offers:
  custom adstock variants, hierarchical channel structures, non-standard priors, or
  experimental model architectures.
- You have a team of Bayesian practitioners who view Wanamaker's opinionated defaults as
  constraints rather than features.
- You are doing research or methodology development and need direct access to PyMC
  internals without an abstraction layer in the way.
- You are already using PyMC-Marketing for CLV or attribution and want a single library
  for all probabilistic marketing models.

---

## Summary Table

| | Robyn | Meridian | Recast | PyMC-Marketing | **Wanamaker** |
|---|---|---|---|---|---|
| **License** | MIT | Apache 2.0 | Commercial SaaS | Apache 2.0 | MIT |
| **Language** | R (Python port) | Python / JAX | Hosted service | Python | Python |
| **Inference method** | Gradient-free optimization | HMC / MCMC | Proprietary | HMC / MCMC | HMC / MCMC |
| **Posterior distributions** | No (point estimates) | Yes | Unknown | Yes | Yes |
| **Geo-level modeling** | Limited | Yes (first-class) | Yes | Possible | No (v1) |
| **GPU required for scale** | No | Yes (recommended) | N/A (managed) | No | No |
| **Data readiness diagnostic** | No | No | N/A | No | Yes |
| **Refresh diff + accountability** | No | No | Partial | No | Yes |
| **Trust Card / credibility audit** | No | No | Partial | No | Yes |
| **Executive summary output** | Partial | No | Yes | No | Yes |
| **Self-contained HTML stakeholder report** | No | No | Yes (web UI) | No | Yes |
| **Risk-adjusted allocation ramp** | No | No | Partial | No | Yes |
| **Lift-test calibration** | Yes | Yes | Yes | Yes | Yes |
| **Budget optimizer** | Yes (mature) | Partial | Yes | No | No (v1) |
| **Installation complexity** | Low (R) / Medium (Python) | High (JAX/GPU) | None (SaaS) | Low | Low |
| **Cost** | Free | Free | $80K–$120K/yr | Free | Free |

---

## A Note on Honest Positioning

Wanamaker is not the right tool for every situation. The table above reflects genuine
tradeoffs, not marketing.

If you have geo-level data and a data science team, **Meridian** is more capable.
If you have an R-native team that needs fast iteration, **Robyn** is more practical.
If you can afford it and want a managed service, **Recast** eliminates operational
friction. If you need maximum modeling flexibility, **PyMC-Marketing** gives you
more control.

Wanamaker's case is narrower: a Python-native team, national-level data, recurring
model runs where refresh accountability matters, and stakeholder reporting to non-technical
audiences. Within that space, it is designed to be the most complete and trustworthy
option available at zero cost.
