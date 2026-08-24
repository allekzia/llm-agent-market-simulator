# LLM Agent Market Simulator

**Research question:** when autonomous LLM agents set prices in a competitive
market, do they behave like textbook competitors (prices driven toward
marginal cost), or does something closer to tacit collusion emerge, even
without being told to cooperate?

This project builds a testbed to study that question. Multiple agents run
competing firms in a simulated market over many rounds. The market's
economics (demand, pricing, profit) are deterministic and hand-coded; only
the agents' *strategy* is LLM-driven. Varying what agents can see and
whether they can communicate lets us measure how information conditions
affect pricing behavior.

> **Status: Stage 6 complete, a real finding.** Under a same-model,
> contamination-filtered comparison (14 isolated vs 11 connected
> runs), connected agents priced lower (17.2% vs 30.4% average markup,
> p = 0.0004) and converged far more tightly to each other (p = 0.0003)
> than isolated agents, the opposite of the tacit-collusion hypothesis
> this project set out to test. See [Roadmap](#roadmap) below.

## Why this question matters

Algorithmic pricing is already widespread in e-commerce, ride-hailing, and
ad auctions. Research in economics has shown that reinforcement-learning
pricing agents can learn to sustain higher-than-competitive prices without
any explicit coordination, a form of "algorithmic collusion." As LLMs
increasingly get deployed as autonomous pricing/business agents, whether
they exhibit similar tendencies is an open, practically important question.
This project doesn't claim to settle it (it's a small, honest testbed, not
a research paper), but it's built to produce a real, falsifiable finding
rather than just a demo.

## Architecture

```
market_sim/
├── Dockerfile
├── .dockerignore
├── .github/
│   └── workflows/
│       ├── tests.yml             # CI: runs the full test suite on push/PR
│       └── docker-build.yml      # CI: builds and smoke-tests the Docker image
├── .streamlit/
│   └── config.toml              # dark theme config for the dashboard
├── core/
│   ├── demand_model.py         # deterministic multinomial logit demand + profit
│   ├── environment.py          # round loop connecting agents to the market
│   └── agents/
│       ├── base.py             # Agent interface (Observation to Decision)
│       ├── rule_agent.py       # baseline strategies: cost-plus, undercut, noisy-match
│       └── llm_agent.py        # LLM-backed agent, messaging-capable
├── experiments/
│   ├── stage1_baseline_runner.py
│   ├── stage2_llm_vs_baseline_runner.py
│   ├── stage3_conditions_runner.py
│   ├── stage3_repeats_runner.py
│   ├── stage6_connected_completion_runner.py
│   └── stage6_isolated_completion_runner.py
├── analysis/
│   ├── plot_baseline.py
│   ├── plot_stage3.py
│   ├── stage3_stats.py         # permutation test on repeated-run results
│   ├── collusion_metrics.py    # price correlation and dispersion
│   └── collusion_stats.py      # permutation test on collusion-proxy metrics
├── dashboard/
│   ├── app.py                  # Streamlit dashboard: live sim + recorded results
│   └── theme.py                # design tokens and CSS, kept separate from app logic
├── tests/
│   └── (one test file per module above)
└── requirements.txt
```

**Design decision: deterministic economics, LLM-driven strategy only.**
The demand model (how price and marketing spend translate into market
share and profit) is a standard multinomial logit model, coded directly,
not learned. Only the agents' pricing/marketing *decisions* will be
LLM-driven from here on. This keeps results reproducible and testable,
keeps API costs bounded, and means any collusion-like signal found later
comes from agent behavior, not from quirks in the simulated market itself.

## Stage 1: validating the deterministic core

Before introducing any LLM, the underlying economic model needs to behave
the way theory predicts, otherwise there's no way to tell later whether
LLM agents are doing something interesting or the simulation is just
broken.

**10 unit tests** (`tests/test_demand_model.py`) check the demand model's
basic economic sanity: market shares stay in [0,1], cheaper firms win more
share, higher marketing spend increases share, pricing at marginal cost
yields zero profit, pricing below cost yields negative profit, etc. All 10
pass.

**Three rule-based scenarios** (`experiments/stage1_baseline_runner.py`)
validate system-level behavior, run for 30 rounds each with identical
marginal costs ($2.00):

| Scenario | Strategy | Result |
|---|---|---|
| Pure undercutters | Each firm undercuts the cheapest competitor by 5% | Price converges to **$2.02 (1.0% markup)**, Bertrand competition working as expected |
| Pure cost-plus | Each firm holds a fixed 30% markup | Price stable at **$2.60 (30% markup)**, no convergence pressure, as expected |
| Mixed strategies | Undercutter + cost-plus + noisy-match | Prices settle **between the two extremes** ($2.36 to $2.60), with the aggressive undercutter pulling the group down |

![Stage 1 price convergence](experiments/results/baseline_convergence.png)

This confirms the simulation reproduces standard oligopoly pricing
dynamics before any LLM strategy is introduced, the necessary baseline
for everything that follows.

## Stage 3: multiple LLM agents, and comparing information conditions

> **Note:** this stage's data was collected under
> `llama-3.3-70b-versatile`, since deprecated by Groq. The isolated vs
> connected comparison here should not be trusted as a finding, the
> markup levels turned out to be highly model-dependent, see the Stage 6
> section below for the valid, same-model comparison and the actual
> finding. This section is kept as the honest record of how that was
> discovered.

The Agent interface was extended with an optional messaging channel:
agents can now see a short note from other agents' previous round, and
can optionally write one of their own for the next round. This required
no changes to the demand model or existing agents, only two new fields
on the existing Observation and Decision objects, a direct payoff of the
interface design from stage 1. 17 new tests cover the environment's
message passing and the LLM agent's conditional prompt content, all
without a live API key.

Two conditions were compared: **isolated** (three LLM agents, no
visibility into competitor prices, no messaging) versus **connected**
(full visibility, messaging enabled).

**An important technical caveat, discovered while interpreting the
first results:** the environment's random seed controls Python's own
randomness, but not the LLM's. Every API call is made at temperature
0.7, so even repeated runs of the identical setup produce different
outcomes. A single run of each condition is not enough to trust; this
was confirmed directly when a second run of the same setup produced a
noticeably different result from the first. An initial hypothesis from
the first two runs, that agents kept independently landing on a price
of 3.75, also did not hold up once more data came in, a useful early
lesson in not trusting a pattern seen only once or twice.

**Two infrastructure bugs surfaced while trying to collect a larger,
more trustworthy batch of runs, both fixed:**

The first API rate limit hit (HTTP 429) exposed a design flaw: the
API call function retried internally with a growing backoff, while the
agent wrapped that entire thing in a second retry loop of its own.
Under sustained rate limiting the two multiplied together into several
real minutes of silent waiting for a single decision, indistinguishable
from the script being frozen. Fixed by bounding the inner backoff
tightly and adding a live per-round progress line, so a slow round can
never again look identical to a stalled one.

The second issue was a cumulative daily quota, not a pacing problem: a
large batch (8 seeds x 2 conditions x 3 agents x 10 rounds, roughly 480
calls) ran cleanly for the entire isolated condition, then began
failing partway through the connected condition, most likely because
the day's free-tier quota was exhausted by that point in the session.
Every failure was caught and flagged live rather than silently
corrupting the results, which is what made the contamination visible
and preventable rather than something that would have quietly produced
misleading final numbers.

**Isolated condition: replicated across two independent 8-seed batches
(16 seeded runs total), both completed with zero failures.** Averaging
each run's three agents first (see `analysis/stage3_stats.py` for why
this is the correct unit of comparison, not each agent separately):

| Batch | Mean markup | Std dev | Range |
|---|---|---|---|
| First batch | 75.0% | 8.1% | 62.5% to 87.5% |
| Second batch | 73.4% | 8.1% | 62.5% to 100.0% |
| **Combined (n = 16)** | **74.2%** | **8.1%** | **62.5% to 87.5%** (per-seed average) |

Two independent batches landing within one percentage point of each
other, with matching spread, is real replication, not a coincidence of
one lucky sample. With no visibility into competitors and no ability to
communicate, LLM agents consistently converge to a markup around three
quarters above cost.

**A second pattern replicated perfectly: 16 out of 16 isolated runs
opened with the identical sequence**, round 0 at $3.00, round 1 at
$3.50, every agent, both batches, every seed. This is no longer a
one-off curiosity; it looks like the model's very first couple of
pricing decisions, before any real history exists to react to, are
close to deterministic in practice despite temperature 0.7 sampling,
only diverging once there is feedback to respond to. Worth investigating
directly in a later stage.

**Connected condition: two clean data points so far, both notable.**
Across two separate attempts at the full batch, both interrupted by API
rate limiting partway through (see above), exactly one seed (42)
completed cleanly each time, and both times it converged to the exact
same result: all three agents at $3.25, a 62.5% markup. That specific
repetition is a striking detail worth keeping, but it is two data
points, not sixteen, and both share the same nominal seed, so it cannot
yet be compared statistically against the isolated result. A clean,
larger connected batch remains the immediate next step.

## Stage 2: first LLM agent on the street

An LLM-backed agent (`core/agents/llm_agent.py`) was added, implementing
the same `Agent` interface as the rule-based agents from stage 1. It
builds a prompt from its `MarketObservation`, calls an LLM (Groq's free
tier), and parses the response into a price and marketing decision. The
prompt is deliberately neutral: it never uses words like compete,
undercut, cooperate, or collude, so any behavior that emerges reflects
the agent's own reasoning, not an instruction. 18 new tests cover prompt
content, response parsing (including malformed and markdown-wrapped
responses), retries, and safe fallback behavior, all without needing a
live API key.

**A real bug was found and fixed during the first live test.** The LLM
agent set an extreme marketing budget in its first round and captured
almost the entire market for a trivial cost, since marketing spend
increased market share but was not charged proportionally to the
advantage it bought. This is a small example of reward hacking: an
unmodeled free lever in the environment, found immediately by an agent
optimizing for profit. The fix caps marketing spend to a scale
comparable to price (`max_marketing` in `MarketParams`) and charges it as
a real cost every round, enforced centrally in `compute_round` so no
agent can bypass it. All stage 1 baseline results were re-verified
unchanged after the fix, since they never used marketing.

**Preliminary observation (single run, one seed, not yet a finding):**
with the bug fixed, across 15 rounds the LLM agent's price never dropped
below either rule-based competitor, settling in a $2.80 to $3.00 range
(40% to 50% markup) against an Undercutter at 23.5% and a CostPlus agent
at a fixed 30%. It also showed a repeating price cycle rather than
settling at a stable value, climbing in small steps, overshooting past
the point where profit actually peaked, then correcting back down,
consistent with its limited memory window (it only sees its last 5
rounds of history each round). This is one run at one temperature
setting with one random seed, and should not be read as evidence of any
real pattern yet. Confirming whether this holds requires the repeated,
seeded experiments planned for later stages.

## Stage 4: interactive dashboard

A Streamlit dashboard (`dashboard/app.py`) provides three views: a live,
instant rule-based simulation (no API key needed, since it's pure local
computation), a browser for already-recorded LLM experiment results
(reads the real CSV logs from stages 2 and 3), and a statistics view
comparing the isolated and connected conditions.

**Visual design** lives in `dashboard/theme.py`, kept separate from the
app logic: a deep indigo-to-violet gradient background with frosted
glass cards, matching a specific visual reference. Cyan and violet are
reserved specifically to mean the isolated and connected conditions
everywhere in the app, charts, header stats, badges, not used
decoratively for anything else. The header's live stats (average
markup, run counts) are computed from the real CSV data on load, not
hardcoded, so they can't silently go stale. A dark Streamlit theme
(`.streamlit/config.toml`) keeps native widgets consistent with the
custom styling. 5 tests (`tests/test_dashboard_theme.py`) cover the
color-assignment logic directly.

**Deliberate design decision: the dashboard never makes a live LLM API
call.** All LLM-related results shown are ones already collected and
saved by the stage 2/3 runner scripts. This matters once the dashboard
is deployed publicly in a later stage: a public page able to trigger
real, billed API calls on a personal key, or able to retrigger the same
rate limits worked through in stage 3, would be a real risk, not a
hypothetical one. Two tests (`tests/test_dashboard.py`) use Streamlit's
own headless testing tool to confirm the app loads and its live
simulation runs without error, without needing a real browser.

To run it locally:
```bash
streamlit run dashboard/app.py
```

## Stage 5: Docker, CI, and deployment

**Docker.** A `Dockerfile` packages the dashboard and its dependencies
into a self-contained image: anyone with Docker can run it identically,
regardless of what's installed on their own machine. No API key is
baked into the image; the dashboard never needs one, see the stage 4
design decision above.

**The Docker image is built and smoke-tested automatically by CI**
(`.github/workflows/docker-build.yml`), on GitHub's own free runners,
rather than requiring Docker installed locally. This matters
practically: Docker Desktop can require a paid license depending on the
machine or organization, so relying on it for local verification isn't
a safe assumption to build a portfolio project around. The workflow
builds the image, starts the container, and polls Streamlit's built-in
health endpoint (`/_stcore/health`) until it responds or times out,
confirming the image doesn't just build, it actually starts and serves
traffic correctly.

**CI.** `.github/workflows/tests.yml` runs the full test suite
automatically on every push to `main` and every pull request, so a
broken change is caught before it merges, not after. One real gotcha
worth documenting: YAML interprets an unquoted `on` key as the boolean
`true` rather than the string "on" (a YAML 1.1 quirk with
`on`/`off`/`yes`/`no`). GitHub's parser handles this specific case
correctly regardless, but the key is quoted (`"on":`) in both workflow
files to remove the ambiguity entirely, caught by validating each file
with a real YAML parser before trusting it.

**A real bug caught while preparing this stage:** `.gitignore` was
still excluding `experiments/results/*.csv` from stage 1, when those
files were just scratch output. Left as-is, the dashboard's recorded
results and stats tabs would have silently shown no data once
deployed, since the CSVs they depend on would never have been
committed. Fixed before it caused a confusing, hard-to-diagnose empty
deployment.

**Deployment target: Streamlit Community Cloud.** The original plan
considered a Docker-based hosting platform, specifically to exercise
the Docker skill in the live deployment path. That platform's free
tier changed to require a paid plan partway through this stage (a
real, recent product change, not a workaround-able restriction), so
Docker was kept as a demonstrated local/CI skill instead, and the
actual public deployment uses Streamlit Community Cloud, which remains
genuinely free for public apps and deploys directly from this GitHub
repo with no Docker required on their end at all.

The Docker work wasn't wasted: `Dockerfile` and its CI validation
(`.github/workflows/docker-build.yml`) stay in the project as a real,
demonstrated skill, buildable and runnable locally with Docker Desktop
or Podman, and verified automatically on every push, they're just not
the live public deployment path.

To deploy on Streamlit Community Cloud: go to
[share.streamlit.io](https://share.streamlit.io), connect the GitHub
account, choose this repo, set the main file path to
`dashboard/app.py`, and deploy. No config file needed beyond the
existing `requirements.txt`.

Cost/abuse guardrails: the deployed dashboard makes zero live API
calls (inherited from stage 4's design), so there's no key to leak and
no usage cost that a visitor could run up.

To run the Docker image locally, if you have Docker Desktop or a
compatible tool installed
([Podman](https://podman.io) is a free, Docker-compatible alternative
that uses the same commands):
```bash
docker build -t market-sim .
docker run -p 7860:7860 market-sim
```
Then open `http://localhost:7860`. This step is optional, not required
to trust the image works: CI already builds and smoke-tests it on
every push.

## Stage 6: closing the sample gap, and a real collusion-proxy metric

**The sample-size problem going into this stage:** isolated has 16
independent, replicated runs. Connected has only 2 clean data points.
`experiments/stage6_connected_completion_runner.py` runs 14 fresh,
previously-unused seeds for the connected condition specifically, to
bring it up to a comparable sample size. It deliberately avoids reusing
any seed value already present as a result file, reusing a seed would
silently overwrite that run's data rather than adding to it, exactly
the kind of quiet data-loss bug worth designing around up front rather
than discovering after the fact.

**A metric beyond average markup.** Average price alone doesn't
directly measure collusion, agents could independently land on similar
high prices without ever moving together. `analysis/collusion_metrics.py`
adds two metrics computed from each run's actual round-by-round price
data: **price correlation** (how closely each pair of agents' prices
move together over time, a more direct alignment signal than similar
average levels) and **price dispersion** (how spread out final prices
are across agents in a run, lower means more converged). Both are
covered by 9 tests using hand-constructed data with known correct
answers (perfectly correlated series score 1.0, perfectly opposed
series score -1.0, etc.), so the metrics themselves are trustworthy
before being pointed at real experiment data.

**`analysis/collusion_stats.py`** scans every real per-seed CSV on disk
and runs the same permutation test used for markup in
`stage3_stats.py` (reused directly, not duplicated) on both new
metrics, so the correlation and dispersion comparisons get the same
statistical rigor as the original markup comparison.

**Status: complete. Real, same-model, contamination-filtered results
below.** The final clean batch: 14 isolated runs (0% fallback, a
completely clean batch) and 11 connected runs (11 of 14 attempted
seeds completed clean; 3 were excluded automatically by the
contamination filter after hitting the same daily quota limit late in
the batch, a repeatable pattern across two separate attempts now,
worth knowing this API key reliably supports roughly 11 clean
multi-agent seeds per day before the ceiling).

**Two real problems surfaced on the first live attempt at this batch,
both fixed:** Groq deprecated the `llama-3.3-70b-versatile` model used
since stage 2 (confirmed via their own deprecations page), causing
every single call to fail with HTTP 404 rather than the 429 rate-limit
errors seen before, a different failure mode the existing warning
system still caught correctly and immediately. Fixed by switching to
`openai/gpt-oss-20b`, Groq's current recommended smaller replacement.

While fixing this, a second, quieter bug surfaced: `stage3_conditions_runner.py`
had a `MODEL` constant at the top with a comment inviting it to be
edited if needed, but it was never actually passed into the `LLMAgent`
instances below it, so editing it would have silently done nothing.
The real default was buried inside `llm_agent.py` itself. Fixed by
wiring the constant through properly, so it now does what its own
comment always claimed it did.

**A third issue appeared on the next attempt, after the model switch:**
`openai/gpt-oss-20b` is a reasoning model, it spends tokens on an
internal chain-of-thought before writing its actual answer. With the
original `max_tokens: 200` budget, generation was getting cut off
entirely during that reasoning phase, before any real content was ever
written, producing a technically successful API response with
completely empty content. A first fix (`include_reasoning: false` plus
a higher token ceiling) reduced but didn't eliminate the problem: real
data started coming through, but failures got progressively worse
round over round. The reason was subtle: `include_reasoning` only
controls what Groq *returns* in the response, it doesn't reduce how
many tokens the model actually *spends* thinking, and that spend still
counts against `max_tokens`. As the prompt grew each round (more
history to react to, approaching the 5-round window cap), the model
needed more reasoning tokens and kept exhausting the budget before
ever reaching the answer. Properly fixed by adding
`reasoning_effort: "low"`, which caps how much internal reasoning
happens in the first place, combined with raising `max_tokens` to 800
for real headroom either way.

**A first live attempt at the full batch, with all three fixes in
place, surfaced two further, real data-quality issues, both now
fixed:**

**Contaminated seeds were silently entering the statistics.** Of 14
fresh seeds run, 11 completed with zero API failures, but 3 (seeds
512, 513, 514) hit the same cumulative daily quota issue from earlier
in stage 3, with 10%, 77%, and 83% of their decisions falling back to
default prices respectively. `regenerate_summary()` had no way to know
this and would have blended contaminated seeds in with clean ones. It
now checks each seed's `rationale` column for fallback markers and
excludes any seed above a 5% fallback threshold entirely, reporting
exactly which seeds were dropped and why, rather than letting a
partially-failed run quietly corrupt the comparison.

**A more fundamental issue: the model change broke direct comparability.**
The connected batch was collected under `openai/gpt-oss-20b` (the
replacement for the deprecated `llama-3.3-70b-versatile` used for all
16 original isolated runs). The new connected data showed dramatically
different markup levels (12% to 30%) than anything seen with the old
model (which ranged 62.5% to 100% across isolated and connected alike).
That difference could reflect the isolated-vs-connected conditions, or
it could simply be two different models behaving differently, there is
no way to tell without a same-model comparison. `regenerate_summary()`
gained a `min_seed` filter so a summary can be rebuilt containing only
seeds collected under the current model on both sides, and
`experiments/stage6_isolated_completion_runner.py` collects a fresh
14-seed isolated batch under the current model specifically to enable
that fair comparison, using entirely new seed values so neither the
original llama-based isolated data nor the new connected data is ever
overwritten. 5 tests cover the contamination filtering and seed-based
exclusion directly, and 1 covers the reasoning-output request settings.

### The findings

Using only the 14 clean isolated runs and 11 clean connected runs
collected under the same current model, compared with the permutation
test from `stage3_stats.py` and the collusion-proxy metrics from
`collusion_metrics.py`:

| Metric | Isolated | Connected | Difference | p-value |
|---|---|---|---|---|
| Average price (markup over $2.00 cost) | $2.61 (30.4%) | $2.34 (17.2%) | Isolated higher | **0.0004** |
| Price correlation (do agents move together over time) | 0.246 | 0.185 | No real difference | 0.625 |
| Price dispersion (how spread out final prices are) | 0.225 | 0.038 | Connected far tighter | **0.0003** |

**This is the opposite of what the tacit-collusion hypothesis
predicts.** If visibility and messaging caused agents to quietly settle
on a comfortable, elevated price together, connected agents should
have priced *higher* than isolated ones, not lower. Instead, connected
agents priced meaningfully lower, and did so far more consistently
with each other (final prices landing almost on top of one another,
dispersion of 0.038 versus 0.225), while showing no more tendency to
move together round over round than isolated agents did.

The more plausible read: visibility gives an agent something real to
react to, competitors' actual prices, and reacting to that pulls prices
down toward a more competitive level and keeps agents in a tight band
around each other. Isolated agents, with nothing to anchor against,
drift further and more independently, landing both higher and more
scattered.

**Caveats that matter:** only one underlying model was tested
(`openai/gpt-oss-20b`), so this cannot be generalized to LLMs as a
category, a different model could behave completely differently, this
project's stage 3 history is itself a demonstration of that. Three
metrics were compared without correcting for multiple comparisons,
though two of the three are significant by a wide enough margin
(p < 0.001) that this is unlikely to be an artifact of that alone. And
this remains one specific, simplified market configuration (a
multinomial logit demand model, 10 rounds, 3 agents), not a claim about
real-world pricing behavior.

## Roadmap

- [x] **Stage 1**: Deterministic demand model, rule-based agents, validated convergence behavior
- [x] **Stage 2**: LLM-backed agent, first live run against rule-based baselines, one environment bug found and fixed
- [ ] **Stage 3** (superseded by Stage 6): Original multi-agent experiments used a model since deprecated by Groq; see Stage 6 for the valid, same-model comparison
- [x] **Stage 4**: Interactive dashboard for live simulation and browsing recorded results
- [x] **Stage 5**: Dockerized deployment, CI, cost guardrails
- [x] **Stage 6**: Same-model isolated vs connected comparison complete (14 vs 11 clean runs). Finding: connected agents priced lower and converged more tightly than isolated agents, opposite of the tacit-collusion hypothesis
- [ ] **Stage 7**: Write-up of findings, limitations, final polish

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -v                    # run the validation suite
python experiments/stage1_baseline_runner.py   # run the three baseline scenarios
python analysis/plot_baseline.py               # generate the convergence plot
streamlit run dashboard/app.py                 # launch the interactive dashboard
```

## Limitations (honest, as of Stage 3, in progress)

- The demand model is a simplified logit model, not calibrated to any real
  market. It's a controllable testbed, not a prediction of real-world
  prices.
- Rule-based agents are intentionally simple; they exist to validate the
  environment, not to represent realistic firm behavior.
- The LLM agent only sees its last 5 rounds of history each round, not
  its full history. Observed price cycling in earlier runs may be
  partly caused by this limited memory window.
- The environment's random seed does not control the LLM's own
  sampling randomness (temperature 0.7), so repeated runs of the same
  setup produce different outcomes.
- All agents in a given run share the same underlying model. Any
  alignment between them could reflect shared training behavior rather
  than something specific to the market conditions; this is not yet
  disentangled.
- The connected (visibility + messaging) condition does not yet have a
  clean batch of results; two separate attempts were both partially
  contaminated by API rate limiting, though the one seed that completed
  cleanly each time landed on the same result both times. The isolated
  condition's 16-run result is trustworthy on its own, but no
  isolated-versus-connected comparison can be made until a clean,
  larger connected batch exists.
- No claim about collusion or competitive behavior can be made yet,
  that is the purpose of the larger, seeded experiment suite planned
  for a later stage.
