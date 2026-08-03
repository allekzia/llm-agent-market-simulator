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

> **Status: Stage 1, deterministic core validated.**
> LLM agents land in the next stage. See [Roadmap](#roadmap) below.

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
├── core/
│   ├── demand_model.py         # deterministic multinomial logit demand + profit
│   ├── environment.py          # round loop connecting agents to the market
│   └── agents/
│       ├── base.py             # Agent interface (Observation to Decision)
│       └── rule_agent.py       # baseline strategies: cost-plus, undercut, noisy-match
├── experiments/
│   └── stage1_baseline_runner.py  # scenario runner + CSV logging
├── analysis/
│   └── plot_baseline.py        # convergence visualization
├── tests/
│   └── test_demand_model.py    # 10 tests validating economic sanity
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

## Roadmap

- [x] **Stage 1**: Deterministic demand model, rule-based agents, validated convergence behavior
- [ ] **Stage 2**: LLM-backed agent (structured JSON decisions), single LLM agent vs. rule-based competitors
- [ ] **Stage 3**: Multi-LLM-agent markets, information-condition experiments (full/partial visibility, agent-to-agent messaging)
- [ ] **Stage 4**: Streamlit dashboard to configure and visualize experiments live
- [ ] **Stage 5**: Dockerized deployment, CI (GitHub Actions), cost guardrails for public demo
- [ ] **Stage 6**: Full experiment suite, seeded runs across conditions, collusion-proxy metrics
- [ ] **Stage 7**: Write-up of findings, limitations, final polish

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -v                    # run the validation suite
python experiments/stage1_baseline_runner.py   # run the three baseline scenarios
python analysis/plot_baseline.py               # generate the convergence plot
```

## Limitations (honest, as of Stage 1)

- The demand model is a simplified logit model, not calibrated to any real
  market. It's a controllable testbed, not a prediction of real-world
  prices.
- Rule-based agents are intentionally simple; they exist to validate the
  environment, not to represent realistic firm behavior.
- No LLM agents yet, so the interesting research question hasn't been
  tested yet. This section will be replaced with real findings and their
  limitations once later stages are complete.
