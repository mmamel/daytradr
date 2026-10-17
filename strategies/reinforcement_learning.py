"""
Reinforcement-learning strategy (e.g. PPO/DQN trading agent) — NOT YET
IMPLEMENTED.

RL frames the problem differently than everything else in strategies/: an
agent is trained end-to-end to output actions (buy/sell/hold) that maximize
a reward (e.g. Sharpe ratio) over a simulated trading episode, rather than
being trained to classify a next-bar direction label. That means it needs
an actual gym-style trading environment (state = your feature vector,
action = {buy, sell, hold}, reward = P&L or risk-adjusted P&L) — which in
turn depends on testing_mode.py's backtesting engine existing first. Build
that before this.

Backing research:
  - Théate & Ernst (2021), "An Application of Deep Reinforcement Learning to
    Algorithmic Trading" (arXiv:2004.06627, Expert Systems with
    Applications) — introduces TDQN (a DQN variant adapted for trading),
    trained directly to maximize Sharpe ratio, with promising but
    market/period-dependent results; the authors themselves argue prior RL
    trading papers under-rigorously evaluated performance and propose a
    stricter methodology, which is worth reading before trusting any single
    reported number in this space.
  - Yang, Liu, Chen & Wang (2020), "Deep Reinforcement Learning for
    Automated Stock Trading: An Ensemble Strategy" (Columbia/FinRL) —
    trains PPO, A2C, and DDPG agents and dynamically picks whichever had the
    best recent Sharpe ratio; the ensemble outperformed any single algorithm
    and buy-and-hold/min-variance baselines in their backtest. The "don't
    commit to one algorithm, keep several and pick the current best" idea
    maps directly onto this app's registry pattern.

Caveat that applies to both of the above (and to RL-for-trading papers
generally): Bailey & López de Prado's Deflated Sharpe Ratio work
(SSRN 2460551) shows that backtested Sharpe ratios get inflated by
selection bias (you tried many configs/seeds/hyperparameters and reported
the best one) and non-normal returns — exactly the failure mode RL training
runs are prone to (many random seeds, easy to cherry-pick the lucky one).
Any RL strategy built here should report a deflated/adjusted Sharpe ratio
in backtesting, not a raw one.

To add (once testing_mode.py exists):
    1. `pip install gymnasium stable-baselines3`
    2. Build a minimal trading Env around the backtest engine
    3. Train an agent (PPO is a reasonable first choice per both papers above)
    4. Implement a class here whose predict() calls `self.policy.predict(obs)`
       and maps the discrete action {0, 1, 2} to Signal.SELL/HOLD/BUY
    5. Register it: `@register("ppo")`
"""
