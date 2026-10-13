# llmoose

`llmoose` is a research environment for training and benchmarking SLLMs on the Spanish card game Mus.


# Code

The project is structured as follows:

```text
src/llmoose/
  rules/          cards, pure hand comparison, versioned rulesets
  game/           immutable state, reset/step environment, legal transitions
  observations/   one private, action-masked view per seat
  agents/         policy protocol; random and heuristic baselines
  simulation/     seeded matches, JSON traces, deterministic replay
  benchmark/      fixed-deal suites and aggregate match reports
```


# Play

Install the local package (Python 3.11+):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Watch the heuristic partnership play against the conservative partnership:

```bash
llmoose-play --seed 12 --target-score 8 --team-0 heuristic --team-1 conservative --watch
```

The spectator view shows all four hands, mano, each decision, and scoring as it
happens. Team 0 is North/South; team 1 is East/West. Agents still receive only
their own private cards and public information.

Use `--step` to advance one action at a time with Enter (`q` quits), or
`--delay 0.1` to change the automatic pace. `--delay 0` prints the whole game
immediately. Both options enable the spectator view.

```bash
llmoose-play --seed 12 --target-score 8 --policy heuristic --step
llmoose-play --seed 12 --target-score 8 --policy random --watch --delay 0.1 --trace match.json
```

Available policies are `conservative`, `heuristic`, and `random`. `--policy`
sets all seats; `--team-0` and `--team-1` override individual partnerships.
These are baseline agents; an LLM adapter is still to come.

Without spectator options, the command prints only the final result. `--trace`
saves a completed match for deterministic replay. Stopping early does not save
a partial trace.

You can also run directly from the checkout without installing:

```bash
PYTHONPATH=src python -m llmoose.cli --seed 12 --target-score 8 --policy heuristic --watch
```


# References
- [RLCard](https://github.com/datamllab/rlcard),
- [DeepStack](https://pubmed.ncbi.nlm.nih.gov/28254783/)
- [ReBeL](https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html)
- [SPIRAL](https://arxiv.org/abs/2506.24119) (self-play multi-turn RL on text games, Qwen3 base models),
- [TextArena](https://github.com/TextArena/TextArena) (Gym-style text-game interface)
- [Liquid's TRL guide](https://docs.liquid.ai/lfm/fine-tuning/trl)
