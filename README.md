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

## v0 - game environment, heuristic policy

```console
$ llmoose-play --seed 12 --target-score 4 --policy heuristic --trace match.json
winner=team-0 scores=(6, 2) actions=21 events=30
```


# References
References: Poker work such as [RLCard](https://github.com/datamllab/rlcard),
[DeepStack](https://pubmed.ncbi.nlm.nih.gov/28254783/), and
[ReBeL](https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html)
provide a previous example supporting this model.
