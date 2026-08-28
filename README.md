# llmoose

`llmoose` is a research sandbox for building and benchmarking LLM agents for the Spanish card game Mus.


# Architecture (planned)

```text
src/llmoose/
  rules/          cards, hand comparison, scoring, versioned rulesets
  game/           game state, legal actions, state transitions
  observations/   one public/private view per seat
  agents/         random, heuristic, LLM, and future search agents
  simulation/     seeded matches, traces, replay
  benchmark/      fixed deal suites, metrics, comparison reports
```