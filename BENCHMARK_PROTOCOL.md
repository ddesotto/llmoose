# MooseBench: first runnable slice

`llmoose-bench` currently measures one **candidate partnership** against one
frozen opponent. Either side may be `conservative`, `heuristic`, `random`, or an
LLM served through the existing chat-completions transport. Two LLM sides may
use different model IDs and API endpoints. Each seed runs twice with the
candidate on team 0 and team 1. All four seats get independent policy objects
and every LLM seat gets its own provider instance. Prompt, temperature, token
limit, and strict/assisted mode are shared within a suite.

The run directory contains `benchmark.json`, `summary.json`, and a validated v2
run for each seed and assignment. Rerunning the same command reuses completed
games and resumes partial ones using the saved journal. The manifest pins source,
rules, prompt settings, seeds, opponent, and inference mode. Use a new output
directory after changing any of these. Do not publish raw run directories without
review: they contain private hands and exact model conversations.

The primary score is the candidate's win fraction across **complete seed pairs**.
An incomplete half-pair remains visible in coverage and job rows and contributes
no fabricated loss. The 95% percentile bootstrap resamples seed pairs, with a
fixed bootstrap seed, so the two mirrored games stay together. Stone margin is
reported from the candidate's perspective. Small smoke suites and intervals are
plumbing checks, not reliable model rankings. Hosted model output may vary even
when the environment setup is identical.

The first release intentionally omits dealer rotations, request/cost caps,
batch concurrency, and a held-out benchmark suite.
It also does not claim that the current heuristic is strong Mus play. These are
the next TASK-11/06/12 steps before a public model comparison.

## Game-AI design precedents

- [CallMus](https://github.com/JuanMartinFajardo/muson_line) is a **two-player**
  Mus project. It separates Mus/discard and betting decisions, uses hand win
  probability features, and reports that imitation with random forests was much
  weaker for betting than for the earlier decisions. Its rules and results are
  not interchangeable with this four-seat partnership benchmark.
- [Catanatron](https://docs.catanatron.com/) illustrates the progression from
  fast simulation and interpretable bots to learned agents and a watchable UI.
- [IRIS Mus](https://github.com/intersystems-ib/iris-mus) demonstrates a local
  4B-class LLM as a decision adviser with the game backend validating actions.
  Its repository is an architecture example, not a measured strength baseline.
- A [team Briscola MCTS thesis](https://is.muni.cz/th/s8h0x/) addresses hidden
  information and cooperation, both relevant when stronger Mus opponents are
  added. Search must sample hidden hands consistent with each player's view.
- A [card-game imitation study](https://projekter.aau.dk/learning-based-decision-making-in-a-competitive-card-game-ca046171.html)
  found that copying a search teacher can be limited by information the student
  never receives. A future Mus dataset must keep acting-seat information and
  teacher-only search features separate.

