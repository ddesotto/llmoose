# TODO

Build **MooseBench**: a reproducible Mus benchmark and a sandbox for teaching small
language models the rules, distilling stronger players, and improving through play.
Measure rule understanding, gameplay, and inference efficiency separately.

Start with the [learning guide](docs/README.md). Each task below links to an
explanation, evidence, or design. Unchecked items are planned capabilities; proposed
commands in the guides are not installed commands. Existing task IDs are retained.

The [HTML workspace](docs/index.html) presents this roadmap interactively with
browser-local progress and notes. Its checkmarks do not modify this source.

## Current checkpoint

- [x] Deterministic game engine, legal masks, private observations, scripted players,
      spectator CLI, match replay, and hand/match environment modes.
- [x] Per-seat hand episodes and JSONL serialization; these are collection
      foundations, not an implemented optimizer or RL training pipeline.
- [x] LLM transport and policy, versioned prompts, forced-action bypass, exact
      conversation logs, incremental moves, reports, and inspection.
- [x] Basic Python benchmark function with seed suites and team outcomes.
- [x] Optional training dependencies declared; GPU installation/training unverified.
- [x] Repository review and learning artifacts added on 2026-10-14. See the
      [prompt/game audit](docs/game-and-prompt.md) and [evidence snapshot](docs/evidence/review-2026-10-14.json).
- [x] Offline HTML workspace with roadmap filters/checklists, guide reading notes,
      evidence inspection, a synthetic trace explorer, and progress export/import.
      Regenerate with `.venv/bin/python scripts/build_dashboard.py` after source edits.

The medias ranking bug found in the audit was fixed in `1c5baf2`. The default v2
ruleset now implements federation-based deferred scoring and distinct órdago
handling; v1 retains the former scoring for replay. The [rules contract](README.md#rules-contract)
names the remaining benchmark adaptations. Saved runs verify the workflow, not a model ranking: both use seed 12; the
short game won mostly through uncontested juego, while the longer game lost with
all 57 inference decisions syntactically valid. Rule correctness, clear instructions,
and durable artifacts must precede expensive data generation.

## Milestones and dependencies

| Milestone | Tasks | Acceptance gate |
| --- | --- | --- |
| Explain and stabilize play | 09, 10 | Independently checked rules and self-contained player inputs |
| Preserve experiments | 18 | Versioned, linked, recoverable records; existing runs remain readable |
| Compare models | 11, 06, 12 | Two distinct models, mirrored suites, uncertainty, failure accounting |
| Teach an SLM | 13, 14, 04, 05 | Reproducible datasets and improved held-out knowledge/gameplay |
| Learn through play | 15, 07, 08 | Verified RL updates and improvement over the SFT parent |
| Expand and reproduce | 16, 17 | Cross-play, human play, and reusable release artifacts |

TASK-18 can begin alongside the rules audit; its schema must record the eventual
rules and prompt versions. Its core recording/validation gate precedes benchmark
execution; downstream export adapters are implemented with their training tasks.
Existing-model evaluation does not depend on training.
Use development suites during iteration and reserve test suites for final comparisons.

## TASK-09 — Validate the game contract

**Priority:** blocking trusted benchmark outcomes and training labels.
**Guide:** [game and prompt audit](docs/game-and-prompt.md).

- [x] Fix `_evaluate_pares`: choose the rank occurring three times, not the highest
      card. Higher-singleton and normalized-rank regressions are covered in
      `1c5baf2`; `4,4,4,K` now correctly loses to `7,7,7,A`.
- [x] Choose and document the intended Mus variant: federation-based v2 with
      end-of-hand settlement, separate órdago, and a retained legacy v1 path.
- [x] Decide scoring for called/folded pares and juego, showdown timing, public
      card reveals, minimum raises, discards/recycling, and partnership responses.
      Document explicit deviations from the selected reference rules.
- [x] Resolve the engine's equivalence of a numeric bet at the target and órdago.
      Preserve a distinct action/event flag if semantics differ.
- [x] Add independent expected-result fixtures for each comparison, eligibility,
      tie, betting chain, uncontested award, score threshold, and órdago outcome.
      `test_rules_contract.py` states comparison and mano tie outcomes from the
      rules contract; `test_federation_scoring.py` checks eligibility and scoring
      transitions against explicit expected results. Deterministic replay of the
      same engine is not an independent rules oracle.
- [x] Version scoring semantics and reconcile affected docstrings. Preserve old
      trace IDs/files and reject unsupported rulesets. The legacy fixture records
      its source commit; pre-medias-fix traces still need their original evaluator.

**Done when:** a frozen rules specification and independent fixtures agree with
all scoring and transition paths, with known deviations explicitly named.

## TASK-10 — Make the player interface self-contained

**Depends on:** TASK-09 for final semantics.
**Guide:** [game and prompt audit](docs/game-and-prompt.md).

- [x] Add prompt v3 without rewriting v1/v2. Explain four-card hands,
      physical deck composition, normalized rank order, juego threshold, mano,
      explicit table order, the hand lifecycle, and the partnership win condition.
- [x] Define every action and its consequences. Explain pass versus fold, binding
      team responses, the best eligible individual hand, declarations, and
      uncontested lances. State whether intrinsic values add to bets.
- [x] Explain score timing and that displayed totals already include resolved
      awards. Include worked comparisons and a bet–raise–fold example.
- [x] Render player-facing context instead of unexplained enum labels: active
      lance, eligible seats, tie order, bidding team, and recipient of refusal points.
      Mark discard counts as cumulative within the hand; describe contextual actions.
- [x] Preserve same-seat private recall of own discards within the current hand.
      The current cards plus those discards cover cards that seat has seen this
      hand. Keep partners' private memories separate and reset on the next deal.
- [x] Version the renderer/history policy as well as system text. Renderer 3 keeps
      current-hand events complete and summarizes the last three prior public
      hands, explicitly counting older hands omitted.
- [x] Do not provide hand-evaluation summaries; the player must evaluate its own
      cards. Document this assistance boundary in the player-interface contract.
- [x] Check scenarios for coverage and private-information invariance, including
      recall isolation, forced declarations, and near-target situations.

**Done when:** a player can explain all offered actions from its input alone,
without source code or assumed familiarity with a different Mus variant.

## TASK-18 — Durable trace and artifact foundations

**Depends on:** existing recording; coordinate semantics with TASK-09/10.
**Guide:** [artifacts and traces](docs/artifacts.md).
**Format contract:** [run evidence](ARTIFACT_FORMAT.md).

Core recording gate:

- [x] Define a canonical versioned experiment manifest, written before requests,
      and linked decision/attempt/transition/outcome records. Record observations,
      exact messages, structured actions, legal IDs, rewards, and provenance.
- [x] Use stable run, match, hand, seat, decision, attempt, and event identifiers.
      Record all policies' decisions, not only LLM calls; distinguish forced,
      sampled, retried, fallback, failed, interrupted, and unapplied responses.
- [x] Preserve raw v1 `match.json`, `moves.jsonl`, `decisions.jsonl`, and `report.json`.
      Add readers/migrations rather than rewriting evidence. Backfill only facts
      justified by old files, marking unavailable fields unknown.
- [x] Separate evaluator-only state/seed/replay data from seat-visible inputs.
      A seed can reconstruct hidden hands and must not enter a player prompt or
      ordinary student input. Keep each seat's private trace scoped to that seat.
- [x] Store engine/code and dirty-worktree provenance, rules, prompt/renderer/codec
      versions, model/tokenizer revisions, dependency lock, parameters, and hashes.
      Capture provider-returned identifiers and usage without secrets.
- [x] Define append durability, request-start records, crash-safe finalization,
      partial-line recovery, idempotent recovery verdicts, terminal/truncated
      status, and checksums. Journal records are fsynced before provider calls.
- [x] Add validation tools: replay consistency, log joins, mask legality,
      score deltas, missing attempts, schema versions, and hidden-information boundaries.
- [x] Define local run/data/model/evaluation directories, retention, compression,
      backup, indexing, and eventual object-store publishing. Keep small examples
      in Git and large/private experiment artifacts in managed storage.

Downstream adapters, delivered with TASK-13/14/15 and TASK-04:

- [ ] Derive SFT, preference, knowledge-task, and RL datasets from immutable source
      records with export manifests and hashes. Do not overload a replay trace as
      a training trajectory; preserve original records when filtering.

**Core done when:** an experiment can be inspected, replayed, audited, resumed or
explicitly marked partial with traceable provenance. Export acceptance is checked
with the corresponding training task, not required to start benchmarking.

Further hardening:

- [x] Continue an unfinished match from a validated action prefix with the same
      source snapshot. Preserve uncertain provider outcomes and use an explicit
      recovery fallback without silently resending a request.

## TASK-11 — MooseBench protocol and model configuration

**Depends on:** TASK-09, TASK-10, TASK-18 core recording gate.
**Guide:** [benchmark tutorial](docs/benchmark.md).

- [ ] Define the main track: two private instances of the candidate model form a
      partnership against frozen opponents. Keep unfamiliar-partner evaluation separate.
- [ ] Add per-team/seat model specifications and independent provider instances.
      Current CLI/factory shares one provider across all `llm` seats.
- [ ] Freeze manifests for seeds, dealer schedule, opponents, target, repetitions,
      rules, prompts, context policy, and limits. Separate smoke/dev/test suites.
- [ ] Define constrained/unconstrained and reasoning-budget tracks. Record actual
      tokens, cost coverage, latency, and provider settings; a token cap alone is
      not a hardware-independent compute budget.
- [ ] Predeclare invalid-output, fallback, timeout, and nontermination handling.
      Keep model-only results separate from assisted retry/fallback systems.
      Retain infrastructure failures and coverage instead of silently dropping games.
- [ ] Save reproducibility metadata through TASK-18. Hosted calls may vary even
      when environment traces replay exactly.

**Done when:** one manifest fully defines candidate, opponent, information access,
inference budget, and scoring protocol.

## TASK-06 — Mirrored evaluation and comparisons

**Depends on:** TASK-11. Expands the original `feat/mirrored-eval` task.
**Guide:** [benchmark tutorial](docs/benchmark.md).

- [ ] Run every seed/dealer configuration with both partnership assignments;
      attribute results to policy identity rather than seat-team number.
- [ ] Pair initial environment randomness. Explain that later discards/actions
      can diverge, so mirroring does not guarantee identical complete trajectories.
- [ ] Add frozen-opponent comparisons and model round robins; scripted baselines
      are diagnostic opponents, not proof of broad playing strength.
- [ ] Bootstrap at the seed-group level, retaining mirrors/rotations/repeats
      together. Report win rates and paired candidate differences with uncertainty.
- [ ] Report primary win/loss plus stone differential, per-lance outcomes,
      contested/uncontested awards, legality, fallbacks, failures, usage, and latency.
      Distinguish legacy órdago overshoots from v2's terminal score encoding.
- [ ] Provide named suites, machine-readable reports, and a `llmoose-bench` CLI.
      Predeclare sample sizes or precision goals; smoke runs are not leaderboard evidence.
- [ ] Test attribution, pairing, and aggregation. Identical deterministic policies
      with mirrored identity assignments should produce exactly 50%, not a flaky
      “near 0.5” assertion. Test stochastic behavior separately.

**Done when:** two different models complete comparable mirrored suites with
correct attribution, uncertainty, failure accounting, and linked artifacts.

## TASK-12 — Resumable experiment execution

**Depends on:** TASK-18, TASK-11, TASK-06.

- [ ] Schedule model × opponent × seed × mirror × repeat jobs with stable IDs,
      isolated logs, bounded concurrency, request/rate limits, and cost caps.
- [ ] Resume completed jobs without repeating paid requests. Specify whether
      partial games resume from saved transcripts or restart as distinct attempts.
      A call interrupted before its response was durably recorded has an unknown
      outcome; do not promise exactly-once inference or invent its response/cost.
- [ ] Add bounded transient transport retries/backoff, cancellation, explicit
      partial statuses, and aggregation without double counting.
- [ ] Export comparison tables/JSON/CSV with links to evidence and cost coverage.

**Done when:** interrupted batches retain their evidence and resume predictably.

## TASK-13 — Actual knowledge and decision evaluation

**Depends on:** TASK-09, TASK-10, TASK-11, TASK-18.
**Guide:** [training tutorial](docs/training.md).

- [ ] Generate rule-grounded probes: normalization, comparisons, eligibility, ties,
      action consequences, refusal points, scoring, and near-target decisions.
- [ ] Separate factual correctness, output compliance, teacher imitation, and
      strategic quality. The engine knows outcomes, not an optimal action under
      hidden information merely because it knows the actual deal.
- [ ] Build strategic scenarios using justified decisions or estimated values under
      explicit hidden-hand beliefs/opponent policies, with uncertainty and multiple
      acceptable choices where appropriate.
- [ ] Hold out deal families and paraphrases; test transfer across seats, scores,
      equivalent physical cards, unfamiliar partners, and opponent styles.
- [ ] Compare full-rule, compact-rule, and no-rule probes as distinct tracks for
      internalized knowledge. Keep full-rule competitive evaluation available.

**Done when:** improvements in rules, format, and unseen gameplay can be measured
separately rather than inferred from fluent explanations or a lucky win.

## TASK-14 — Distillation and versioned datasets

**Depends on:** TASK-18, TASK-10, TASK-13; reuses TASK-02.
**Guide:** [training tutorial](docs/training.md).

- [ ] Collect heuristic, strong-teacher, and later learned-checkpoint decisions
      with only the acting seat's information. Keep engine factual labels separate.
- [ ] Export complete system/user messages and assistant targets; observation-only
      export loses the rules. Filter invalid/fallback labels and balance forced turns.
- [ ] Split by seed/deal family before collection: all seats, mirrors, and
      continuations stay together. Deduplicate and balance phases, stakes, positions,
      hand strengths, and score conditions.
- [ ] Label student-reached states with teachers to address distribution shift.
      Preserve teacher identity, prompt/config, cost, and action provenance.
- [ ] Treat provider reasoning as optional, fallible evidence. Any concise rationale
      targets need checks against visible facts and the frozen rules.
- [ ] Explore action-value/preference distillation using sampled hidden worlds
      consistent with observations and controlled opponents, not hindsight labels
      derived solely from a lucky actual hand or an omniscient teacher.

**Done when:** datasets reproduce from immutable sources and export manifests,
with clean splits, traceable teachers, and validated information boundaries.

## TASK-04 — SLM supervised fine-tuning

**Depends on:** TASK-03, TASK-14; evaluates through TASK-06 and TASK-13.
**Guide:** [training tutorial](docs/training.md).

- [x] Shared action completion contract: `<answer>ID</answer>`.
- [ ] Implement SFT export/training entry points with configurable checkpoint,
      tokenizer, LoRA/full tuning, precision, context, batching, seeds, and resume.
- [ ] Validate dependencies on available hardware; measure memory and throughput
      before choosing a model size or distributed infrastructure.
- [ ] Apply the chat template and assistant-target loss mask correctly. Prevent
      truncation of current cards, rules, and legal choices.
- [ ] Curriculum: rules/format, diverse demonstrations, then score-aware gameplay.
      Target-8 smoke games do not substitute for target-40 endgame decisions.
- [ ] Save/reload weights/adapters, tokenizer, configs, optimizer state, and dataset
      provenance through the same inference interface used by the benchmark.
- [ ] Compare base, format-only, heuristic/teacher-distilled, and knowledge-plus-play
      variants. Select checkpoints on development data, not the final test suite.

**Done when:** a reproducible SLM artifact improves over its untuned baseline on
held-out knowledge and gameplay, with resource use reported.

## TASK-05 — Constrained action generation

**Depends on:** TASK-10; can run alongside TASK-04.

- [ ] Implement a grammar or tokenizer-aware prefix automaton over complete legal
      answers. A set of permitted digit tokens allows illegal combinations/prefixes.
- [ ] Add local inference and test tokenizer boundaries, multi-token IDs, and endings.
- [x] Strict policy mode disables invalid-output retries and conservative fallback;
      invalid responses and provider failures are separately recorded.
- [ ] Keep constrained/unconstrained benchmark tracks separate. For on-policy RL,
      never silently substitute a baseline action; record probabilities under the
      actual constrained sampling distribution when using constraints.

**Done when:** every constrained answer is legal and unassisted format reliability
remains independently measurable.

## TASK-15 — RL environment and learning records

**Depends on:** TASK-09, TASK-10, TASK-18.
**Guide:** [training tutorial](docs/training.md), [artifacts](docs/artifacts.md).

- [ ] Add explicit termination/truncation, limits, policy reset hooks, and independent
      RNG streams for deals, opponents, and model sampling. Bound repeated Mus rounds.
- [ ] Add a driver for learner/partner/opponent policies supporting one-seat and
      partnership training, preserving separate private recall.
- [ ] Specify reward modes: current step rewards are per-team score deltas;
      hand trajectories use zero-sum differential. Use match outcomes for the
      competitive objective and evaluate hand rewards/shaping separately.
- [ ] Support full-match rollouts and validated score-conditioned starts. Hand
      episodes currently start at zero scores and miss important endgames.
- [ ] Record sampled tokens, behavior log probabilities, legal/action/token masks,
      policy revision, turn ownership, rewards, and terminal/truncated flags.
      Exclude forced actions from policy-gradient decisions.
- [ ] Isolate any privileged centralized critic inputs from deployed player inputs.

**Done when:** records support a verifiable RL update against interchangeable
partners/opponents without hidden-information leakage.

## TASK-07 — Batched rollouts

**Depends on:** TASK-05, TASK-15.

- [ ] Batch pending decisions by model across independent environments while
      preserving seat identity, memory, and action routing; refill completed games.
- [ ] Measure throughput/memory and track policy revision, rollout age, and queues.
      Begin with frozen behavior policies during synchronous collection.
- [ ] Verify deterministic scalar/batched equivalence with separate RNG streams.
      Scale batch size to measured capacity rather than assuming 64 fits.

**Done when:** throughput improves without changing transition semantics or
corrupting behavior probabilities and private observations.

## TASK-08 — RL training and self-play

**Depends on:** TASK-04, TASK-06, TASK-07, TASK-15.
**Guide:** [training tutorial](docs/training.md).

- [ ] Implement one explicit policy-gradient objective from the SFT parent against
      frozen opponents, with masks, credit assignment, regularization, and resume.
- [ ] Choose the algorithm before naming it GRPO. Subtracting a mano-conditioned
      EMA alone is not GRPO; group-relative training needs comparable rollouts,
      defined randomness, normalization, and zero-variance handling.
- [ ] Verify update signs, team/seat rewards, log probabilities, forced-turn
      exclusion, truncation, and reference/KL terms on controlled tiny trajectories.
- [ ] Evaluate position-conditioned baselines and reward choices as ablations.
      Log entropy, action diversity, KL, gradient norms, reward distributions,
      invalids, output lengths, cost, and gameplay by phase/position.
- [ ] Evaluate every K updates on frozen development opponents. Require improvement
      over the SFT parent, not merely a weak heuristic. Use multiple training seeds
      and final held-out evaluation with uncertainty.
- [ ] Add frozen checkpoint pools/self-play after the fixed-opponent loop works;
      compare SFT, fixed-opponent RL, and league training.

**Done when:** RL reproducibly improves held-out gameplay over its SFT parent
without losing rule understanding or action reliability.

## TASK-16 — Opponent league and human play

**Depends on:** TASK-11, TASK-15; learned players arrive from TASK-08.

- [ ] Maintain immutable scripted/hosted/checkpoint opponent pools and recorded
      sampling weights; retain diverse styles and freeze opponents within batches.
- [ ] Evaluate unfamiliar partners and cross-play between checkpoints; report
      head-to-head matrices and nontransitive results before optional Elo/TrueSkill.
- [ ] Add terminal `HumanPolicy` with only the assigned seat's view. The all-hands
      spectator display is unsuitable for a human player in a fair match.
- [ ] Add a versioned external-agent protocol, timeouts, and invalid-action handling.
      Network multiplayer is a later layer over the same boundary.

**Done when:** compatible bots, models, and a human can occupy arbitrary seats,
and training/evaluation reproducibly samples partner/opponent pools.

## TASK-17 — Reproducible release

**Depends on:** TASK-12; include training outputs when available.

- [ ] Publish rules, manifests, baselines, instructions, schemas, failure policy,
      representative replays, and results with evidence links.
- [ ] Add dataset/model cards covering splits, teachers, objectives, assistance,
      compute, licenses/usage constraints, and limitations.
- [ ] Provide a local smoke path: collect, tiny SFT, reload, benchmark, tiny RL,
      reload. Keep hardware checks separate from CPU CI.
- [ ] Release benchmark-only results independently of training experiments.

**Done when:** another user can reproduce a baseline, load a checkpoint, and
run the documented protocol from preserved artifacts.

## Completed foundations

### TASK-01 — Hand episodes

- [x] Match/hand modes, direct seeded-hand reset, hand-boundary termination,
      per-hand score deltas, and deterministic-deal tests.

### TASK-02 — Trajectories

- [x] Per-seat turns with prompts, legal/chosen IDs, position, and hand returns;
      JSONL round trip and partnership-return tests.
- [x] Current return is undiscounted hand differential stored once per seat episode.
      A trainer may broadcast it to that seat's turns; that credit-assignment choice
      still needs validation as a learning objective.

### TASK-03 — Training dependencies

- [x] Optional training dependencies declared; obsolete transformer policy removed.
      Hardware/runtime validation is tracked in TASK-04.

## Later backlog

- Direct provider integrations when the compatible transport is insufficient.
- Gymnasium/PettingZoo adapters once the internal RL contract is stable.
- Distributed/cloud workers after local training and batching are measured.
- Señas or alternate Mus variants as separately versioned environments.
- Event-log storage optimization if profiling establishes a bottleneck.
