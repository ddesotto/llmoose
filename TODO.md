# MooseBench roadmap

**Priority: run existing LLMs against each other and publish a provisional Mus
leaderboard. Training is a later project that can reuse selected run evidence.**

Updated 2026-10-18. Aim for a small public result today, then expand overnight
only after measuring latency/cost and setting spending limits. Publication is a
milestone, not a promise of statistically decisive rankings by a deadline.
Existing task IDs are retained for links from the learning guides.

## What works now

- [x] Versioned Mus rules, private observations, legal actions, scripted players,
      and independent expected-result rules/scoring tests (TASK-09/10).
- [x] LLM play with exact conversations, strict output mode, per-seat providers,
      and distinct model IDs/endpoints for opposing partnerships (TASK-11).
- [x] Mirrored two-model suites, per-pair bootstrap intervals, completion coverage,
      and per-game artifacts through `llmoose-bench` (TASK-06).
- [x] Replay, validation, durable records, source snapshots, and compatible-source
      resume (TASK-18). A recovered uncertain request can use a logged fallback.
- [x] Individual-game terminal spectator mode and completed-game replay.
- [x] Hand episode collection exists; no model training is implemented.
- [x] Frozen development environment contract, 46 authored diagnostic cases,
      model-neutral prediction grading, and separate deal allocations. See
      [DEVELOPMENT_SUITE.md](DEVELOPMENT_SUITE.md).

The benchmark runner currently handles one model pairing per invocation. A
round-robin scheduler, combined leaderboard, Elo ratings, spending caps, and a
live browser viewer are still to be built. Training dependencies do not establish
hardware readiness. The learning HTML workspace is not a live tournament viewer.

## Today: a small provisional leaderboard

### TASK-11 — Freeze one simple competition

- [ ] Start with three named LLMs. Each model controls both seats of its partnership
      through separate private observations; this measures partnership strength.
- [ ] Freeze exact model IDs/endpoints, ruleset, prompt version, target 40,
      temperature, output-token limit, timeout, step limit, and source snapshot.
      Record requested and returned model identities. Label provider defaults or
      unobservable reasoning settings; equal token caps do not mean equal compute.
- [ ] Use strict mode for the first track. Report invalid output, provider errors,
      step limits, interruption, and recovery fallback separately. Recovered games
      containing substituted actions belong outside the unassisted score.
- [ ] Fix the smoke and publication seed lists before running. Run each unordered
      model pair once per seed with both partnership assignments. Keep the existing
      dealer convention fixed and disclose it; dealer rotations can follow later.
- [ ] Freeze a future evaluation seed pool separately from games intended for
      training. All seats, mirrors, and continuations from a deal stay together.

**Done when:** one short manifest defines the whole competition and its failure
policy. This release measures gameplay under the supplied rules and legal menu;
it does not independently measure internalized rule knowledge.

### TASK-12 — Run and resume the games

- [ ] Let current benchmark integration finish; require the test suite to pass.
      Check model attribution, mirroring, resume, and failure accounting.
- [ ] Smoke-test every pairing on one mirrored seed pair. Inspect conversations,
      validate runs, and measure actual requests, latency, tokens, and reported cost.
      Target-8 smoke results must remain separate from target-40 leaderboard games.
- [ ] Add a thin round-robin queue over the existing pair runner: stable job IDs,
      distinct output directories, progress, and resume. Start sequentially;
      add small bounded concurrency only if the measured runtime requires it.
- [ ] Add request and wall-clock limits plus a spending guard before unattended
      execution. Use a provider-side hard budget where available; missing usage
      is unknown, not free, and reported cost may arrive after a request completes.
- [ ] Predeclare a pilot of 5 mirrored seeds per pairing: three models produce
      3 pairings × 5 seeds × 2 assignments = 30 games. Reduce the planned pilot
      before starting if smoke measurements exceed the available time/budget.
- [ ] Preserve a frozen executable source checkout during collection. Resume
      requires matching source; code/config changes start a new suite identity.

**Done when:** every planned game has a durable completed or explicit incomplete
status, and restarting the queue reuses validated work without duplicating scores.
The current pair runner is usable manually while the queue is being implemented.

### TASK-06 — Build the results table

- [ ] Aggregate all pair reports into JSON/CSV and a readable leaderboard. Show
      model identity, completed/planned games and mirrored pairs, win rate,
      uncertainty, invalids/failures, tokens, latency, and cost-reporting coverage.
- [ ] Show a head-to-head matrix. Compute overall score as equal-weight mean win
      rate against the same opponent set; display incomplete models as provisional
      or unranked rather than treating missing pairings as wins or losses.
- [ ] Keep the existing complete-mirrored-pair gameplay score, but expose all
      excluded games and reasons next to it. Low completion can bias that score;
      do not present a reliable total order when coverage differs substantially.
- [ ] Resample seed groups together across the tournament for aggregate uncertainty;
      preserve mirrors and repeated uses of the same seed. Tiny-sample intervals
      may be misleading, so show counts and label the pilot provisional.
- [ ] Optional after the table works: add Elo with documented initial rating,
      update constant, deterministic game ordering, and included outcomes.
      Recompute from immutable results; retain the head-to-head matrix because
      a single rating can hide matchup-specific strengths. Elo is not today's gate.

**Done when:** the published numbers trace back to validated games and distinguish
observed wins, incomplete coverage, and statistical uncertainty.

### TASK-17 — Publish the first result

- [ ] Publish a simple static table or README with protocol, date, exact model
      identities, source revision/hash, sample counts, limitations, and selected replays.
      A custom website is optional; do not make it the publication blocker.
- [ ] Label it **provisional / pilot**. A handful of seeds supports an initial
      observation, not a definitive model ranking or a claim of expert Mus play.
- [ ] Export a public view of selected evidence. Keep raw private run directories
      separate from public pages and from player inputs.
- [ ] Track authored learning guides in Git; `docs/` is currently ignored despite
      README links. Decide separately whether generated HTML belongs in Git.

**Done when:** someone can read the results, understand the experiment, and inspect
representative games without needing the private local workspace.

## Overnight and next: more evidence and a viewer

### Expand the frozen competition (continues TASK-12)

- [ ] After the pilot, predeclare the next seed batch using measured cost/runtime.
      For three models, 20 mirrored seeds per pairing means 120 games in total;
      this is a planning example, not a guaranteed overnight capacity or precision.
- [ ] Keep models/settings fixed, retain the pilot snapshot, and publish updated
      coverage and uncertainty. Larger batches do not repair protocol differences.
- [ ] Add bounded backoff for transient transport failures and cancellation; never
      silently resend a request whose outcome is unknown after a crash.

### TASK-16 — Watch games and inspect decisions

- [x] Watch an individual game with `llmoose-play --watch`; inspect flushed records
      during a run and replay completed games with `llmoose-inspect --replay`.
- [ ] Add a read-only browser viewer that tails/projects existing run records:
      active matchup, scores, phase, cards, bids, current actor, and recent actions.
      It must tolerate incomplete final lines and must not issue model requests.
- [ ] Add pause/step/speed controls for replay and a seat-view/spectator toggle.
      The all-hands spectator view is evaluator-only and never a player input.
- [ ] Link leaderboard rows to selected matches. Start with local viewing; public
      live hosting and human multiplayer are separate later work.

**Done when:** a running game is understandable visually and a finished result
can be explored one decision at a time. Viewing must not block game execution.

## Later: learn from the games

### TASK-14 — Preserve reusable evidence now, export later

Uses the existing TASK-18 recording foundation.

- [x] Preserve exact messages, chosen/applied actions, model/prompt/rules provenance,
      outcomes, retries/fallbacks, and source hashes in raw run artifacts.
- [ ] Before training, select suitable demonstrations and export structured chat
      messages plus assistant targets with source IDs/hashes and teacher identity.
      Exclude invalid/fallback labels and distinguish forced decisions.
- [ ] Use games allocated to training/development for demonstrations. Once a game
      enters training, it cannot remain held-out evaluation for that student.
      Collect fresh evaluation games when needed; never split individual turns
      randomly across train/test while related deals appear in both.
- [ ] Do not label every winning action as good strategy. These runs are candidate
      demonstrations, not optimal-action labels or complete on-policy RL records.

### TASK-04 — One small supervised experiment

Includes TASK-13 knowledge evaluation.

- [ ] Add verified rule/state probes separate from gameplay evaluation.
- [ ] Choose one model/tokenizer and one training recipe after checking hardware.
      Separate training dependencies from optional serving dependencies.
- [ ] Export a tiny dataset, verify assistant-only loss masking and input visibility,
      deliberately overfit it, save/reload, and play through the same policy interface.
- [ ] Evaluate on unseen deal groups against the untuned parent. Only then expand
      the dataset or compare better teachers. Include full-match score contexts.

### Deferred research (TASK-05/15/07/08)

Constrained generation, explicit terminal/truncation and policy-reset contracts,
behavior token probabilities, reward design, batched rollouts, RL updates, and
self-play leagues follow a working supervised baseline. Saved hosted-model games
alone do not supply the learner probabilities needed for on-policy RL.

### Existing foundations (TASK-01/02/03/09/10)

Keep the hand/match environment, episode serialization, rules regression tests,
private-observation boundary, and versioned prompts. Improve them when a concrete
experiment exposes a gap. Defer broad refactors, distributed infrastructure,
network multiplayer, and alternate rulesets until they serve a measured need.
