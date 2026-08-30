# TODO


## TASK-01 — `feat/hand-episodes`

**Depends on:** nothing.

An episode is currently a whole match to 40 points, with only the win/loss bit of information.

`deal_hand` in `src/llmoose/game/state.py` already derives the deal from
`Random(f"{seed}:{hand_number}")`.

We just need to expose it in `MusEnv`.

- [X] Add `MusEnv(episode="match" | "hand")`, defaulting to `"match"` so
      nothing existing changes behaviour.
- [X] In `"hand"` mode, terminate when `hand_number` increments or the match
      completes, whichever comes first.
- [X] Report the hand's stone differential per team in `StepResult.info`
      (`hand_reward`), separate from the existing per-transition `reward`.
- [X] Add `MusEnv.reset(seed=..., hand_number=...)` so a specific hand can be
      dealt directly, without playing the hands before it.
- [X] Test: the same `(seed, hand_number)` deals identical hands regardless of
      how the match reached that point.
- [X] Test: a `"hand"` episode terminates within one hand and its `hand_reward`
      sums to the score change.

**Done when:** a caller can loop `reset(seed=n, hand_number=k)` -> play -> read a team reward, on a per-hand basis.

---

## TASK-02 — `feat/trajectories`

**Depends on:** TASK-01.

`MatchTrace` is a replay artifact with actions only - no prompts. 

For training we need:
- what the model saw
- what it chose
- what it got. 

Keep them as separate types; do not overload `MatchTrace`.

Followinbg SPIRAL: terminal return broadcast to every one of that seat's turns, **no discounting**.

- [ ] New `src/llmoose/training/trajectories.py` with a `TurnRecord`
      (`seat`, `mano_offset`, `phase`, `prompt_text`, `legal_ids`, `chosen_id`)
      and an `Episode` (`turns`, `team_return`, `seed`, `hand_number`).
- [ ] `mano_offset = (seat - mano) % 4` — position relative to mano, not the
      absolute seat. TASK-08 keys its baseline on this.
- [ ] `collect_episode(env, policies) -> tuple[Episode, ...]`, one episode per
      seat, sharing the same played hand.
- [ ] Broadcast the hand's team stone differential to every turn that seat
      played; store the raw differential, do not normalize here.
- [ ] JSONL writer/reader — one episode per line, so datasets append cheaply.
- [ ] Test: turn counts per seat sum to the hand's action count.
- [ ] Test: returns are equal and opposite between the two partnerships.

**Done when:** self-play with the existing `HeuristicPolicy` produces a JSONL
file of episodes that a `datasets.load_dataset("json", ...)` call can read.

---

## TASK-03 — `chore/train-extras`

**Depends on:** nothing.

- [ ] Add `trl` and `vllm` to the `train` optional-dependency group in
      `pyproject.toml` .
- [ ] Remove `src/llmoose/agents/transformer.py`..

**Done when:** `pip install -e '.[train]'` gives a working TRL + vLLM environment.

---

## TASK-04 — `feat/sft-bootstrap`

**Depends on:** TASK-02, TASK-03.

Distil `HeuristicPolicy` into the SLM before any RL. 

Why? Adheres to formatting, gives RL a non-random starting policy. 

- [ ] `llmoose-export-sft` CLI: run N seeded self-play hands, write
      `{"prompt": <rendered observation>, "completion": "<answer>ID</answer>"}`
      pairs from `render_observation` in `src/llmoose/agents/prompt.py`.
- [ ] Fix the completion format now and reuse it unchanged in TASK-05 and
      TASK-08 — a format change mid-project invalidates every checkpoint.
- [ ] Include `SYSTEM_PROMPT` as the system turn; add seat and mano-offset to
      it so the model is role-conditioned from the start (SPIRAL uses
      role-specific system prompts).
- [ ] Generate at `target_score=8`: 39 actions and 31 decisions/match, versus
      103 and 167 at 40. Evaluate at 40 later.
- [ ] Reference LoRA SFT script under `scripts/` using TRL `SFTTrainer`.
- [ ] Test: exported records round-trip, and every `completion` names an id
      that was in that turn's legal set.

**Done when:** a LoRA-tuned LFM2-350M emits a parseable, legal action id for
the large majority of prompts in a held-out set.

---

## TASK-05 — `feat/constrained-decoding`

**Depends on:** TASK-04.

The action space is small integers, so mask the logits and illegal actions become impossible.

This forces split, the retry-and-fallback added to `StructuredLLMPolicy` is correct for **eval** (it measures format compliance) and wrong for **training** (it silently rewrites the model's action and corrupts on-policy data).

- [ ] `src/llmoose/agents/decoding.py`: build a per-turn allowed-token set from
      `observation.legal_action_mask` and the tokenizer, for the digits inside
      `<answer>…</answer>`.
- [ ] Local HF/vLLM provider implementing the `LLMProvider` callable, applying
      the mask.
- [ ] Document and test the two modes explicitly: eval uses
      `max_retries=2, fallback_on_invalid=True`; rollouts use
      `max_retries=0, fallback_on_invalid=False`.
- [ ] Keep `PolicyTelemetry.invalid_outputs` reported in both modes — for a
      350M model this is the primary early metric.
- [ ] Test: with masking on, a provider that always samples the highest-logit
      token still returns a legal action.

**Done when:** a rollout at `max_retries=0` over 100 hands records zero invalid
outputs, and the unmasked baseline's invalid rate is recorded for comparison.

---

## TASK-06 — `feat/mirrored-eval`

**Depends on:** TASK-01.

Mus variance is high enough that an RL win-rate curve without duplicate deals is
noise. 
We need something trustworthy to measure against. 

- [ ] `run_benchmark(..., mirror_seats=True)`: replay every seed with the two
      partnerships swapped, so both policies see identical deals from both
      sides.
- [ ] Attribute wins to the policy pair rather than the seat index —
      `BenchmarkReport.team_wins` currently means "team 0 / team 1" and stops
      being meaningful under swapping. Change the report shape deliberately and
      update its docstring.
- [ ] Add a bootstrap confidence interval on the win rate and mean score
      difference.
- [ ] Per-lance win-rate breakdown — `lance_counts` already exists but is not
      split by policy.
- [ ] Named public suites (`SUITE_SMOKE`, `SUITE_STANDARD`) so results are
      comparable across runs and checkpoints.
- [ ] Test: mirroring a policy against itself gives a win rate near 0.5 with the
      CI containing 0.5.

**Done when:** `llmoose-bench --a heuristic --b conservative --mirror` prints a
win rate with a CI, on a fixed named suite.

---

## TASK-07 — `feat/batched-rollouts`

**Depends on:** TASK-05.

One decision per generate call wastes almost all of a GPU. 
Envs are immutable dataclasses and cheap to hold in bulk, so batching is mostly bookkeeping.

- [ ] `VectorMusEnv` holding N independent `MusEnv` instances at
      `episode="hand"`.
- [ ] Rollout worker: gather the pending decision from every live env, render
      all prompts, issue one batched `generate`, scatter the actions back.
- [ ] Handle envs finishing at different times — refill with the next seed
      rather than blocking on the slowest.
- [ ] Record throughput (hands/sec, decisions/sec) so regressions are visible.
- [ ] Test: batched rollout over N envs produces byte-identical episodes to
      running the same seeds one at a time with a deterministic policy.

**Done when:** N=64 rollouts run with one generate call per decision round and
match single-env output exactly.

---

## TASK-08 — `feat/grpo-selfplay`

**Depends on:** TASK-02, TASK-06, TASK-07.

Self-play GRPO with a shared policy across all four seats. `run_match` already
takes a per-seat policy mapping and `MusEnv` already returns the acting seat's
observation, so shared-policy self-play needs no engine change.

- [ ] GRPO training loop over `Episode` batches from TASK-02.
- [ ] **Role-conditioned advantage baseline, keyed on `mano_offset`.** SPIRAL
      needed this for first-move advantage in two-player games; Mus is worse —
      mano acts first *and* wins every tie, postre acts last. Without a
      per-position baseline the gradient encodes "mano is a good seat" instead
      of "that was a good bet".
- [ ] EMA update `b[offset] <- a*b[offset] + (1-a)*R`, advantage
      `A = R - b[offset]`.
- [ ] Frozen-checkpoint opponent pool, sampled per rollout, to avoid self-play
      collapse.
- [ ] Log the metrics SPIRAL used to catch its failure mode: mean response
      length, policy gradient norm, invalid-action rate. Their run collapsed
      from 3,500 chars to near-zero after 200 steps without RAE — watch for the
      same shape.
- [ ] Evaluate every K steps against the frozen baselines on the TASK-06
      mirrored suite; a checkpoint that does not beat `HeuristicPolicy` with a
      CI excluding 0.5 has not improved.
- [ ] Ablation to run and record: with and without the mano-conditioned
      baseline.

**Done when:** a trained checkpoint beats `HeuristicPolicy` on the mirrored
standard suite with a confidence interval excluding 0.5.

---

## Backlog


- **`feat/anthropic-provider`** — `StructuredLLMPolicy` takes a bare callable
  and nothing implements it. Needed to compare an SLM against a frontier model,
  not to train one.
- **`feat/senas`** — `CommunicationMode.DISABLED` is stubbed.
- **`feat/elo`** — TrueSkill or Elo across checkpoints, once there are enough
  checkpoints so it means something.
- **`perf/event-log`** — `state.events + (event,)` is O(n) per event, so
  quadratic over a long match. Irrelevant at hand-level episodes; revisit only
  if full-match rollouts become a bottleneck.
