# Mus development suite v1

This is a rules-assisted diagnostic benchmark and a frozen development environment,
separate from the gameplay leaderboard. It does not grade strategic moves or claim
that a model is a strong Mus player. Gameplay continues to use prompt v5.

Run the offline checks first:

```bash
uv run --frozen llmoose-diagnose --check
uv run --frozen llmoose-diagnose --manifest
```

The 46 authored questions cover hand comparisons, scoring, action consequences,
information visibility, and episode limits. An evaluator independently executes
engine checks against the fixed expected labels; it never regenerates labels from
the implementation. This catches regressions against these cases, not every
possible rules error. The rules are this repository's frozen Mus variant.

To evaluate an LLM (this makes provider requests):

```bash
uv run --frozen llmoose-diagnose --model YOUR_MODEL_ID \
  --output runs/diagnostics-first-model --max-tokens 2048
```

OpenRouter uses `OPENROUTER_API_KEY` from the environment or `.env`. Local servers
can use `--base-url http://localhost:8000/v1`; custom endpoints use no API key unless
`--api-key-env VARIABLE` is supplied. `--reasoning` requests a separate native
reasoning channel; recorded reasoning is kept apart from the final JSON answer.
This diagnostic track requires strict `{"choice_id": integer}` output and schema
support. Gameplay's compatible prose adapter is not used in this track.

Each new output directory contains the manifest, source archive (including frozen
fixtures), individual records, category scores, and checksums. Records include
exact inputs, request bodies, completions and available reasoning. Provider errors
stop further requests and leave the remaining cases explicitly unanswered. Missing,
malformed and unanswered cases stay in the accuracy denominator. Interruptions
produce a partial report. These diagnostic artifacts are not gameplay journals and
are not inputs to `llmoose-validate`.

## Non-LLM adapters

Export questions without answer keys:

```bash
uv run --frozen llmoose-diagnose --export mus-development-inputs.json
```

The export includes the rulebook, suite hash, choice-order seed, and structured
questions. A decision model can encode facts and choices directly, then return one
choice ID per question. The Python `evaluate(callback)` interface passes only the
same public question packet to a callback, which returns an integer ID or `None`
for an unanswered case. This is an adapter interface, not a trained classifier.

For external predictions, write JSON with this shape (copy the actual hash from
the export; omitted answers count as unanswered):

```json
{
  "suite_sha256": "COPY_FROM_EXPORT",
  "order_seed": 0,
  "answers": {"probe-001": 0, "probe-002": 1}
}
```

```bash
uv run --frozen llmoose-diagnose --predictions predictions.json \
  --output runs/diagnostics-external-model
```

Choice IDs are local to the exported question and permutation. Use `--order-seed`
on both export and grading to test another deterministic choice ordering. The
public packets exclude labels, explanations and categories. Full hands in some
questions are explicitly hypothetical supplied facts, not hidden game state.
The answer keys are public in this development repository: do not present these
questions as a secret or uncontaminated held-out test.

## Environment and gameplay allocation

`environment_v1.json` freezes the rules, deck, action catalog, observation fields,
prompt hash and lifecycle contract. Drift fails the check rather than silently
changing the benchmark. A future deliberate contract change needs a reviewed new
version. The manifest pins diagnostic questions, rulebook and baseline opponent source
hashes too.

Player inputs expose current own cards, current-hand own discard memory and public
match events, including previous showdowns. Opponent private cards, private
discards, stock and deal seed remain hidden. Own discard memory resets each hand;
this is not a perfect-recall environment. Keep seat policies separate and create
fresh policies each match. Evaluator state and `step().info` are not policy inputs.

`MusEnv(max_steps=N)` now reports `truncated=True` separately from natural
`terminated=True`. The limit does not invent a winner or reward. Natural endings
win at an exact limit; either ending requires reset. Truncated observations remain
available for bootstrapping. The hand trajectory collector rejects truncated hands
instead of assigning completed-hand training targets.

The frozen allocation reserves train seeds 0–199, development seeds 1000–1019,
and test seeds 2000–2049. All hands, seats, mirrors, dealer rotations and
continuations derived from one seed belong to its group. The test seeds are public
reserved evaluation data, not secret deals. `split_for_seed()` checks allocation;
existing general-purpose training/gameplay commands do not automatically enforce
these partitions, so adapters must call it and retain seed provenance.

The fixed gameplay opponents are conservative and heuristic, target 40, initial
dealer NORTH, both partnership assignments, strict unassisted play, and a 500-step
limit. For example, run the complete development pool against one opponent:

```bash
uv run --frozen llmoose-bench --suite-id mus-dev-v1-conservative \
  --seeds 1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019 \
  --candidate llm --model YOUR_MODEL_ID --opponent conservative \
  --target-score 40 --max-steps 500 --prompt-version 5 \
  --output runs/mus-dev-v1-conservative
```

Repeat with opponent `heuristic` and a new suite ID/output directory: 80 scheduled
games across both opponents and both assignments. Source snapshots identify the
exact opponent implementations. Report incomplete coverage alongside wins;
complete-pair win rates are conditional on completion. These gameplay results and
the rules-assisted diagnostic scores measure different things and must remain
separate. No live inference or model training was needed to validate this release.
