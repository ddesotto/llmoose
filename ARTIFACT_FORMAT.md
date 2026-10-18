# Run evidence format and compatibility

`runs/<name>/` contains evaluator-only evidence. A run may expose seeds, hidden
hands, exact model inputs, and provider responses. Do not use the directory as a
player observation or a public dataset.

## Raw records and common reader

The immutable identity of a v2 run is its `manifest.json` (`llmoose.run-manifest`,
version 2). `records.jsonl` uses `llmoose.run-event`, version 2. The journal has
monotonic `sequence` and `record_id`, stable run/match/decision/attempt IDs, and
typed records. `action_selected` is synced before the engine applies a choice;
`action_applied` commits the resulting state transition. `run_resumed` links a
new process to the previous record and the replayed state hash.

[`read_evidence()`](src/llmoose/simulation/evidence.py) is the stable, read-only
projection for v1 and v2 runs. It returns recorded actions with their original
rows attached. A v1 move has unknown action source and state hash; the adapter
leaves those fields `None`. It does not infer facts from today's prompt or engine.
Unknown optional v2 fields remain in the original records. Unknown schema
families or major versions fail explicitly.

The raw files are the source of truth. Derived validation, training, or index
files must name their source run IDs and hashes. A reader may add a new
projection, but must not silently rewrite old evidence or upgrade an unknown
field into an observed fact.

## Evolution rules

Within version 2, new optional fields may be added with explicit absence or a
documented default. Existing field meanings, IDs, reward definitions, visibility
rules, and action semantics must not change. A breaking change requires a new
major schema version and an explicit reader adapter. Preserve the writer's
version and the original bytes; never rely on a current dataclass layout alone.
This follows the writer/reader distinction in the
[Apache Avro specification](https://avro.apache.org/docs/1.12.0/specification/#schema-resolution),
but the on-disk format here remains JSON and JSONL.

The common reader preserves *structure*. Gameplay replay still requires the
recorded rules and compatible engine behavior. `source.zip` and its hash preserve
the exact source and lockfile for that purpose. Validation with changed local
code may report a mismatch; `read_evidence()` can still inspect the original
actions. Continuation requires the current source snapshot to match the run.

## Crash boundaries

Resume validates and replays the `action_applied` prefix. A saved
`action_selected` without an application is applied once. A saved legal model
answer without a selection becomes a selection and is applied without another
request. A started request without a saved answer has an unknown outcome; resume
records a conservative `recovery_fallback` and does not resend it. A choice or
response that existed only in process memory cannot be recovered after a hard
kill.

Only one process may resume a run at once. Torn journal and compatibility-log
tails are preserved as `.bin` backups before the complete prefix is extended.
Checksums include those backups. The v2 journal is authoritative when an older
`moves.jsonl` or `decisions.jsonl` file lags a crash boundary.

## Design references

- [RLDS](https://github.com/google-research/rlds/blob/main/README.md) distinguishes
  terminal steps from incomplete episodes caused by preemption.
- [OpenSpiel](https://github.com/google-deepmind/open_spiel/blob/master/docs/concepts.md)
  distinguishes full game state, player observations, chance, and action history.
- The [Poker Hand History paper](https://arxiv.org/abs/2312.11753) records game
  parameters and action history in a format designed for multiple poker variants.
- [Catanatron's data export](https://docs.catanatron.com/advanced/data-and-machine-learning)
  keeps game state samples, actions, and rewards as distinct data products.

These are design precedents, not a claim that any one format directly specifies
Mus recovery or model-request provenance.
