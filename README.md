# llmoose

`llmoose` is a research environment for training and benchmarking SLLMs on the Spanish card game Mus.

Start the [learning guide](docs/README.md) for the path from playing a game to
building MooseBench, preserving traces, distilling an SLM, and training through
play. The [roadmap](TODO.md) separates implemented foundations from planned work.

Open [the interactive HTML workspace](docs/index.html) in a browser for task
checklists, personal notes, the full learning library, evidence, and a trace
explorer. It works offline. Progress is browser-local and can be exported/imported;
it does not edit `TODO.md`. Rebuild after source changes with
`.venv/bin/python scripts/build_dashboard.py`.

The [game and prompt audit](docs/game-and-prompt.md) records the historical ranking
and scoring findings. The ranking fix and the federation-based scoring contract
below are implemented; the broader rules audit and player-interface work remain
tracked in [TODO.md](TODO.md).

## Rules contract

New games default to **`mus_8r8a_federation_no_senas_v2`**. This is a benchmark
adaptation of the Federación de Mus rules dated **16-02-2025, V.7**, particularly
articles 44, 61–62, 69–71: accepted bets settle at hand-end, refusal points are
paid during bidding, cards are revealed at showdown, and an accepted órdago
overrides earlier accepted bets. See the [reference regulation](https://lafederaciondemus.es/wp-content/uploads/2026/02/Reglamento-de-Juego-Federacion.pdf).
The [Pagat explanation](https://www.pagat.com/vying/mus.html) supplies comparison
examples, intrinsic hand values, and a minimum raise increment of two.

Four seats form two partnerships. Each receives four cards from the 40-card
Spanish deck, with twos equivalent to aces and threes equivalent to kings.
Suits do not affect comparisons. Grande compares descending ranks, chica ascending
ranks. Pares orders duples (including four equal) above medias above a pair;
singletons do not break ties. Juego requires at least 31, ordered
31, 32, 40, 37, 36, 35, 34, 33. If nobody has juego, highest total wins punto.
The best eligible individual hand wins for its team; ties follow priority from mano.
No juego real exception is used. Pares and juego declarations are truthful.
Only eligible seats participate; a sole eligible team wins without betting.

The v2 settlement contract is:

| Outcome | Paid immediately | Paid at showdown |
| --- | --- | --- |
| All pass, or an uncontested lance | Nothing | Natural value |
| Numeric bet called | Nothing | Stake, plus intrinsic value for pares/juego/punto |
| Opening bet or opening órdago refused | 1 to the bidding team | That team's intrinsic value for pares/juego/punto |
| Raise or raised órdago refused | Previous numeric stake to the bidding team | That team's intrinsic value for pares/juego/punto |
| Órdago called | Match won by the best eligible hand's team | All pending awards cancelled |

Natural values are 1 for grande/chica/punto; for pares, sum the winning team's
eligible hands (pair 1, medias 2, duples 3); for juego, sum its qualifying hands
(31 scores 3, other qualifying totals 2). Punto's intrinsic value is 1, including
after a call or fold. Called grande/chica pay only the stake; folded grande/chica
have no further award. There is no additional *deje* on a refused raise.

At ordinary showdown all four hands become public, then settlement proceeds in
grande → chica → pares → juego/punto order. Stop at the first team reaching the
target; later pending awards are not paid. Immediate refusal points can end a
match before showdown, even if an earlier accepted bet would favor the other team.
A fold concedes the intrinsic values as well, regardless of who held better cards.
Already-paid refusal points are never paid again. A called órdago raises the
winner's displayed score to the target, preserving the loser's paid score; this
is a terminal encoding, not a target-sized stone bonus.

For example, bet 2 → raise to 6 → fold pays 2 immediately to the raiser's team.
At pares, if that team holds medias plus a pair, it receives 3 more at showdown
unless the match ends first. Conversely, calling a numeric bet of 40 at grande
does not end bidding on the hand: a later accepted órdago at chica can overturn it.

Explicit benchmark adaptations retained in v2:

- One game to a configurable target (default 40), rather than a multi-game match.
- No señas, dialogue, simultaneous speech, revocation of a partner's answer,
  physical dealing faults, tournament penalties, or special invalid-órdago claims.
  The next eligible opposing seat gives one binding team response.
- The finite numeric menu runs from 2 to the target, with raises increasing the
  total by at least 2. Reference play permits larger bets; a menu-limit numeric
  bet remains distinct from órdago and can itself be answered with órdago.
- The existing deterministic seat/deal schedule is retained: NORTH → EAST →
  SOUTH → WEST, priority starting at `(dealer - 1) mod 4`. This is a simulation
  convention rather than a reproduction of physical dealer/postre seating.
- Mus votes are sequential and unanimous; each seat discards 1–4 cards, starting
  at mano. Recycling retains the existing algorithm: if stock cannot fill a draw,
  shuffle all accumulated discards, including that player's new discards, into
  the available stock. Recycled cards are drawn before the residual stock.

Replay selects behavior by the saved ruleset ID. **`mus_8r8a_no_senas_v1`** retains
immediate scoring, one-stone raise increments, no public reveals, and the old
numeric-target/órdago equivalence. Use `--ruleset mus_8r8a_no_senas_v1` for that
legacy environment. Unknown IDs are rejected; trace schema 1 and old files remain
readable. Both rulesets use the corrected hand evaluator: pre-fix traces affected
by the medias bug require their original engine commit, and replay rejects an
outcome mismatch rather than rewriting the artifact.
New training episodes use schema 2 and record their ruleset ID. Schema 1 episodes
remain readable with unknown ruleset provenance (`null`), without guessing from
their rewards or assigning the new default retroactively.

For v2 logs, `lance_pending` records public settlement terms without revealing a
winner. `score_awarded` events are the score-delta ledger. Each folded lance has
one result, first recording refusal points and later accumulating intrinsic points;
`fold_intrinsic` awards reference its result index. Count that lance once in reports.
`cards_revealed` events contain the completed hand's physical cards. Historical
v1 logs continue to use `lance_resolved` events for score deltas.


# Code

Start with three responsibilities:

- **The game** holds the cards and enforces the rules. It knows nothing about
  HTTP requests or language models.
- **A player policy** chooses one move from what that seat can see. A heuristic
  uses Python rules; an LLM policy asks a model. Both expose `act(observation)`.
- **The match loop** connects them, asking the next player for a move and giving
  that move to the game engine.

The important part of each turn is just:

```python
observation = observe(state, actor, codec)
action = policies[actor].act(observation)
state = apply_action(state, action)
```

`codec` translates between numbered menu choices and structured actions such as
`Action(BET, amount=2)`. `observe` removes hidden information; `apply_action`
checks legality and produces the next state.

Read the code in this order:

| File | Question it answers |
| --- | --- |
| [cli.py](src/llmoose/cli.py) → `play_command` | How are options, players, a match, and output wired together? |
| [agents/factory.py](src/llmoose/agents/factory.py) → `build_policies` | Who controls each of the four seats? |
| [simulation/match.py](src/llmoose/simulation/match.py) → `run_match` | What happens on every turn? |
| [observations/core.py](src/llmoose/observations/core.py) → `observe` | What is this player allowed to know? |
| [game/engine.py](src/llmoose/game/engine.py) → `apply_action` | What does the chosen move do? |

Once that loop makes sense, follow the LLM path in
[agents/llm.py](src/llmoose/agents/llm.py): render a prompt, call the provider,
parse the answer, return an action. Logging lives in
[simulation/reporting.py](src/llmoose/simulation/reporting.py) and the spectator
display in [simulation/terminal.py](src/llmoose/simulation/terminal.py).

A human seat would fit at exactly the same boundary: `HumanPolicy.act()` would
display the private observation, wait for your selection, and return an action.
That input policy is not implemented yet. The engine and match loop would not
need to change.

For now, leave `training/` and `benchmark/` until after the turn loop is clear.
`training/` collects hand episodes and returns; it does not yet update model
weights. `game/environment.py` wraps the same engine in `reset()` and `step()`
for that collection. The CLI match runner uses the engine directly.

The project is structured as follows:

```text
src/llmoose/
  rules/          cards, pure hand comparison, versioned rulesets
  game/           immutable state, reset/step environment, legal transitions
  observations/   one private, action-masked view per seat
  agents/         player construction, baselines, prompts, LLM policy and transport
  simulation/     match loop, terminal display, reporting, traces and replay
  benchmark/      fixed-deal suites and aggregate match reports
  training/       per-seat hand trajectories and JSONL datasets
```


# Play

Use `uv` and the committed lockfile (Python 3.11+):

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev pytest -q
```

`uv` creates and uses the project’s `.venv`; activation is optional.

Watch the heuristic partnership play against the conservative partnership:

```bash
uv run --frozen llmoose-play --seed 12 --target-score 8 --team-0 heuristic --team-1 conservative --watch
```

The spectator view shows all four hands, mano, each decision, and scoring as it
happens. Team 0 is North/South; team 1 is East/West. Agents still receive only
their own private cards and public information.

Use `--step` to advance one action at a time with Enter (`q` quits), or
`--delay 0.1` to change the automatic pace. `--delay 0` prints the whole game
immediately. Both options enable the spectator view.

```bash
uv run --frozen llmoose-play --seed 12 --target-score 8 --policy heuristic --step
uv run --frozen llmoose-play --seed 12 --target-score 8 --policy random --watch --delay 0.1 --trace match.json
```

Available policies are `conservative`, `heuristic`, `random`, and `llm`. `--policy`
sets all seats; `--team-0` and `--team-1` override individual partnerships.

Without spectator options, the command prints only the final result. `--trace`
saves a completed match for deterministic replay. Stopping early does not save
a partial trace.

## OpenRouter player

Put your key and a model ID from [OpenRouter's catalog](https://openrouter.ai/models)
in `.env` in the directory where you run the command:

```dotenv
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=provider/model-id
```

The CLI loads `.env` automatically. Existing shell variables take precedence;
`--model` takes precedence over both. `.env` is gitignored. Exporting the variables
in your shell also works. The client uses OpenRouter's
[OpenAI-compatible chat completions format](https://openrouter.ai/docs/quickstart).

```bash
uv run --frozen llmoose-play --seed 12 --target-score 8 \
  --team-0 llm --team-1 heuristic --watch \
  --run-dir runs/llm-seed-12
```

`--model` overrides `OPENROUTER_MODEL`. Both seats in an LLM partnership use
that model, with separate private observations. `--policy llm` plays all four
seats with the same model. Use `--step` to inspect each move; quitting happens
before the next model request.

The model receives its cards, partnership, mano position, scores, public
history, bidding context, and the legal action IDs with descriptions. It must
answer exactly `<answer>ID</answer>`. A forced action, such as a truthful
declaration, is performed without an API request.

Invalid responses get up to two retries, then a logged conservative fallback.
Use `--llm-strict` to stop on the first invalid response. Connection or API
failures always stop the match.

`--llm-log` saves prompts, responses, chosen actions, and usage as JSONL.
Records are saved as the match runs; each invocation overwrites the file.
`--trace` saves moves for replay without model calls. See
[the policy](src/llmoose/agents/llm.py) and
[reporting](src/llmoose/simulation/reporting.py) for log details.

## Trace and score a run

`--run-dir runs/llm-seed-12` creates a **new** directory with:

| File | Contents |
| --- | --- |
| `decisions.jsonl` | Exact LLM prompts/responses, prompt version, request timing, usage, retries, and fallbacks; only created for LLM games |
| `moves.jsonl` | Every applied move, public events, and before/after scores, flushed during play |
| `match.json` | Deterministic replay of a completed match |
| `report.json` | Run settings, completion status, automatic game scores, and LLM metrics |

Existing run directories are rejected to protect earlier experiments. Use a new
name for each run. `--run-dir` supplies the log and trace paths; do not combine it
with `--llm-log` or `--trace`.

The report scores each partnership with **1 for a win, 0 for a loss**, plus its
stones, stone differential, and per-lance wins and points. Lance results include
folds and uncontested awards, with reasons broken out. These describe the game's
outcome; they are not strategic decision grades or a benchmark win rate.

LLM metrics are reported overall, by team, and by seat. First-try legal rate and
fallback rate use decisions requiring inference as the denominator; forced
moves are excluded. Invalid-response rate counts returned responses, excluding
connection failures. Token and provider-reported cost totals include retries;
each total includes the number of requests that reported it. Missing usage is
`null`, not zero. Cost is recorded as supplied by the provider.

`run_id` identifies the experiment. A zero-based `action_index` links all attempts
for a decision to `moves.jsonl` and the action list in `match.json`.
`decision_index` is local to each seat. The exact prompts remain in the decision
log, alongside the requested and returned model IDs.

If interrupted, limited by `--max-steps`, or stopped by a provider/format error,
the run retains its logs and a partial report. An incomplete game has no winner
or win points. `match.json` is written only after the game completes. A hard
process kill may leave only the already-flushed JSONL logs.

For an offline smoke run:

```bash
uv run --frozen llmoose-play --seed 12 --target-score 8 \
  --team-0 heuristic --team-1 conservative --run-dir runs/baseline-seed-12
```

## Inspect and replay games

```bash
# Results, reliability, tokens (including reasoning), and reported cost.
uv run --frozen llmoose-inspect runs/llm-first

# Find an interesting zero-based action index, then read its exact conversation.
uv run --frozen llmoose-inspect runs/llm-first --decisions
uv run --frozen llmoose-inspect runs/llm-first --action 14

# Watch the saved game again, with all four hands visible and no API calls.
uv run --frozen llmoose-inspect runs/llm-first --replay --step
```

Inspection also works during a run: it reads the flushed JSONL records and shows
the latest saved score until the final report exists. Action inspection includes
all retries and any provider-returned reasoning text or summary. Old logs without
reasoning text cannot reconstruct it from token counts.

To request reasoning on a longer game:

```bash
uv run --frozen llmoose-play --seed 12 --target-score 40 \
  --team-0 llm --team-1 heuristic --reasoning --max-tokens 2048 \
  --run-dir runs/llm-long
```

`--reasoning` sends OpenRouter's `reasoning: {enabled: true, exclude: false}`.
The client preserves returned `reasoning` and `reasoning_details` separately from
the final action. Availability depends on the model/provider; a reasoning-token
count does not guarantee readable text. Reasoning tokens are part of the output
budget, so increase `--max-tokens` if answers are truncated. See
[OpenRouter's reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

For background play from your terminal, use a fresh run name:

```bash
mkdir -p runs
nohup uv run --frozen llmoose-play --seed 12 --target-score 40 \
  --team-0 llm --team-1 heuristic --reasoning --max-tokens 2048 \
  --run-dir runs/llm-background > runs/llm-background.console.log 2>&1 < /dev/null &

uv run --frozen llmoose-inspect runs/llm-background
```

The target is a score threshold, not a fixed number of hands; an órdago can end
even a 40-point match early. Use multiple seeded games for performance comparisons.

## Shared system prompts

Game instructions live in
[`src/llmoose/agents/system_prompts/`](src/llmoose/agents/system_prompts/).
`v1.txt` through `v3.txt` preserve the earlier instructions; **`v4.txt` is the
default** and explains the player-facing context, private recall, and history policy.
Every model uses the same selected instructions, followed by its seat/team role.
Private cards, public history, and legal choices stay in the user message.

Use `--prompt-version 1`, `2`, `3`, or `4` to select a version. Logs and reports
record the prompt and renderer versions. Prompts 1–3 retain renderer 2; prompt 4
uses renderer 3. Training renderers accept the same `prompt_version`.
Keep published versions unchanged: add a new `vN.txt` and register it in
`agents/prompt.py` when changing instructions. All versions ship with the package.

Renderer 3 supplies the active lance, eligible seats, tie order, bidding team,
refusal recipient, and contextual action effects. It remembers only the acting
seat's discards during the current hand; those cards may return through recycling.
The current hand's public events are complete. Up to three earlier public hands
are summarized by bids, folds, and awards, with the number omitted stated in the
input. Past-hand card reveals and detailed actions are omitted from that summary;
scores retain all paid awards. No hand-strength summary is supplied, so evaluating
the private cards remains the player's task. The current hand, private cards,
rules, and legal menu are never truncated by the renderer; very long hands may
exceed a provider's context window.

Request settings are `--max-tokens` (1024), `--temperature` (0), and `--timeout`
(60 seconds). If reasoning uses up the token budget before an answer, increase
`--max-tokens`.

To use a local or hosted OpenAI-compatible server:

```bash
uv run --frozen llmoose-play --team-0 llm --team-1 heuristic --watch \
  --base-url http://localhost:8000/v1 --model your-served-model
```

Custom endpoints use no key unless `--api-key-env YOUR_KEY_VARIABLE` is supplied.
Training and cloud integration plans are in [TODO.md](TODO.md).

# References
- [RLCard](https://github.com/datamllab/rlcard),
- [DeepStack](https://pubmed.ncbi.nlm.nih.gov/28254783/)
- [ReBeL](https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html)
- [SPIRAL](https://arxiv.org/abs/2506.24119) (self-play multi-turn RL on text games, Qwen3 base models),
- [TextArena](https://github.com/TextArena/TextArena) (Gym-style text-game interface)
- [Liquid's TRL guide](https://docs.liquid.ai/lfm/fine-tuning/trl)
