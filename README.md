# llmoose

`llmoose` is a research environment for training and benchmarking SLLMs on the Spanish card game Mus.


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

Available policies are `conservative`, `heuristic`, `random`, and `llm`. `--policy`
sets all seats; `--team-0` and `--team-1` override individual partnerships.

Without spectator options, the command prints only the final result. `--trace`
saves a completed match for deterministic replay. Stopping early does not save
a partial trace.

You can also run directly from the checkout without installing:

```bash
PYTHONPATH=src python -m llmoose.cli --seed 12 --target-score 8 --policy heuristic --watch
```

## OpenRouter player

Set your key and a model ID from [OpenRouter's catalog](https://openrouter.ai/models).
Replace both placeholder values below. The client uses OpenRouter's
[OpenAI-compatible chat completions format](https://openrouter.ai/docs/quickstart)
and needs no additional Python dependencies.

```bash
export OPENROUTER_API_KEY='your-key'
export OPENROUTER_MODEL='provider/model-id'

llmoose-play --seed 12 --target-score 8 \
  --team-0 llm --team-1 heuristic --watch \
  --llm-log runs/decisions.jsonl --trace runs/match.json
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

Request settings are `--max-tokens` (1024), `--temperature` (0), and `--timeout`
(60 seconds). If reasoning uses up the token budget before an answer, increase
`--max-tokens`.

To use a local or hosted OpenAI-compatible server:

```bash
llmoose-play --team-0 llm --team-1 heuristic --watch \
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
