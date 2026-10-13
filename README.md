# llmoose

`llmoose` is a research environment for training and benchmarking SLLMs on the Spanish card game Mus.


# Code

The project is structured as follows:

```text
src/llmoose/
  rules/          cards, pure hand comparison, versioned rulesets
  game/           immutable state, reset/step environment, legal transitions
  observations/   one private, action-masked view per seat
  agents/         baselines, shared prompts, LLM policy, OpenRouter transport
  simulation/     seeded matches, JSON traces, deterministic replay
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

By default, invalid output gets up to two corrective retries, then an explicit
conservative fallback. Invalid responses and fallbacks are reported on stderr,
along with aggregate request and error counts. Connection or API failures stop
the match instead of silently substituting a move. `--llm-strict` disables
retries and fallback, stopping on the first invalid response. Use strict mode
for future on-policy rollouts; constrained decoding is not implemented yet.

The JSONL decision log contains the exact messages for each attempt, response
text, legal IDs, selected action, requested and returned model names, finish
reason, and provider-reported usage (including cost when supplied). Forced
actions, invalid attempts, fallbacks, and provider errors have distinct status
values. It is written incrementally, so completed records survive an interrupted
match. Each invocation replaces the selected log file. API keys are not logged.
Replay traces contain the actual moves and can replay without a model call;
a live model run with the same deal seed may choose different moves.

Use `--max-tokens` (default 1024), `--temperature` (default 0), and `--timeout`
(default 60 seconds) to tune requests. Models that spend the token allowance
on reasoning may return no answer; inspect `finish_reason` in the log and
increase `--max-tokens` if needed.

For a future local or hosted OpenAI-compatible inference server, supply its API
base URL and served model name:

```bash
llmoose-play --team-0 llm --team-1 heuristic --watch \
  --base-url http://localhost:8000/v1 --model your-served-model
```

Custom endpoints use no key unless `--api-key-env YOUR_KEY_VARIABLE` is supplied.
The prompt renderer and policy are independent of this HTTP transport, so
training and inference workers on Modal or Lambda can reuse them. Those cloud
integrations and the training loop remain future work.

For heuristic training examples, pass
`render=llmoose.agents.prompt.render_prompt` to `collect_episode`. This includes
the same role-conditioned system text and observation used during inference.

# References
- [RLCard](https://github.com/datamllab/rlcard),
- [DeepStack](https://pubmed.ncbi.nlm.nih.gov/28254783/)
- [ReBeL](https://proceedings.neurips.cc/paper/2020/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html)
- [SPIRAL](https://arxiv.org/abs/2506.24119) (self-play multi-turn RL on text games, Qwen3 base models),
- [TextArena](https://github.com/TextArena/TextArena) (Gym-style text-game interface)
- [Liquid's TRL guide](https://docs.liquid.ai/lfm/fine-tuning/trl)
