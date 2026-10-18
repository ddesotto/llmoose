import json
import runpy
from pathlib import Path

Progress = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/watch_benchmark.py'))['JournalProgress']


def test_monitor_reads_incrementally_and_omits_private_input(tmp_path):
    path = tmp_path / 'records.jsonl'
    rows = [
        {'record_type': 'decision_opened', 'payload': {
            'seat': 3, 'hand_number': 0,
            'observation': {'phase': 'mus_vote', 'private_hand': ['SECRET_CARD']},
        }},
        {'record_type': 'action_applied', 'payload': {'action_index': 0, 'scores_after': [0, 1]}},
    ]
    path.write_text(json.dumps(rows[0]) + '\n')
    progress = Progress(path)
    progress.poll()
    assert 'mus_vote' in progress.text()
    assert 'SECRET' not in progress.text()
    partial = json.dumps(rows[1])
    with path.open('a') as f:
        f.write(partial[:20])
    progress.poll()
    assert progress.moves == 0
    with path.open('a') as f:
        f.write(partial[20:] + '\n')
    progress.poll()
    assert progress.moves == 1
    assert 'score 0–1' in progress.text()
    progress.poll()
    assert progress.moves == 1
