"""Summarize saved policy failures without requests, replay, or modifying runs.

Usage: python scripts/audit_policy_inputs.py [runs_directory]
Output includes source hashes and decision indices, not private cards or prompts.
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def audit(root: Path) -> list[dict]:
    summaries = []
    for path in sorted(root.rglob('decisions.jsonl')):
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.splitlines() if line]
        summaries.append({
            'path': str(path),
            'sha256': hashlib.sha256(data).hexdigest(),
            'statuses': dict(Counter(row['status'] for row in rows)),
            'prompt_versions': sorted({row.get('prompt_version', 0) for row in rows}),
            'failures': [
                {key: row.get(key) for key in (
                    'action_index', 'seat', 'status', 'model', 'error', 'finish_reason'
                )}
                for row in rows if row['status'] in ('invalid', 'provider_error')
            ],
        })
    for path in sorted(root.rglob('records.jsonl')):
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.splitlines() if line]
        summaries.append({
            'path': str(path),
            'sha256': hashlib.sha256(data).hexdigest(),
            'record_types': dict(Counter(row['record_type'] for row in rows)),
            'action_sources': dict(Counter(
                row['payload']['source'] for row in rows
                if row['record_type'] == 'action_applied'
            )),
        })
    return summaries


if __name__ == '__main__':
    print(json.dumps(audit(Path(sys.argv[1] if len(sys.argv) > 1 else 'runs')), indent=2))
