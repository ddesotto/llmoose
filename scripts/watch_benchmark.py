"""Watch flushed benchmark journals without model calls or private-card output."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


class JournalProgress:
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.pending_bytes = b''
        self.moves = 0
        self.hand = 1
        self.phase = 'starting'
        self.seat = None
        self.scores = [0, 0]
        self.waiting_since = None
        self.finished = None

    def poll(self):
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset = stream.tell()
        lines = (self.pending_bytes + chunk).split(b'\n')
        self.pending_bytes = lines.pop()
        for line in lines:
            if not line:
                continue
            row = json.loads(line)
            kind, payload = row['record_type'], row['payload']
            if kind == 'decision_opened':
                observation = payload['observation']
                self.phase = observation['phase']
                self.seat = payload['seat']
                self.hand = payload['hand_number'] + 1
            elif kind == 'attempt_started':
                self.waiting_since = datetime.fromisoformat(row['recorded_at'])
            elif kind == 'attempt_finished':
                self.waiting_since = None
            elif kind == 'action_applied':
                self.moves = payload['action_index'] + 1
                self.scores = payload['scores_after']
            elif kind == 'run_finished':
                self.finished = payload.get('status', 'finished')
                self.waiting_since = None

    def text(self):
        seat = ('NORTH', 'EAST', 'SOUTH', 'WEST')[self.seat] if self.seat is not None else 'table'
        status = self.finished or 'running'
        if self.waiting_since is not None:
            seconds = max(0, int((datetime.now(timezone.utc) - self.waiting_since).total_seconds()))
            status = f'waiting for model ({seconds}s)'
        return (f'{self.path.parent.name}: {self.moves} moves | hand {self.hand} '
                f'| {self.phase} | {seat} | score {self.scores[0]}–{self.scores[1]} | {status}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--once', action='store_true', help='Show current progress and exit')
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error('Run directory does not exist')
    journals = {}
    previous = {}
    try:
        while True:
            for path in sorted(args.directory.rglob('records.jsonl')):
                progress = journals.setdefault(path, JournalProgress(path))
                progress.poll()
                line = progress.text()
                if previous.get(path) != line:
                    print(line, flush=True)
                    previous[path] = line
            manifest_path = args.directory / 'benchmark.json'
            planned = None
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                planned = 2 * len(manifest['seeds'])
            if args.once or (planned and len(journals) == planned
                             and all(p.finished for p in journals.values())):
                break
            if not journals and not previous:
                print('Waiting for the first recorded game...', flush=True)
                previous[None] = 'waiting'
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nMonitor stopped; the benchmark continues unchanged.', flush=True)


if __name__ == '__main__':
    main()
