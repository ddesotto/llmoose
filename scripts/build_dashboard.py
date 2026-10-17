"""Build a portable, offline HTML dashboard from the repository's learning artifacts.

Run `.venv/bin/python scripts/build_dashboard.py` after changing TODO or guides.
No dependencies or network access are required. Browser progress is independent
of source checkboxes and uses stable text-derived IDs across rebuilds.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GUIDES = {
    "overview": ("Learning path", "README.md"),
    "game": ("Game & prompt audit", "game-and-prompt.md"),
    "artifacts": ("Traces & artifacts", "artifacts.md"),
    "benchmark": ("Build MooseBench", "benchmark.md"),
    "training": ("Train a small model", "training.md"),
}
STAGES = [
    (
        "rules",
        "01",
        "Understand the game",
        "Correct rules and clear player inputs.",
        ["09", "10"],
    ),
    (
        "records",
        "02",
        "Preserve the evidence",
        "Capture decisions you can trust and reuse.",
        ["18"],
    ),
    (
        "benchmark",
        "03",
        "Compare models",
        "Paired games, honest metrics, durable runs.",
        ["11", "06", "12"],
    ),
    (
        "teach",
        "04",
        "Teach a small model",
        "Knowledge, demonstrations, and fine-tuning.",
        ["13", "14", "04", "05"],
    ),
    (
        "rl",
        "05",
        "Learn through play",
        "Verified rollouts, rewards, and updates.",
        ["15", "07", "08"],
    ),
    (
        "release",
        "06",
        "Expand & reproduce",
        "Cross-play, humans, and reusable releases.",
        ["16", "17"],
    ),
    (
        "foundation",
        "00",
        "Existing foundations",
        "Environment, trajectories, and dependencies.",
        ["01", "02", "03"],
    ),
]


def inline(value: str) -> str:
    """Render only the small inline Markdown subset used by the local guides."""
    tokens = []

    def keep(rendered):
        tokens.append(rendered)
        return f"\x00{len(tokens) - 1}\x00"

    value = re.sub(
        r"`([^`]+)`", lambda m: keep("<code>" + html.escape(m[1]) + "</code>"), value
    )

    def link(m):
        label, target = m[1], m[2]
        parsed = urlsplit(target)
        if parsed.scheme and parsed.scheme not in ("http", "https"):
            return label
        slug = next(
            (key for key, (_, file) in GUIDES.items() if target.split("#")[0] == file),
            None,
        )
        if target == "../TODO.md":
            attrs = 'href="#roadmap" data-view="roadmap"'
        elif slug:
            attrs = f'href="#learn/{slug}" data-guide="{slug}"'
        else:
            attrs = f'href="{html.escape(target, quote=True)}"'
            if parsed.scheme:
                attrs += ' target="_blank" rel="noopener noreferrer"'
        label_html = re.sub(
            r"\x00(\d+)\x00", lambda n: tokens[int(n[1])], html.escape(label)
        )
        return keep(f"<a {attrs}>{label_html}</a>")

    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, value)
    value = html.escape(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", value)
    return re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m[1])], value)


def markdown(source: str) -> str:
    """Small build-time renderer; source HTML is always escaped."""
    lines = source.splitlines()
    output, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            language, block = line[3:].strip(), []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            label = "Diagram source" if language == "mermaid" else language or "Example"
            output.append(
                f'<div class="code-block"><div class="code-label">{html.escape(label)}<button class="copy-code" type="button">Copy</button></div><pre><code>{html.escape(chr(10).join(block))}</code></pre></div>'
            )
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading:
            level = min(len(heading[1]) + 1, 6)
            output.append(f"<h{level}>{inline(heading[2])}</h{level}>")
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.fullmatch(r"[|\s:\-]+", lines[i]):
                    rows.append(
                        [
                            inline(c.strip())
                            for c in lines[i].strip().strip("|").split("|")
                        ]
                    )
                i += 1
            output.append(
                '<div class="table-wrap"><table><thead><tr>'
                + "".join(f"<th>{c}</th>" for c in rows[0])
                + "</tr></thead><tbody>"
                + "".join(
                    "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                    for row in rows[1:]
                )
                + "</tbody></table></div>"
            )
            continue
        bullet = re.match(r"^(?:[-*]|\d+\.)\s+(.+)", line)
        if bullet:
            ordered = bool(re.match(r"^\d+\.", line))
            tag, items = ("ol" if ordered else "ul"), []
            while i < len(lines):
                match = re.match(r"^(?:[-*]|\d+\.)\s+(.+)", lines[i])
                if not match:
                    break
                body = match[1]
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                    body += " " + lines[i].strip()
                    i += 1
                items.append("<li>" + inline(body) + "</li>")
            output.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
            continue
        paragraph = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^(#|```|\||[-*] |\d+\. )", lines[i])
        ):
            paragraph.append(lines[i])
            i += 1
        output.append("<p>" + inline(" ".join(paragraph)) + "</p>")
    return "\n".join(output)


def tasks_from_todo(source: str) -> list[dict]:
    tasks = []
    sections = re.split(r"^#{2,3} TASK-(\d+) — (.+)\n", source, flags=re.M)
    for i in range(1, len(sections), 3):
        number, title, body = sections[i : i + 3]
        body = re.split(r"^## ", body, maxsplit=1, flags=re.M)[0]
        checks = []
        for match in re.finditer(
            r"^- \[([ xX])\] (.+(?:\n[ \t]+\S.*)*)", body, flags=re.M
        ):
            text = " ".join(part.strip() for part in match[2].splitlines())
            digest = hashlib.sha256(text.encode()).hexdigest()[:16]
            checks.append(
                {
                    "id": f"{number}-{digest}",
                    "text": text,
                    "html": inline(text),
                    "done": match[1].lower() == "x",
                    "core": number == "18"
                    and match.start() < body.index("Downstream adapters"),
                }
            )
        stage = next(s[0] for s in STAGES if number in s[4])
        done = re.search(
            r"\*\*(?:Done when|Core done when):\*\*\s*(.+?)(?:\n\n|\Z)", body, re.S
        )
        depends = re.search(r"\*\*Depends on:\*\*\s*(.+)", body)
        guide = {
            "rules": "game",
            "records": "artifacts",
            "benchmark": "benchmark",
            "teach": "training",
            "rl": "training",
            "release": "training",
            "foundation": "training",
        }[stage]
        tasks.append(
            {
                "id": number,
                "title": title,
                "stage": stage,
                "checks": checks,
                "guide": guide,
                "description": " ".join(done[1].split())
                if done
                else "Foundation implemented; review the linked guide for context.",
                "depends": depends[1]
                if depends
                else "Start here; correctness is the first gate.",
            }
        )
    return tasks


def main():
    todo = (ROOT / "TODO.md").read_text()
    evidence = json.loads((DOCS / "evidence/review-2026-10-14.json").read_text())
    data = {
        "schema": 1,
        "sourceHash": hashlib.sha256(todo.encode()).hexdigest(),
        "tasks": tasks_from_todo(todo),
        "stages": [
            {
                "id": s[0],
                "number": s[1],
                "title": s[2],
                "description": s[3],
                "tasks": s[4],
            }
            for s in STAGES
        ],
        "guides": {
            slug: {
                "title": title,
                "path": filename,
                "html": markdown((DOCS / filename).read_text()),
                "words": len((DOCS / filename).read_text().split()),
            }
            for slug, (title, filename) in GUIDES.items()
        },
        "evidence": evidence,
        "trace": [
            json.loads(line)
            for line in (DOCS / "examples/trace-v2-example.jsonl")
            .read_text()
            .splitlines()
        ],
    }
    template = (DOCS / "dashboard/template.html").read_text()
    css = (DOCS / "dashboard/style.css").read_text()
    js = (DOCS / "dashboard/app.js").read_text()
    encoded = (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    result = (
        template.replace("/* INLINE_STYLE */", css)
        .replace("/* INLINE_APP */", js)
        .replace("JSON_PAYLOAD", encoded)
    )
    (DOCS / "index.html").write_text(result)
    print(
        f"Built docs/index.html: {len(data['tasks'])} tasks, {sum(len(t['checks']) for t in data['tasks'])} checklist items, {len(data['guides'])} guides."
    )


if __name__ == "__main__":
    main()
