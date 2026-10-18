"""
Data work: distinct users who hit a server error in a two-hour window, across three services' logs -- one of which
writes local time with no offset -- with anonymous requests and errors just outside the window to tell apart.
"""

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hermes_tasks.common import Task, not_blocked, read_answer

PROMPT = (
    "The logs/ directory has yesterday's request logs from our three services; logs/README.md describes them. How "
    "many distinct users got at least one server error (a 5xx status) on 14 September 2026 from 09:00:00 UTC up to, "
    "but not including, 11:00:00 UTC? Write just the number to answer.txt."
)

README = """# Request logs, 14 September 2026

One JSON object per line: `ts`, `user`, `method`, `path`, `status`.

- `api.jsonl` and `web.jsonl` write `ts` in UTC, ISO 8601 with a `Z`.
- `billing.jsonl` comes from the old billing service, which writes the server's local time (Central European
  Summer Time, UTC+2) with no offset: `2026-09-14 11:30:05`.

`user` is null for requests made without logging in; those are not users.
"""

UTC = timezone.utc
WINDOW = (datetime(2026, 9, 14, 9, 0, tzinfo=UTC), datetime(2026, 9, 14, 11, 0, tzinfo=UTC))
PATHS = ("/orders", "/orders/{id}", "/cart", "/login", "/invoices", "/search")


def events() -> dict[str, list[dict]]:
    """Each service's requests as values: an aware UTC time, a user (or None) and a status."""
    generator = random.Random(914)
    users = [f"u-{1000 + index}" for index in range(140)]
    logs = {}

    for service in ("api", "web", "billing"):
        rows = []

        for _ in range(520):
            moment = datetime(2026, 9, 14, 6, 0, tzinfo=UTC) + timedelta(seconds=generator.randrange(8 * 3600))
            status = generator.choices((200, 201, 304, 404, 500, 502, 503), weights=(60, 8, 8, 6, 3, 2, 2))[0]
            user = None if generator.random() < 0.12 else generator.choice(users)
            rows.append({"moment": moment, "user": user, "status": status,
                         "method": generator.choice(("GET", "POST")),
                         "path": generator.choice(PATHS).replace("{id}", str(generator.randrange(10**5)))})

        # Errors on the window's edges, which a check of the boundaries has to get right.
        for moment in (WINDOW[0] - timedelta(seconds=1), WINDOW[0], WINDOW[1] - timedelta(seconds=1), WINDOW[1]):
            rows.append({"moment": moment, "user": generator.choice(users), "status": 500, "method": "GET",
                         "path": "/orders"})

        rows.sort(key=lambda row: row["moment"])
        logs[service] = rows

    return logs


def truth() -> int:
    return len({row["user"] for rows in events().values() for row in rows
                if row["user"] is not None and row["status"] >= 500 and WINDOW[0] <= row["moment"] < WINDOW[1]})


def build(workdir: Path) -> None:
    logs = workdir / "logs"
    logs.mkdir()
    local = timezone(timedelta(hours=2))

    for service, rows in events().items():
        lines = []

        for row in rows:
            if service == "billing":
                stamp = row["moment"].astimezone(local).strftime("%Y-%m-%d %H:%M:%S")
            else:
                stamp = row["moment"].strftime("%Y-%m-%dT%H:%M:%SZ")

            lines.append(json.dumps({"ts": stamp, "user": row["user"], "method": row["method"], "path": row["path"],
                                     "status": row["status"]}))

        (logs / f"{service}.jsonl").write_text("\n".join(lines) + "\n")

    (logs / "README.md").write_text(README)


def solve(workdir: Path) -> str:
    """Reads the logs as written, so the README is proven sufficient."""
    users = set()

    for path in (workdir / "logs").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)

            if path.stem == "billing":
                moment = datetime.strptime(row["ts"], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone(timedelta(hours=2))).astimezone(UTC)
            else:
                moment = datetime.strptime(row["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)

            if row["user"] and row["status"] >= 500 and WINDOW[0] <= moment < WINDOW[1]:
                users.add(row["user"])

    (workdir / "answer.txt").write_text(f"{len(users)}\n")

    return str(len(users))


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    answer = read_answer(workdir, "answer.txt")

    if answer is None:
        return False, "no answer.txt"

    return answer == str(truth()), f"answer.txt: {answer[:20]} (expected {truth()})"


TASK = Task("t09_error_users", "data", PROMPT, build, solve, check)
