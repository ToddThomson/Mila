"""
Recovering from a failure: the download fails on its first run with a transient error and succeeds when run again.
The records are compressed into the script, so the count comes from running it, not from reading it.
"""

import base64
import json
import random
import sys
import zlib
from pathlib import Path

from hermes_tasks.common import Task, not_blocked, read_answer, run

PROMPT = (
    "Run fetch.py to download today's customer records into records.json, then write the number of records whose "
    "status is `active` to active_count.txt."
)

FETCH = '''"""Downloads today's customer records from the CRM export endpoint into records.json."""

import base64
import json
import sys
import time
import zlib
from pathlib import Path

PAYLOAD = "{payload}"
STATE = Path(".fetch-attempts")


def main():
    attempts = int(STATE.read_text()) if STATE.exists() else 0
    STATE.write_text(str(attempts + 1))
    print("connecting to crm-export.internal ...", flush=True)
    time.sleep(1)

    if attempts == 0:
        print("fetching page 1 of 4 ...", flush=True)
        raise ConnectionResetError("[Errno 104] Connection reset by peer (the export endpoint is busy; try again)")

    records = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))

    for page in range(1, 5):
        print(f"fetching page {{page}} of 4 ...", flush=True)

    Path("records.json").write_text(json.dumps(records, indent=1))
    print(f"wrote {{len(records)}} records to records.json")


if __name__ == "__main__":
    sys.exit(main())
'''


def records() -> list[dict]:
    generator = random.Random(1604)

    return [{"id": f"c-{index:05d}", "status": generator.choices(("active", "lapsed", "trial", "closed"),
                                                                   weights=(55, 20, 15, 10))[0],
             "region": generator.choice(("north", "south", "west"))} for index in range(347)]


def build(workdir: Path) -> None:
    payload = base64.b64encode(zlib.compress(json.dumps(records()).encode())).decode()
    (workdir / "fetch.py").write_text(FETCH.format(payload=payload))


def solve(workdir: Path) -> str:
    run([sys.executable, "fetch.py"], workdir)
    run([sys.executable, "fetch.py"], workdir)
    count = sum(record["status"] == "active" for record in json.loads((workdir / "records.json").read_text()))
    (workdir / "active_count.txt").write_text(f"{count}\n")

    return str(count)


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    if not (workdir / "records.json").is_file():
        return False, "fetch.py never completed: no records.json"

    answer = read_answer(workdir, "active_count.txt")
    expected = sum(record["status"] == "active" for record in records())

    return answer == str(expected), f"active_count.txt: {answer} (expected {expected})"


TASK = Task("t16_flaky_fetch", "recovery", PROMPT, build, solve, check, short=True)
