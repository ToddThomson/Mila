"""
Long context: a day of service logs and two lookup tables, where the answer is four hops from the error a person
reports -- error, request, user, the account the user belonged to that day, its plan -- with the same error code at
other times, a transposed code at the right time, and an account move that changes the answer if missed.
"""

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hermes_tasks.common import Task, not_blocked, read_answer

PROMPT = (
    "Support says a customer's order failed with error E-4471 on 17 October 2026 at about 14:20 UTC. Using the "
    "files here -- service.log, users.csv and accounts.json -- find which subscription plan that customer's account "
    "was on, and write the plan's name to answer.txt."
)

ANSWER = "Business Plus"

UTC = timezone.utc
SERVICES = ("gateway", "order-svc", "inventory", "payments", "notify", "search")
CODES = ("E-1002", "E-2210", "E-3307", "E-4410", "E-5120", "E-7781")
MESSAGES = {
    "E-1002": "upstream timeout", "E-2210": "inventory reservation conflict", "E-3307": "address validation failed",
    "E-4410": "card expired", "E-5120": "rate limit exceeded", "E-7781": "notification bounce",
}

# The planted requests: (time, request, user, code, message).
PLANTED = (
    (datetime(2026, 10, 17, 14, 18, 42, 311000, tzinfo=UTC), "r-8c41e2", "usr_58213", "E-4471",
     "payment authorization declined by processor"),
    (datetime(2026, 10, 17, 9, 2, 7, 18000, tzinfo=UTC), "r-1f07aa", "usr_10457", "E-4471",
     "payment authorization declined by processor"),
    (datetime(2026, 10, 17, 14, 20, 5, 640000, tzinfo=UTC), "r-77d0c1", "usr_61002", "E-4417",
     "duplicate order reference"),
    (datetime(2026, 10, 16, 14, 19, 51, 902000, tzinfo=UTC), "r-03be19", "usr_22980", "E-4471",
     "payment authorization declined by processor"),
)

USERS = [
    ("usr_58213", "acct_2210", "2025-03-02"),
    ("usr_58213", "acct_3391", "2026-09-30"),
    ("usr_10457", "acct_1188", "2024-11-19"),
    ("usr_61002", "acct_4002", "2025-07-08"),
    ("usr_22980", "acct_2210", "2026-01-15"),
]

ACCOUNTS = {
    "acct_2210": {"name": "Hollis & Grieve", "plan": "Team", "seats": 12},
    "acct_3391": {"name": "Pemberton Freight", "plan": "Business Plus", "seats": 40},
    "acct_1188": {"name": "Okoro Bakery", "plan": "Starter", "seats": 2},
    "acct_4002": {"name": "Northwind Ltd", "plan": "Enterprise", "seats": 250},
}


def stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def log_lines() -> list[str]:
    generator = random.Random(1017)
    users = [f"usr_{generator.randrange(10000, 99999)}" for _ in range(300)]
    start = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    entries = []

    for _ in range(1500):
        moment = start + timedelta(milliseconds=generator.randrange(36 * 3600 * 1000))
        request = f"r-{generator.randrange(16 ** 6):06x}"
        service = generator.choice(SERVICES)

        if generator.random() < 0.1:
            code = generator.choice(CODES)
            entries.append((moment, f"ERROR {service} req={request} code={code} msg=\"{MESSAGES[code]}\""))
        elif service == "gateway":
            entries.append((moment, f"INFO gateway req={request} auth ok user={generator.choice(users)} "
                                    f"ip=10.{generator.randrange(256)}.{generator.randrange(256)}.{generator.randrange(256)}"))
        else:
            entries.append((moment, f"INFO {service} req={request} handled in {generator.randrange(3, 900)}ms"))

    for moment, request, user, code, message in PLANTED:
        entries.append((moment - timedelta(milliseconds=409), f"INFO gateway req={request} auth ok user={user} "
                                                             f"ip=10.4.{generator.randrange(256)}.{generator.randrange(256)}"))
        entries.append((moment - timedelta(milliseconds=120), f"INFO order-svc req={request} creating order"))
        entries.append((moment, f"ERROR order-svc req={request} code={code} msg=\"{message}\""))

    entries.sort(key=lambda entry: entry[0])

    return [f"{stamp(moment)} {text}" for moment, text in entries]


def build(workdir: Path) -> None:
    (workdir / "service.log").write_text("\n".join(log_lines()) + "\n")
    (workdir / "users.csv").write_text(
        "# One row per user per account they have belonged to; `since` is when the user joined that account.\n"
        "user_id,account_id,since\n" + "".join(f"{user},{account},{since}\n" for user, account, since in USERS))
    (workdir / "accounts.json").write_text(json.dumps(ACCOUNTS, indent=2) + "\n")


def solve(workdir: Path) -> str:
    (workdir / "answer.txt").write_text(ANSWER + "\n")

    return f"Request r-8c41e2 at 14:18:42 was usr_58213, in acct_3391 since 30 September: {ANSWER}."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    answer = read_answer(workdir, "answer.txt")

    if answer is None:
        return False, "no answer.txt"

    return answer.strip(" .\"'").lower() == ANSWER.lower(), f"answer.txt: {answer[:40]}"


TASK = Task("t13_log_trace", "long-context", PROMPT, build, solve, check)
