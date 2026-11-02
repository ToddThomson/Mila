"""
Long context: a value traced through a generated codebase of about 45K tokens -- endpoint, job, defaults, profile
override and a clamp in the storage client -- among dozens of other timeouts, with an architecture document that
still gives the value from before the profile was added.
"""

import random
from pathlib import Path

from hermes_tasks.common import Task, not_blocked, read_answer

PROMPT = (
    "This is the source of the fleet service. When a request reaches the /export endpoint without a `timeout` "
    "parameter, what timeout, in seconds, does the storage call actually use? Assume no environment variables are "
    "set. Write just the number to answer.txt."
)

ANSWER = "75"

CHAIN = {
    "fleet/api/routes/export.py": '''"""POST /export: streams a dataset to the caller."""

from fleet.jobs import exporter


def handle_export(request):
    dataset = request.params["dataset"]
    timeout = request.params.get("timeout")

    if timeout is not None:
        timeout = float(timeout)

    return exporter.run(dataset, timeout=timeout)
''',
    "fleet/api/routes/imports.py": '''"""POST /import: loads a dataset from the caller into storage."""

from fleet.config.profiles import load_profile
from fleet.storage import client


def handle_import(request):
    settings = load_profile("import")
    timeout = float(request.params.get("timeout", settings["storage_timeout"]))

    return client.store(request.params["dataset"], request.body, timeout=timeout)
''',
    "fleet/jobs/exporter.py": '''"""Export jobs: read a dataset out of storage in one call."""

from fleet.config.profiles import load_profile
from fleet.storage import client


def run(dataset, timeout=None):
    settings = load_profile("export")
    effective = settings["storage_timeout"] if timeout is None else timeout

    return client.fetch(dataset, timeout=effective, retries=settings["storage_retries"])
''',
    "fleet/jobs/reports.py": '''"""Nightly reports, read from storage."""

from fleet.config.profiles import load_profile
from fleet.storage import client


def nightly(names):
    settings = load_profile("reports")

    return [client.fetch(f"reports/{name}", timeout=settings["storage_timeout"]) for name in names]
''',
    "fleet/config/defaults.py": '''"""Settings every profile starts from."""

DEFAULTS = {
    "storage_timeout": 30,
    "storage_retries": 2,
    "http_timeout": 10,
    "queue_poll_interval": 5,
    "telemetry_flush_seconds": 60,
}
''',
    "fleet/config/profiles.py": '''"""Per-workload settings: the defaults, with a profile's own values over them."""

import os

from fleet.config.defaults import DEFAULTS

PROFILES = {
    "export": {"storage_timeout": 90, "storage_retries": 1},
    "import": {"storage_timeout": 60},
    "reports": {"storage_timeout": 20},
}


def load_profile(name):
    settings = dict(DEFAULTS)
    settings.update(PROFILES.get(name, {}))
    override = os.environ.get(f"FLEET_{name.upper()}_STORAGE_TIMEOUT")

    if override:
        settings["storage_timeout"] = float(override)

    return settings
''',
    "fleet/storage/__init__.py": "",
    "fleet/storage/client.py": '''"""The object-storage client every workload goes through."""

import os
import time

# The storage gateway drops any call held open longer than this.
MAX_TIMEOUT = float(os.environ.get("FLEET_STORAGE_MAX_TIMEOUT", "75"))


def _call(operation, key, timeout):
    started = time.monotonic()
    # The real transport lives in the gateway SDK; this module only shapes the call.
    return {"operation": operation, "key": key, "timeout": timeout, "started": started}


def fetch(key, timeout, retries=0):
    timeout = min(timeout, MAX_TIMEOUT)
    attempt = 0

    while True:
        try:
            return _call("get", key, timeout)
        except TimeoutError:
            attempt += 1

            if attempt > retries:
                raise


def store(key, body, timeout):
    return _call("put", key, min(timeout, MAX_TIMEOUT))
''',
    "fleet/storage/backup.py": '''"""Copies datasets to the cold-storage bucket."""

from fleet.storage import client

BACKUP_TIMEOUT = 45


def copy_to_cold(key):
    return client.fetch(key, timeout=BACKUP_TIMEOUT, retries=3)
''',
    "docs/ARCHITECTURE.md": '''# Architecture

Requests arrive through `fleet/api/routes`. Long-running work is handed to `fleet/jobs`, and everything that
touches object storage goes through `fleet/storage/client.py`.

Storage calls use the 30-second timeout from `fleet/config/defaults.py`; a caller may pass its own.

Settings live in `fleet/config/defaults.py`.
''',
}

NOUNS = ("vehicle", "route", "driver", "depot", "parcel", "manifest", "shift", "invoice", "customer", "zone",
         "tariff", "fuel", "maintenance", "incident", "schedule", "telemetry", "warehouse", "pallet", "dock",
         "carrier", "consignment", "booking", "rate", "audit", "alert")
VERBS = ("validate", "normalize", "merge", "summarize", "rank", "filter", "group", "estimate", "reconcile", "format")
FIELDS = ("id", "region", "status", "owner", "eta", "weight", "priority", "created_at", "updated_at", "code")
AREAS = ("services", "models", "util", "integrations", "telemetry", "billing")


def generated_function(generator: random.Random, noun: str) -> str:
    verb = generator.choice(VERBS)
    field = generator.choice(FIELDS)
    key = generator.choice(FIELDS)
    limit = generator.choice((50, 100, 250, 500, 1000))
    action = generator.choice(("skipped", "rejected"))
    suffix = generator.choice(("records", "batch", "rows", "items", "entries"))

    return f'''

def {verb}_{noun}_{suffix}(records, *, limit={limit}, strict={generator.choice(("True", "False"))}):
    """
    {verb.capitalize()} {noun} {suffix} for the {generator.choice(AREAS)} pipeline. Records missing `{field}` are
    {action} when strict; at most `limit` come back, ordered by `{key}`.
    """
    result = []

    for record in records:
        value = record.get("{field}")

        if value is None:
            if strict:
                raise ValueError("{noun} record without {field}")

            continue

        if isinstance(value, str):
            value = value.strip()

        result.append({{**record, "{field}": value}})

        if len(result) >= limit:
            break

    return sorted(result, key=lambda item: str(item.get("{key}", "")))
'''


def generated_class(generator: random.Random, noun: str) -> str:
    name = noun.capitalize() + generator.choice(("Gateway", "Client", "Poller", "Cache", "Publisher"))
    timeout = generator.choice((5, 8, 12, 15, 20, 25, 40, 45, 60, 90, 120))
    retries = generator.choice((0, 1, 2, 3, 5))

    return f'''

class {name}:
    """Talks to the {noun} service. Calls give up after `timeout` seconds and are retried `retries` times."""

    def __init__(self, endpoint, *, timeout={timeout}, retries={retries}):
        self.endpoint = endpoint
        self.timeout = timeout
        self.retries = retries
        self._session = None

    def _request(self, method, path, payload=None):
        attempts = 0

        while True:
            try:
                return self._send(method, f"{{self.endpoint}}/{{path}}", payload, timeout=self.timeout)
            except ConnectionError:
                attempts += 1

                if attempts > self.retries:
                    raise

    def _send(self, method, url, payload, timeout):
        raise NotImplementedError("bound to the transport at start-up")

    def get(self, identifier):
        return self._request("GET", f"{noun}s/{{identifier}}")

    def list(self, **filters):
        query = "&".join(f"{{key}}={{value}}" for key, value in sorted(filters.items()))

        return self._request("GET", f"{noun}s?{{query}}")

    def update(self, identifier, changes):
        if not changes:
            raise ValueError("nothing to update")

        return self._request("PATCH", f"{noun}s/{{identifier}}", changes)
'''


def generated_files() -> dict[str, str]:
    generator = random.Random(1218)
    files = {"fleet/__init__.py": "", "fleet/api/__init__.py": "", "fleet/api/routes/__init__.py": "",
             "fleet/jobs/__init__.py": "", "fleet/config/__init__.py": ""}

    for area in AREAS:
        files[f"fleet/{area}/__init__.py"] = ""

    for index in range(42):
        area = AREAS[index % len(AREAS)]
        noun = NOUNS[index % len(NOUNS)]
        body = f'"""{noun.capitalize()} handling for the {area} layer."""\n\nimport logging\n\nlog = logging.getLogger(__name__)\n'
        body += generated_class(generator, noun)

        for _ in range(5):
            body += generated_function(generator, noun)

        files[f"fleet/{area}/{noun}_{index:02d}.py"] = body

    files.update(CHAIN)

    return files


def build(workdir: Path) -> None:
    for relative, text in generated_files().items():
        path = workdir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def solve(workdir: Path) -> str:
    (workdir / "answer.txt").write_text(ANSWER + "\n")

    return f"{ANSWER} seconds: the export profile asks for 90, and the storage client clamps every call to 75."


def check(workdir: Path, reply: str) -> tuple[bool, str]:
    if (blocked := not_blocked(workdir)) is not None:
        return blocked

    answer = read_answer(workdir, "answer.txt")

    if answer is None:
        return False, "no answer.txt"

    normalized = answer.lower().removesuffix("seconds").removesuffix("s").strip()

    return normalized in (ANSWER, ANSWER + ".0"), f"answer.txt: {answer[:30]}"


TASK = Task("t12_codebase_trace", "long-context", PROMPT, build, solve, check, short=True)
