"""
The mila-server command line: serves when given no command, and lists and installs the models it serves.

Serving stays configuration (MILA_* variables or a .env), so the bare command takes no flags. The model
commands are the store's own operations through the binding, and loading still never downloads.
"""
import argparse
import sys
from importlib.metadata import version


def _format_bytes(count: int) -> str:
    mebibyte = 1024.0 * 1024.0
    gibibyte = mebibyte * 1024.0

    if count >= gibibyte:
        return f"{count / gibibyte:.2f} GiB"

    return f"{count / mebibyte:.1f} MiB"


def _fit(mila, model, device_index: int) -> str:
    """Whether the model runs on the card, as the library's planner answers against its whole memory."""
    if not model.complete:
        return "-"

    fitting = mila.fitting_weight_quantizations(model.name, device_index)

    if fitting is None:
        return "unknown"

    if not fitting:
        return "no"

    if fitting[0] == mila.stored_weight_quantization(model.name):
        return "yes"

    # The forms the weights quantize to as they load, named as a load takes them.
    return "yes, at " + " or ".join(fitting)


def _list_installed() -> int:
    import mila
    from mila_llm_server.config import settings

    mila.initialize(log_level="error")

    models = mila.ModelStore().list()

    if not models:
        print("No models installed. 'mila-server models --online' lists what is published, and "
              "'mila-server install <name>' installs one.")

        return 0

    device = next((card for card in mila.cuda_devices() if card.index == settings.device_index), None)
    fit_heading = f"FITS {device.name}" if device else ""

    print(f"{'MODEL':<36}  {'ON DISK':>10}  {fit_heading}")

    for model in models:
        fit = _fit(mila, model, settings.device_index) if device else ""
        incomplete = "" if model.complete else "  (incomplete)"

        print(f"{model.name:<36}  {_format_bytes(model.bytes_on_disk):>10}  {fit}{incomplete}")

    if device is None:
        print(f"\nNo CUDA device {settings.device_index} (MILA_DEVICE_INDEX), so no model can be served here.")

    return 0


def _list_published() -> int:
    import mila

    owner = mila.default_hub_owner()
    published = mila.ModelStore().list_hub_models(owner)

    for model in published:
        print(f"{model.repository}{'  (gated)' if model.gated else ''}")

    if not published:
        print(f"{owner} publishes no model this version of Mila can load.")

    return 0


def _progress_transport(transport):
    """The binding's transport, reporting the bytes received so far on one line."""
    received = 0

    def fetch(url, headers, writer):
        def counted(chunk):
            nonlocal received
            received += len(chunk)
            print(f"\r  {_format_bytes(received)} received", end="", flush=True)

            return writer(chunk)

        return transport(url, headers, counted)

    return fetch, lambda: received > 0


def _install(names: list[str]) -> int:
    import mila

    store = mila.ModelStore()
    owner = mila.default_hub_owner()
    failures = 0

    for name in names:
        print(f"Installing {name} from {owner} ...", flush=True)

        fetch, reported = _progress_transport(mila.http_transport)

        try:
            installed = store.pull(name, owner, transport=fetch)
        except (RuntimeError, OSError) as error:
            if reported():
                print()

            print(f"Could not install '{name}': {error}", file=sys.stderr)
            failures += 1

            continue

        if reported():
            print()

        print(f"Installed {installed.name} ({_format_bytes(installed.bytes_on_disk)}).")

        if installed.license:
            print(f"License: {installed.license}. The terms are published with the model.")

    return 0 if failures == 0 else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mila-server",
        description="Serve an installed model over HTTP. With no command, starts the server, configured by "
                    "MILA_* environment variables or a .env file in the working directory.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('mila-llm-server')}")

    commands = parser.add_subparsers(dest="command", metavar="command")

    models = commands.add_parser(
        "models",
        help="list installed models and whether each runs on the card MILA_DEVICE_INDEX names",
        description="List installed models and whether each runs on the card MILA_DEVICE_INDEX names.")
    models.add_argument("--online", action="store_true", help="list the published models instead")

    install = commands.add_parser(
        "install",
        help="install published models into the local store",
        description="Install published models into the local Mila store, which Chat shares.")
    install.add_argument("names", nargs="+", metavar="name")

    return parser


def main(argv: list[str] | None = None) -> int:
    """The mila-server console script."""
    arguments = _parser().parse_args(argv)

    if arguments.command == "models":
        return _list_published() if arguments.online else _list_installed()

    if arguments.command == "install":
        return _install(arguments.names)

    from mila_llm_server.app import serve

    serve()

    return 0
