"""Does a build call each tool by exactly the name it was declared with?

The 2.82-bit Qwen 3.8 build named four of RECORD_PROBES wrong where the FP4 build named all twelve
(Qwen3.8.md, "The 2.82-bit build misnames tools"). HELD_OUT_PROBES are names calibration_text.py keeps out
of the calibration text, so a build that passes them is not recalling text it was fitted on. Greedy, thinking
off, one tool declared per probe, rendered and parsed by the installed model's own chat template.

    python tool_name_probe.py Qwen3.8-27B-cb2-3 --device 1
"""

import argparse
import json
import sys

# (tool name, user request, the one parameter it declares)
RECORD_PROBES = [
    ("get_weather", "What is the weather in Paris right now?", "city"),
    ("get_current_weather", "What is the weather in Paris right now?", "city"),
    ("read_file", "Show me the contents of notes.txt", "file_path"),
    ("search_web", "Search the web for the latest CUDA release", "search_query"),
    ("getWeather", "What is the weather in Paris right now?", "city"),
    ("Bash", "List the files in the current directory", "command"),
    ("list_directory", "List the files in the current directory", "dir_path"),
    ("get_stock_price", "What is NVIDIA's stock price?", "ticker"),
    ("send_email", "Email Bob that the build is green", "recipient"),
    ("weather_forecast", "Will it rain in Paris tomorrow?", "city"),
    ("fetch_weather_data", "What is the weather in Paris right now?", "city"),
    ("create_calendar_event", "Book a meeting with Alice at 3pm", "title"),
]

HELD_OUT_PROBES = [
    ("get_tide_times", "When is high tide in Brighton today?", "location"),
    ("reserve_parking_space", "Reserve a parking space near the stadium for Saturday", "venue"),
    ("lookup_flight_status", "Is flight BA117 on time?", "flight_number"),
    ("summarize_pull_request", "Summarize pull request 4821", "pull_request_id"),
    ("translate_subtitle_file", "Translate movie.srt into French", "file_path"),
    ("check_library_hold", "Is my library hold for Dune ready to collect?", "book_title"),
    ("query_inventory_level", "How many M8 bolts are left in the warehouse?", "item_name"),
    ("archive_stale_tickets", "Archive the support tickets older than 90 days", "age_days"),
    ("fetch_pollen_count", "What is the pollen count in Denver today?", "city"),
    ("create_shipping_label", "Make a shipping label for order 5531", "order_id"),
    ("getTrainDepartures", "When does the next train leave Leeds?", "station"),
    ("Grep", "Find every TODO comment in the source tree", "pattern"),
]


def probe_names():
    """Every tool name either probe set declares."""
    return [name for name, _, _ in RECORD_PROBES + HELD_OUT_PROBES]


def run(model_name, device, context_length):
    import mila

    mila.initialize()
    tokenizer = mila.BpeTokenizer.from_store(model_name)
    model = mila.QwenModel.from_store(model_name, context_length=context_length, device_index=device)
    protocol = mila.ChatProtocol.for_model(model_name)
    failures = 0

    for label, probes in (("record", RECORD_PROBES), ("held out", HELD_OUT_PROBES)):
        wrong = 0
        print(f"-- {label}")

        for tool, request, parameter in probes:
            tools = [{"type": "function", "function": {
                "name": tool, "description": request,
                "parameters": {"type": "object", "properties": {parameter: {"type": "string"}},
                               "required": [parameter]}}}]
            prompt = protocol.format_prompt([{"role": "user", "content": request}],
                                            {"enable_thinking": "false"}, json.dumps(tools))
            tokens = []
            model.generate(tokenizer.encode(prompt["text"]), tokens.append, max_new_tokens=80,
                           temperature=1.0, top_k=1)
            reply = protocol.parse_reply(tokenizer.decode(tokens), prompt)
            calls = reply["tool_calls"]
            called = calls[0]["name"] if calls else "(no call)"
            arguments = calls[0]["arguments"] if calls else {}

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            correct = called == tool and parameter in arguments
            wrong += not correct
            print(f"{'ok' if correct else 'WRONG':5s} {tool:24s} -> {called!r:28s} arguments {sorted(arguments)}")

        print(f"{label}: {len(probes) - wrong} of {len(probes)} correct\n")
        failures += wrong

    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="installed store name, e.g. Qwen3.8-27B-cb2-3")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--context-length", type=int, default=4096, help="tokens to load at (default: 4096)")
    args = parser.parse_args()

    sys.exit(1 if run(args.model, args.device, args.context_length) else 0)


if __name__ == "__main__":
    main()
