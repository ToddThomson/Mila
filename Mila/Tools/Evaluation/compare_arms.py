"""Compare two arms of a paired evaluation, document by document.

Two arms that scored the same documents from the same prompts can be compared pairwise, which
says far more than two averages: how many answers changed, in which direction, and whether a
difference is larger than the benchmark can resolve. The comparison refuses arms whose prompts
differ, since a difference between them would then measure the prompt as well as the engine.
"""

import argparse
import json
import math
import pathlib
import re
import statistics
import sys

SAMPLES_NAME = re.compile(r"samples_(?P<task>.+)_(?P<stamp>\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")


def load_arm(directory):
    """
    Every logged sample under an arm's directory, keyed by (task, filter, doc_id). A task run
    more than once keeps its latest file, by the time lm-eval stamps into the name.
    """
    latest = {}

    for path in directory.rglob("samples_*.jsonl"):
        match = SAMPLES_NAME.search(path.name)

        if match and (match["task"] not in latest or match["stamp"] > latest[match["task"]][0]):
            latest[match["task"]] = (match["stamp"], path)

    if not latest:
        sys.exit(f"{directory} holds no samples_*.jsonl. Was the arm run with run_arm.py?")

    samples = {}

    for task, (_, path) in latest.items():
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                sample = json.loads(line)
                samples[(task, sample["filter"], sample["doc_id"])] = sample

    return samples


def check_pairing(reference, candidate):
    """The keys both arms hold, after refusing arms that saw different documents or prompts."""
    shared = sorted(reference.keys() & candidate.keys())
    unpaired = len(reference.keys() ^ candidate.keys())

    if not shared:
        sys.exit("The arms share no documents: different tasks, or different --limit.")

    different_documents = [key for key in shared if reference[key]["doc_hash"] != candidate[key]["doc_hash"]]
    different_prompts = [key for key in shared if reference[key]["prompt_hash"] != candidate[key]["prompt_hash"]]

    if different_documents:
        sys.exit(f"{len(different_documents)} documents differ between the arms, first {different_documents[0]}: "
                 "they were drawn from different dataset revisions.")

    if different_prompts:
        sys.exit(f"{len(different_prompts)} prompts differ between the arms, first {different_prompts[0]}. "
                 "Llama 3's template writes today's date into every prompt, so run both arms on the same day.")

    return shared, unpaired


def mcnemar_p(reference_only, candidate_only):
    """Exact two-sided McNemar test: the discordant pairs against a fair coin."""
    discordant = reference_only + candidate_only

    if discordant == 0:
        return 1.0

    tail = sum(math.comb(discordant, k) for k in range(min(reference_only, candidate_only) + 1))

    return min(1.0, 2 * tail / 2 ** discordant)


def compare_binary(pairs):
    """Paired outcomes scored 0 or 1: accuracy, the answers that changed, and their significance."""
    count = len(pairs)
    reference_only = sum(1 for reference, candidate in pairs if reference and not candidate)
    candidate_only = sum(1 for reference, candidate in pairs if candidate and not reference)
    difference = (candidate_only - reference_only) / count
    # Standard error of a paired difference in proportions.
    spread = (reference_only + candidate_only) - (candidate_only - reference_only) ** 2 / count
    standard_error = math.sqrt(max(spread, 0.0)) / count

    # The normal approximation can leave [-1, 1] on a handful of documents; a --limit run does.
    low = max(-1.0, difference - 1.96 * standard_error)
    high = min(1.0, difference + 1.96 * standard_error)

    return {
        "count": count,
        "reference": sum(reference for reference, _ in pairs) / count,
        "candidate": sum(candidate for _, candidate in pairs) / count,
        "difference": difference,
        "interval": (low, high),
        "lost": reference_only,
        "gained": candidate_only,
        "flip_rate": (reference_only + candidate_only) / count,
        "p": mcnemar_p(reference_only, candidate_only),
    }


def compare_continuous(pairs):
    """Paired outcomes on a scale: the mean difference and its interval."""
    count = len(pairs)
    differences = [candidate - reference for reference, candidate in pairs]
    difference = statistics.fmean(differences)
    standard_error = statistics.stdev(differences) / math.sqrt(count) if count > 1 else 0.0

    return {
        "count": count,
        "reference": statistics.fmean(reference for reference, _ in pairs),
        "candidate": statistics.fmean(candidate for _, candidate in pairs),
        "difference": difference,
        "interval": (difference - 1.96 * standard_error, difference + 1.96 * standard_error),
    }


def score_pairs(reference, candidate, keys):
    """
    Paired values per (task, filter, metric). A metric whose value is a list scores several
    instances of one document -- IFEval's instruction-level accuracy -- and pairs them by position.
    """
    pairs = {}

    for key in keys:
        task, filter_name, _ = key

        for metric in reference[key]["metrics"]:
            reference_value = reference[key][metric]
            candidate_value = candidate[key][metric]

            if isinstance(reference_value, list):
                values = list(zip(reference_value, candidate_value, strict=True))
            else:
                values = [(reference_value, candidate_value)]

            pairs.setdefault((task, filter_name, metric), []).extend(
                (float(reference_score), float(candidate_score)) for reference_score, candidate_score in values)

    results = {}

    for metric_key, values in pairs.items():
        binary = all(value in (0.0, 1.0) for pair in values for value in pair)

        if binary:
            results[metric_key] = compare_binary(values)
        else:
            results[metric_key] = compare_continuous(values)

    return results


def common_prefix_length(first, second):
    length = 0

    for first_character, second_character in zip(first, second):
        if first_character != second_character:
            break

        length += 1

    return length


def compare_replies(reference, candidate, keys):
    """
    How often the two engines wrote the same reply, per task. Greedy decoding from the same
    weights should agree until a near-tie between two tokens resolves differently; where they
    part says how early that happens.
    """
    replies = {}

    for key in keys:
        task, _, doc_id = key
        # A reply is logged once per filter; the filters see the same reply.
        reference_reply = reference[key]["resps"][0][0]
        candidate_reply = candidate[key]["resps"][0][0]
        replies.setdefault(task, {})[doc_id] = (reference_reply, candidate_reply)

    results = {}

    for task, documents in replies.items():
        identical = sum(1 for first, second in documents.values() if first == second)
        divergences = [common_prefix_length(first, second) for first, second in documents.values() if first != second]
        results[task] = {
            "count": len(documents),
            "identical": identical / len(documents),
            "median_divergence": statistics.median(divergences) if divergences else None,
        }

    return results


def percent(value):
    return f"{100 * value:.1f}"


def signed_points(value):
    return f"{100 * value:+.1f}"


def render(reference_name, candidate_name, scores, replies, unpaired):
    lines = [
        f"# {candidate_name} against {reference_name}",
        "",
        "Scores are percentages. The difference is candidate minus reference, in points, with its 95% interval.",
        "Lost and gained count the documents only the reference, or only the candidate, answered correctly.",
        "",
        "| Task | Filter | Metric | n | Reference | Candidate | Difference | 95% interval | Lost | Gained | Flipped | McNemar p |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for (task, filter_name, metric), result in sorted(scores.items()):
        low, high = result["interval"]
        row = (f"| {task} | {filter_name} | {metric} | {result['count']} | {percent(result['reference'])} | "
               f"{percent(result['candidate'])} | {signed_points(result['difference'])} | "
               f"{signed_points(low)} to {signed_points(high)} |")

        if "flip_rate" in result:
            row += f" {result['lost']} | {result['gained']} | {percent(result['flip_rate'])}% | {result['p']:.3f} |"
        else:
            row += " | | | |"

        lines.append(row)

    lines += [
        "",
        "## Replies",
        "",
        "| Task | n | Identical | Median characters before the replies part |",
        "|---|---|---|---|",
    ]

    for task, result in sorted(replies.items()):
        divergence = "" if result["median_divergence"] is None else f"{result['median_divergence']:.0f}"
        lines.append(f"| {task} | {result['count']} | {percent(result['identical'])}% | {divergence} |")

    if unpaired:
        lines += ["", f"{unpaired} samples were held by one arm only and are not counted."]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=pathlib.Path, help="the reference arm's directory, e.g. <run>/hf")
    parser.add_argument("candidate", type=pathlib.Path, help="the candidate arm's directory, e.g. <run>/mila")
    parser.add_argument("--report", type=pathlib.Path,
                        help="write the Markdown here, and the numbers beside it as .json")
    arguments = parser.parse_args()

    reference = load_arm(arguments.reference)
    candidate = load_arm(arguments.candidate)
    keys, unpaired = check_pairing(reference, candidate)
    scores = score_pairs(reference, candidate, keys)
    replies = compare_replies(reference, candidate, keys)
    report = render(arguments.reference.name, arguments.candidate.name, scores, replies, unpaired)
    print(report)

    if arguments.report:
        arguments.report.write_text(report, encoding="utf-8")
        numbers = {
            "scores": [{"task": task, "filter": filter_name, "metric": metric, **result}
                       for (task, filter_name, metric), result in sorted(scores.items())],
            "replies": replies,
            "unpaired": unpaired,
        }
        arguments.report.with_suffix(".json").write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
