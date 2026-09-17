import csv, glob, json, os, argparse, statistics
import numpy as np
from collections import defaultdict

RESULT_DIR = os.path.join(os.path.dirname(__file__), "output")

csv.field_size_limit(10 ** 9)

BENCHMARK_MAP = {
    "aime": "aime",
    "aime2025": "aime",
    "gpqa": "gpqa",
    "gpqa-diamond": "gpqa",
    "medqa": "medqa",
    "legalbench": "legalbench",
    "legalbench-scalr": "legalbench",
}

BENCHMARK_DISPLAY = {
    "aime": "AIME 2025",
    "gpqa": "GPQA (Diamond)",
    "medqa": "MedQA (USMLE)",
    "legalbench": "LegalBench (Scalr)",
}

BENCHMARK_ORDER = ["aime", "gpqa", "medqa", "legalbench"]

# Responses cut off before they state an answer were observed to land around here.
# Only used for the short_response_rate diagnostic; nothing is scored on it.
TRUNCATED_CHARS = 120


def run_stats(path):
    """Per-run figures for one recorded run.

    accuracy is the headline number and is computed exactly as before. The rest
    separate two things accuracy conflates:

      unparsed_rate       no answer could be extracted from the response
      accuracy_if_parsed  accuracy over the rows that did yield an answer

    A response truncated before it states an answer scores as a wrong answer, so
    a large unparsed_rate gap between two endpoints means the accuracy gap is at
    least partly a formatting difference rather than a capability one. Watch the
    asymmetry, though: AIME's extractor falls back to the last number in the
    text, so a truncated AIME response usually yields a *wrong* answer rather
    than no answer, and unparsed_rate stays low. response_chars is there to make
    that visible.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    correct = sum(r["is_correct"] == "True" for r in rows)
    parsed = [r for r in rows if r.get("llm_answer", "").strip()]
    correct_parsed = sum(r["is_correct"] == "True" for r in parsed)
    lengths = [len(r.get("response", "")) for r in rows]
    errors = sum(1 for r in rows if (r.get("api_error") or "").strip())

    def _nums(field):
        out = []
        for r in rows:
            v = (r.get(field) or "").strip()
            if v and v.lower() != "none":
                try:
                    out.append(float(v))
                except ValueError:
                    pass
        return out

    latency = _nums("elapsed_time_sec")
    completion = _nums("completion_tokens")
    prompt_tokens = _nums("prompt_tokens")

    lengths_sorted = sorted(lengths)

    def pct(p):
        return lengths_sorted[min(int(p * (len(lengths_sorted) - 1)), len(lengths_sorted) - 1)]

    return {
        "n": len(rows),
        "accuracy": correct / len(rows),
        "unparsed_rate": 1 - len(parsed) / len(rows),
        "accuracy_if_parsed": (correct_parsed / len(parsed)) if parsed else None,
        # A distribution, not just a median: the truncation these columns exist to
        # expose shows up as a cluster of short responses, and a median hides it.
        # TRUNCATED_CHARS is where truncated replies were observed to land.
        "response_chars_p10": pct(0.10),
        "response_chars_median": statistics.median(lengths),
        "response_chars_p90": pct(0.90),
        "short_response_rate": sum(1 for x in lengths if x < TRUNCATED_CHARS) / len(lengths),
        "api_error_rate": errors / len(rows),
        "latency_median": statistics.median(latency) if latency else None,
        "completion_tokens_median": statistics.median(completion) if completion else None,
        "prompt_tokens_median": statistics.median(prompt_tokens) if prompt_tokens else None,
        "item_keys": {r["id"]: r.get("item_key", "") for r in rows if "id" in r},
    }


def collect(bench_filter=None, model_filter=None, label_filter=None):
    """cell -> list of per-run stats, one entry per run file."""
    cells = defaultdict(list)
    if not os.path.isdir(RESULT_DIR):
        return cells
    for bench in sorted(os.listdir(RESULT_DIR)):
        bdir = os.path.join(RESULT_DIR, bench)
        if not os.path.isdir(bdir) or (bench_filter and bench != bench_filter):
            continue
        for model in sorted(os.listdir(bdir)):
            mdir = os.path.join(bdir, model)
            if not os.path.isdir(mdir) or (model_filter and model != model_filter):
                continue
            for label in sorted(os.listdir(mdir)):
                ldir = os.path.join(mdir, label)
                if not os.path.isdir(ldir):
                    continue
                if label_filter and label not in label_filter:
                    continue
                for path in sorted(glob.glob(os.path.join(ldir, "*.csv"))):
                    stats = run_stats(path)
                    if stats is not None:
                        cells[(bench, model, label)].append(stats)
    return cells


def mean_sd(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return np.mean(vals), np.std(vals, ddof=0)


def field(runs, name):
    return [r[name] for r in runs]


def read_meta(bench, model, label):
    path = os.path.join(RESULT_DIR, bench, model, label, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_meta(bench, model, official_label, shadow_label):
    """Warn about differences that make two collections incomparable."""
    a, b = read_meta(bench, model, official_label), read_meta(bench, model, shadow_label)
    if not a or not b:
        return
    for key, label in (
        ("dataset_fingerprint", "benchmark file (content or row order)"),
        ("prompt_templates_sha", "prompt templates"),
        ("shuffle_seed", "GPQA option shuffle seed"),
        ("limit", "--limit"),
        ("num_runs", "--num-runs"),
        ("temperature", "temperature"),
        ("seed", "seed"),
        ("sampling_overrides", "sampling parameter overrides"),
        ("telemetry", "--telemetry"),
    ):
        if a.get(key) != b.get(key):
            print(f"  [warning] {label} differs: {official_label}={a.get(key)!r} "
                  f"vs {shadow_label}={b.get(key)!r}")
    if a.get("base_url_host_sha256") and \
            a.get("base_url_host_sha256") == b.get("base_url_host_sha256"):
        print(f"  [warning] both sides were collected from the same endpoint host")


def check_item_keys(official_runs, shadow_runs, official_label, shadow_label):
    """Both endpoints must have been asked the same questions.

    Compares the recorded item_key per id across every run on both sides, so
    drift inside one side's own runs is caught too, on non-shared ids as well.
    This is a precondition check only; pairing itself stays on the positional
    id, as in the audit.

    Three outcomes, so do not treat the result as a bool:
      False         -- the question sets provably differ, and the comparison
                       printed below is meaningless.
      "unverified"  -- nothing to compare, or only part of the data carried a
                       key. Absence is "not checked", never "checked and
                       equal". Runs collected before the item_key column
                       existed take this path.
      True          -- every id on both sides carried a key, and they agree.
    """
    def merged(runs):
        """id -> set of keys seen for it, and how many runs carried no key."""
        out, keyless = {}, 0
        for run in runs:
            if not any(run["item_keys"].values()):
                keyless += 1
            for i, k in run["item_keys"].items():
                if k:
                    out.setdefault(i, set()).add(k)
        return out, keyless

    a, a_blank = merged(official_runs)
    b, b_blank = merged(shadow_runs)
    if not a or not b:
        print("  [unverified] no item_key recorded, so whether both sides were "
              "asked the same questions was NOT checked -- a difference below "
              "could come from a different question set.")
        return "unverified"
    # One pass over every id on either side: an id carrying more than one key
    # means the sides disagree about it, or one side changed it between its own
    # runs. Non-shared ids are included, so drift there is caught as well.
    bad = sorted(i for i in set(a) | set(b)
                 if len(a.get(i, set()) | b.get(i, set())) > 1)
    shared = set(a) & set(b)
    if bad or not shared:
        detail = (f"{len(bad)} question ids differ between the two sides, or "
                  f"between runs on one side (first ids: {bad[:5]})" if bad else
                  f"no question id in common at all ({len(a)} vs {len(b)} ids)")
        print(f"  [ERROR] {official_label} vs {shadow_label}: {detail}. The two "
              f"sides were not asked the same thing -- most likely different "
              f"benchmark files. Any comparison below is meaningless.")
        return False
    if a_blank or b_blank or set(a) != set(b):
        # Not reported as a match. The accuracies printed below are each side's
        # own full set -- nothing here is recomputed over the shared ids only.
        print(f"  [unverified] the {len(shared)} shared ids agree, but "
              f"{a_blank + b_blank} run(s) carry no item_key and "
              f"{len(set(a) ^ set(b))} id(s) appear on one side only. The "
              f"accuracies below are each side's own full set, not a "
              f"common-question comparison.")
        return "unverified"
    return True


def main():
    global RESULT_DIR
    parser = argparse.ArgumentParser(
        description="Aggregate recorded runs, and compare one endpoint against "
                    "another when --shadow-label is given.")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="aime, gpqa, medqa or legalbench")
    parser.add_argument("--model", type=str, default=None,
                        help="e.g. GPT-5-Mini")
    parser.add_argument("--official-label", type=str, default="Official",
                        help="baseline label (default Official)")
    parser.add_argument("--shadow-label", type=str, action="append", default=None,
                        help="endpoint to compare against the baseline; repeat "
                             "for several")
    parser.add_argument("--csv", type=str, default=None,
                        help="write the comparison to this path")
    parser.add_argument("--input-root", type=str, default=None,
                        help="directory holding the collected runs, laid out as "
                             "<benchmark>/<model>/<label>/. Default: "
                             "utility/output. Pass whatever you gave "
                             "run_eval.py --output-root, so results kept "
                             "outside the repository can still be analysed.")
    args = parser.parse_args()
    if args.input_root:
        RESULT_DIR = os.path.abspath(args.input_root)

    if not os.path.isdir(RESULT_DIR):
        raise SystemExit(
            f"No results yet: {RESULT_DIR} does not exist.\n"
            f"Collect some first, e.g.\n"
            f"  python utility/run_eval.py --endpoint official --benchmark aime2025 "
            f"--model GPT-4o-Mini")

    bench_filter = BENCHMARK_MAP.get(args.benchmark) if args.benchmark else None
    labels = None
    if args.shadow_label:
        labels = set(args.shadow_label) | {args.official_label}

    cells = collect(bench_filter, args.model, labels)
    if not cells:
        raise SystemExit(f"No run files matched under {RESULT_DIR}")

    benchmarks = [b for b in BENCHMARK_ORDER if any(k[0] == b for k in cells)]
    benchmarks += sorted({k[0] for k in cells} - set(BENCHMARK_ORDER))
    models = sorted({k[1] for k in cells})
    rows_out = []

    for bench in benchmarks:
        for model in models:
            present = sorted({k[2] for k in cells if k[0] == bench and k[1] == model})
            if not present:
                continue
            print(f"=== {BENCHMARK_DISPLAY.get(bench, bench)} / {model} ===")

            # Always print the plain per-endpoint table.
            print(f"{'Endpoint':<16} | {'Runs':>4} | {'Acc':>8} | {'SD':>8} | "
                  f"{'Unparsed':>8} | {'Acc|parsed':>10}")
            print("-" * 72)
            for label in present:
                runs = cells[(bench, model, label)]
                if len(runs) < 3:
                    print(f"  WARNING: {bench}/{model}/{label} has only {len(runs)} run(s)")
                acc, sd = mean_sd(field(runs, "accuracy"))
                unp, _ = mean_sd(field(runs, "unparsed_rate"))
                accp, _ = mean_sd(field(runs, "accuracy_if_parsed"))
                print(f"{label:<16} | {len(runs):>4} | {acc:>8.4f} | {sd:>8.4f} | "
                      f"{unp:>7.1%} | {'  n/a' if accp is None else format(accp, '>10.4f')}")

            # Then the official-vs-shadow differences.
            if args.shadow_label and args.official_label in present:
                base = cells[(bench, model, args.official_label)]
                b_acc, _ = mean_sd(field(base, "accuracy"))
                for label in present:
                    if label == args.official_label:
                        continue
                    print(f"\n  {args.official_label} vs {label}:")
                    ok = check_item_keys(base, cells[(bench, model, label)],
                                         args.official_label, label)
                    compare_meta(bench, model, args.official_label, label)
                    runs = cells[(bench, model, label)]
                    s_acc, _ = mean_sd(field(runs, "accuracy"))
                    delta = s_acc - b_acc
                    rel = (delta / b_acc * 100) if b_acc else float("nan")
                    print(f"    accuracy        {b_acc:>8.4f} -> {s_acc:>8.4f}  "
                          f"({delta:+.4f}, {rel:+.1f}%)")
                    for name, fmt in (("unparsed_rate", "{:>8.1%}"),
                                      ("accuracy_if_parsed", "{:>8.4f}"),
                                      ("response_chars_p10", "{:>8.0f}"),
                                      ("response_chars_median", "{:>8.0f}"),
                                      ("response_chars_p90", "{:>8.0f}"),
                                      ("short_response_rate", "{:>8.1%}"),
                                      ("api_error_rate", "{:>8.1%}"),
                                      ("latency_median", "{:>8.1f}"),
                                      ("prompt_tokens_median", "{:>8.0f}"),
                                      ("completion_tokens_median", "{:>8.0f}")):
                        bv, _ = mean_sd(field(base, name))
                        sv, _ = mean_sd(field(runs, name))
                        if bv is None or sv is None:
                            continue
                        print(f"    {name:<16}" + fmt.format(bv) + " -> " +
                              fmt.format(sv))
                    rows_out.append({
                        "benchmark": bench, "model": model,
                        "official_label": args.official_label, "shadow_label": label,
                        "official_accuracy": round(b_acc, 6),
                        "shadow_accuracy": round(s_acc, 6),
                        "delta": round(delta, 6),
                        "relative_pct": round(rel, 3),
                        "item_keys_match": ok,
                    })
            print()

    if args.csv and rows_out:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"wrote {args.csv} ({len(rows_out)} comparisons)")


if __name__ == "__main__":
    main()
