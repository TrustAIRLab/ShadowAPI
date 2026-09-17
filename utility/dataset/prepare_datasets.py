"""Check which benchmark files are present, and verify them.

This script does not download anything -- the three gated/licensed datasets you
fetch yourself. It tells you exactly what is missing and where it comes from.

AIME 2025 ships with this repository. GPQA-Diamond, MedQA and LegalBench-SCALR
do not: GPQA is released behind a gate and carries a canary string (and its CSV
also holds annotator identities), and the other two come with their own terms.
Fetch them yourself, put them where this script expects, and let it verify them.

Verification compares a fingerprint of the file's content *and row order* against
the file the audit used. Row order matters: GPQA's answer options are shuffled
with random.Random(SHUFFLE_SEED + row_index), so a re-ordered file asks
different questions even when every question is present. A mismatch is reported,
never repaired -- reordering a file to make the fingerprint match would change
the prompts that get sent.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from config.models import DATASET_FINGERPRINTS, dataset_fingerprint  # noqa: E402

PATHS = {
    "aime2025": "aime_2025/aime2025.jsonl",
    "gpqa-diamond": "gpqa/gpqa_diamond.csv",
    "medqa": "MedQA/test_us.jsonl",
    "legalbench-scalr": "LegalBench/scalr/test.tsv",
}

SOURCES = {
    "aime2025": "ships with this repository",
    "gpqa-diamond":
        "https://huggingface.co/datasets/Idavidrein/gpqa (gated: accept the terms, "
        "then take gpqa_diamond.csv). Keep it out of any public repository -- it "
        "carries a canary string and annotator names.",
    "medqa":
        "https://github.com/jind11/MedQA -- the US test split, data_clean/"
        "questions/US/test.jsonl, saved as test_us.jsonl",
    "legalbench-scalr":
        "https://huggingface.co/datasets/nguha/legalbench -- the scalr task's "
        "test split, saved as a TSV with columns index, question, choice_0..4, answer",
}

COLUMNS = {
    "aime2025": "JSON lines with keys: question, answer",
    "gpqa-diamond":
        "CSV with columns: Question, Correct Answer, Incorrect Answer 1, "
        "Incorrect Answer 2, Incorrect Answer 3",
    "medqa": "JSON lines with keys: question, options (a dict), answer_idx",
    "legalbench-scalr":
        "TSV with columns: question, choice_0, choice_1, choice_2, choice_3, "
        "choice_4, answer",
}


def records_for(benchmark, path):
    """Reuse run_eval's reader so this checks exactly what the runner will read."""
    spec_path = os.path.join(os.path.dirname(HERE), "run_eval.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_run_eval", spec_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dataset_records(benchmark, path)


def check(benchmark, verbose=True):
    path = os.path.join(HERE, PATHS[benchmark])
    expected, expected_n = DATASET_FINGERPRINTS[benchmark]

    if not os.path.exists(path):
        print(f"[missing]  {benchmark}")
        print(f"           expected at: {path}")
        print(f"           get it from: {SOURCES[benchmark]}")
        print(f"           format:      {COLUMNS[benchmark]}")
        return False

    try:
        records = records_for(benchmark, path)
    except Exception as exc:
        print(f"[unreadable] {benchmark}: {type(exc).__name__}: {exc}")
        print(f"             expected format: {COLUMNS[benchmark]}")
        return False

    got = dataset_fingerprint(records)
    if got == expected and len(records) == expected_n:
        if verbose:
            print(f"[ok]       {benchmark}: {len(records)} records, fingerprint {got}")
        return True

    print(f"[MISMATCH] {benchmark}")
    print(f"           this file: {len(records):>6} records, fingerprint {got}")
    print(f"           audited:   {expected_n:>6} records, fingerprint {expected}")
    if len(records) != expected_n:
        print(f"           The record count differs -- this looks like a different "
              f"split or release.")
    else:
        print(f"           Same number of records but different content or row "
              f"order. Row order is part of the fingerprint because GPQA shuffles "
              f"its options by row index.")
    print(f"           Nothing has been changed. You can still use this file, but "
          f"collect BOTH endpoints from it, and your numbers will not be "
          f"comparable with the paper's.")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", action="append", default=None,
                        choices=list(PATHS),
                        help="check just this one; repeat for several "
                             "(default: all four)")
    parser.add_argument("--quiet", action="store_true",
                        help="only report problems")
    args = parser.parse_args()

    targets = args.benchmark or list(PATHS)
    results = {b: check(b, verbose=not args.quiet) for b in targets}

    ok = sum(results.values())
    print(f"\n{ok}/{len(results)} benchmark file(s) match the audited ones.")
    if ok != len(results):
        print("See utility/dataset/README.md for where each file comes from.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
