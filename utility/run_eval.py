import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.models import (
    DATASET_FINGERPRINTS,
    ENDPOINT_KINDS,
    MIN_TRIALS,
    NUM_TRIALS,
    OFFICIAL_MODEL_IDS,
    dataset_fingerprint,
    get_model_config,
    item_key,
)
from config.providers import MissingCredential, build_client, list_models
from prompts.benchmark_prompts import (
    AIME_PROMPT_TEMPLATE,
    GPQA_PROMPT_TEMPLATE,
    LEGALBENCH_PROMPT_TEMPLATE,
    MEDQA_PROMPT_TEMPLATE,
)
SHUFFLE_SEED = 42
LETTERS = ("A", "B", "C", "D")


def shuffled_options(correct, incorrect, idx):
    correct = str(correct).strip()
    choices = [str(x).strip() for x in incorrect] + [correct]
    random.Random(SHUFFLE_SEED + idx).shuffle(choices)
    return choices, LETTERS[choices.index(correct)]


def format_choices(options):
    return "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))

BASE_DIR = os.environ.get(
    "SHADOW_API_DATA_ROOT",
    os.path.dirname(os.path.abspath(__file__)),
)

DATASETS = {
    "aime2025": "dataset/aime_2025/aime2025.jsonl",
    "gpqa-diamond": "dataset/gpqa/gpqa_diamond.csv",
    "medqa": "dataset/MedQA/test_us.jsonl",
    "legalbench-scalr": "dataset/LegalBench/scalr/test.tsv",
}

OUTPUT_SUBDIR = {
    "aime2025": "aime",
    "gpqa-diamond": "gpqa",
    "medqa": "medqa",
    "legalbench-scalr": "legalbench",
}

ANSWER_FIELDS = ["id", "question", "gold_answer", "llm_answer", "is_correct", "response"]

TELEMETRY_FIELDS = [
    "elapsed_time_sec", "prompt_tokens", "completion_tokens", "total_tokens",
]

GPQA_EXTRA = ["option_A", "option_B", "option_C", "option_D", "correct_label"]

# medqa and legalbench-scalr recorded no telemetry in the audit; --telemetry adds
# it for all four benchmarks. Off by default so the columns stay as they were.
SENSITIVE_BENCHMARKS = ("medqa", "legalbench-scalr")

# Recorded for every row but never scored: item_key lets the analysis scripts
# confirm both endpoints were asked the same question; api_error separates a
# transport failure from a model answer.
AUDIT_FIELDS = ["item_key", "api_error"]


def csv_fields(benchmark, telemetry=False):
    fields = list(ANSWER_FIELDS)
    if telemetry or benchmark not in SENSITIVE_BENCHMARKS:
        fields += TELEMETRY_FIELDS
    if benchmark == "gpqa-diamond":
        fields += GPQA_EXTRA
    return fields + AUDIT_FIELDS

BOXED = r"(?:\\{1,2})boxed\{\s*(?:(?:\\{1,2})text\{\s*)?([^{}]+?)(?:\}\s*)?\}"

# Answer extraction supports the formats requested by each benchmark prompt and applies ordered fallback rules for common response variations. The latest match within the first applicable rule is used.

def extract_answer_aime(response):
    response = str(response).strip()

    matches = list(re.finditer(BOXED, response, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()

    explicit = r"(?:The final answer is|The answer is|Result is|Answer:)[\s\W]*(\d+(?:\.\d+)?)"
    matches = list(re.finditer(explicit, response, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()

    lines = [x for x in response.splitlines() if x.strip()]
    tail = re.sub(r"[).:!*\s]+$", "", lines[-1] if lines else response)
    matches = list(re.finditer(r"=\s*(-?\d+(?:\.\d+)?)", tail))
    if matches:
        return matches[-1].group(1).strip()
    n = re.findall(r"\d+(?:\.\d+)?", tail)
    if n:
        return n[-1]
    cleaned = re.sub(r"[).:!]+$", "", response)

    numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    return numbers[-1] if numbers else None

def check_correct_aime(llm_answer, gold_answer):
    if llm_answer is None or gold_answer is None or pd.isna(gold_answer):
        return False
    if str(llm_answer).strip() == str(gold_answer).strip():
        return True
    def _num(s):
        m = re.findall(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
        return float(m[0]) if m else None
    a, b = _num(llm_answer), _num(gold_answer)
    return a is not None and b is not None and a == b

def extract_letter(response, options=None, letters="ABCD"):
    response = str(response).strip()
    flat = response.replace("\n", " ")
    cls = f"[{letters}]"

    matches = list(re.finditer(rf"ANSWER:\s*({cls})", flat, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).upper()

    def by_content(content):
        if not options:
            return None
        for i, text in enumerate(options):
            text = str(text).strip()
            if text == content or text.replace(" ", "") == content.replace(" ", ""):
                return letters[i]
            try:
                if float(text) == float(content):
                    return letters[i]
            except Exception:
                pass
        return None

    matches = list(re.finditer(BOXED, response, re.IGNORECASE))
    if matches:
        content = matches[-1].group(1).strip()
        if content.upper() in letters:
            return content.upper()
        prefix = re.match(rf"^(?:Option|Choice)?\s*({cls})(?:[\)\.\:\,]|\s|$)",
                          content, re.IGNORECASE)
        if prefix:
            return prefix.group(1).upper()
        matched = by_content(content)
        if matched:
            return matched

    patterns = [
        rf"(?:^|[\s\W])(?:Correct )?Answer[\s\W]*({cls})(?=$|[\s\W])",
        rf"(?:^|[\s\W])(?:Option|Choice)[\s\W]+({cls})(?=$|[\s\W_])",
        rf"(?:Correct Answer|The correct answer|The answer|Answer|Therefore|Thus|Hence|Conclusion)"
        rf"(?:.{{0,50}}?)\b(?:is|be|select|choose|was|likely)\b(?:.{{0,30}}?)({cls})(?=$|[\s\W_])",
        rf"(?:^|\s)({cls})\)(?=$|[\s\W_])",
        rf"(?:^|\s)(?:\*|\(|\[)+({cls})(?:\*|\)|\])+(?=$|[\s\W_])",
        rf"\*\*({cls})\)",
        rf"(?:correct\s+(?:choice|option)|final\s+answer)\s*(?:is|:)\s*\**\s*({cls})\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, flat, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()

    start = re.search(
        rf"^[\s\W]*({cls})(?=[\s\)\.\:]+(?:Why|Reason|Because|The|Explanation|Rationale)|$|[\s\)\.\:]+$)",
        response, re.IGNORECASE)
    if start:
        return start.group(1).upper()

    if len(response) == 1 and response.upper() in letters:
        return response.upper()

    return None

def extract_index(response, valid="01234"):
    if not response:
        return None
    text = str(response).replace("*", "").strip()
    cls = f"[{valid}]"

    for pattern in (rf"(?:answer|option)\s*[:：]\s*({cls})", rf"is\s*({cls})"):
        ms = list(re.finditer(pattern, text, re.IGNORECASE))
        if ms:
            return ms[-1].group(1)

    numbers = re.findall(cls, text)
    return numbers[-1] if numbers else None


def score_response(benchmark, item, response):
    if benchmark == "aime2025":
        answer = extract_answer_aime(response)
        correct = check_correct_aime(answer, item["gold"])
    elif benchmark == "gpqa-diamond":
        answer = extract_letter(response, item["options"], "ABCD")
        correct = answer is not None and answer == item["gold"]
    elif benchmark == "medqa":
        answer = extract_letter(response, item["options"], item["letters"])
        correct = answer is not None and answer == item["gold"]
    elif benchmark == "legalbench-scalr":
        answer = extract_index(response)
        correct = answer is not None and str(answer) == item["gold"]
    else:
        raise ValueError(f"Unknown benchmark {benchmark!r}")
    return ("" if answer is None else answer), correct


def call_model(client, model_id, temperature, seed, prompt, retries=0, backoff=5.0):
    """One request. With retries=0 (the default) this behaves exactly as the audit
    did: a failure is recorded as the response string "[ERROR] ..." and scored.

    --retries N is opt-in because retrying changes the measurement -- it removes
    rows the audit counted. Note the failure mode is asymmetric across modules: a
    dropped call here scores as a wrong answer (endpoint looks worse), while in
    the safety runner an empty answer scores as a refusal (endpoint looks safer).
    """
    kwargs = {"model": model_id, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed

    start = time.time()
    api_error = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            response_text = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            prompt_tokens = usage.prompt_tokens if usage else None
            completion_tokens = usage.completion_tokens if usage else None
            total_tokens = usage.total_tokens if usage else None
            api_error = None
            break
        except Exception as exc:
            response_text = f"[ERROR] {exc!r}"
            prompt_tokens = completion_tokens = total_tokens = None
            api_error = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {exc!r}")
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                traceback.print_exc()

    return {
        "response": response_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "elapsed_time_sec": round(time.time() - start, 3),
        "api_error": api_error,
    }


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_items(benchmark, path):
    if benchmark == "aime2025":
        for item in _read_jsonl(path):
            question = str(item["question"]).strip()
            yield {
                "question": question,
                "gold": str(item["answer"]).strip(),
                "prompt": AIME_PROMPT_TEMPLATE.format(question=question),
                "options": None,
            }

    elif benchmark == "gpqa-diamond":
        frame = pd.read_csv(path)
        for idx, (_, row) in enumerate(frame.iterrows()):
            question = str(row["Question"]).strip()
            options, gold = shuffled_options(
                row["Correct Answer"],
                [row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                 row["Incorrect Answer 3"]],
                idx,
            )
            yield {
                "question": question,
                "gold": gold,
                "gold_text": options["ABCD".index(gold)],
                "prompt": GPQA_PROMPT_TEMPLATE.format(
                    question=question,
                    choices=format_choices(options),
                    letters="A, B, C, D",
                ),
                "options": options,
            }

    elif benchmark == "medqa":
        for item in _read_jsonl(path):
            question = str(item["question"]).strip()
            opts = item["options"]
            keys = sorted(opts)
            choices = "\n".join(f"{k}. {opts[k]}" for k in keys)
            yield {
                "question": question,
                "gold": str(item["answer_idx"]).strip(),
                "prompt": MEDQA_PROMPT_TEMPLATE.format(
                    question=question, choices=choices),
                "options": [opts[k] for k in keys],
                "letters": "".join(keys),
            }

    elif benchmark == "legalbench-scalr":
        with open(path, "r", encoding="utf-8") as handle:
            for item in csv.DictReader(handle, delimiter="\t"):
                question = str(item["question"]).strip()
                yield {
                    "question": question,
                    "gold": str(item["answer"]).strip(),
                    "prompt": LEGALBENCH_PROMPT_TEMPLATE.format(
                        question=question,
                        **{f"choice_{i}": item[f"choice_{i}"] for i in range(5)},
                    ),
                    "options": [item[f"choice_{i}"] for i in range(5)],
                }

    else:
        raise ValueError(f"Unknown benchmark {benchmark!r}")


def run_once(client, benchmark, path, model_id, temperature, seed,
             csv_path, txt_path, limit=None, retries=0, pause=0.0,
             telemetry=False):
    total = 0
    is_gpqa = benchmark == "gpqa-diamond"
    fieldnames = csv_fields(benchmark, telemetry)

    with open(csv_path, "w", encoding="utf-8", newline="") as fcsv, \
            open(txt_path, "w", encoding="utf-8") as ftxt:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames,
                                quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()

        for idx, item in enumerate(load_items(benchmark, path)):
            if limit is not None and idx >= limit:
                break
            result = call_model(client, model_id, temperature, seed,
                                item["prompt"], retries=retries)
            total += 1

            record = {
                "id": idx,
                "question": item["question"],
                "gold_answer": item["gold"],
                "item_key": item_key(item["prompt"]),
                **result,
            }
            record["llm_answer"], record["is_correct"] = score_response(
                benchmark, item, result["response"])
            # A failed request leaves "[ERROR] RuntimeError('HTTP 500')" in the
            # response column, and the extractors can find a gold answer inside
            # that text -- gold 500 against an HTTP 500, gold 3 against an HTTP
            # 403. Force such a row wrong rather than trusting what was parsed
            # out of an error string, and keep it in the denominator. On the
            # audit's own data every one of these rows already scored wrong, so
            # this changes no historical accuracy. score_response and the rest
            # of the extraction path are untouched; llm_answer still records
            # what was parsed, so the diagnostic columns are unaffected.
            if result["api_error"]:
                record["is_correct"] = False
            if is_gpqa:
                record["gold_answer"] = item["gold_text"]
                (record["option_A"], record["option_B"],
                 record["option_C"], record["option_D"]) = item["options"]
                record["correct_label"] = item["gold"]
            writer.writerow(record)
            fcsv.flush()

            ftxt.write(f"===== [{idx}] =====\nPrompt:\n{item['prompt']}\n\n")
            ftxt.write(f"Gold: {item['gold']}\n\nResponse:\n{result['response']}\n\n")

            print(f"[{benchmark}] [{idx}] {result['elapsed_time_sec']}s | "
                  f"tokens={result['total_tokens']}")
            if pause:
                time.sleep(pause)

    print(f"[{benchmark}] {total} responses -> {csv_path}")
    return total


# A label becomes part of every output filename, and the safety analyser splits
# those filenames on "_<label>_". These names would make that split ambiguous.
# Kept identical to safety/main.py::RESERVED_LABELS so one label is valid, or
# invalid, in both modules.
RESERVED_LABELS = {
    "Official", "None", "FCS", "CoT", "Base64", "Combination", "GCG-T",
    "Subset", "harmful", "Run", "JBB", "AdvBench",
}
LABEL_RE = re.compile(r"^[A-Za-z0-9-]+$")


def check_label(label, allow=()):
    if not LABEL_RE.match(label):
        raise SystemExit(
            f"--label {label!r} must match [A-Za-z0-9-]+ . It becomes part of the "
            f"output filenames, which are split on '_', so '_' is not allowed."
        )
    # Compared case-insensitively on purpose. macOS and Windows filesystems are
    # case-insensitive by default, so --label official would write over the
    # Official baseline's files rather than alongside them.
    folded = label.casefold()
    if folded in {r.casefold() for r in RESERVED_LABELS} and \
            folded not in {a.casefold() for a in allow}:
        raise SystemExit(
            f"--label {label!r} is reserved (compared without regard to case): it "
            f"collides with a benchmark, attack or endpoint name used in the output "
            f"filenames, and on a case-insensitive filesystem it would overwrite "
            f"them. Reserved: {', '.join(sorted(RESERVED_LABELS))}"
        )
    return label


def dataset_records(benchmark, path):
    """The fields that feed a prompt or a gold answer, in file order."""
    if benchmark == "aime2025":
        return [(d["question"], d["answer"]) for d in _read_jsonl(path)]
    if benchmark == "gpqa-diamond":
        frame = pd.read_csv(path)
        return [(r["Question"], r["Correct Answer"], r["Incorrect Answer 1"],
                 r["Incorrect Answer 2"], r["Incorrect Answer 3"])
                for _, r in frame.iterrows()]
    if benchmark == "medqa":
        return [tuple([d["question"]]
                      + [d["options"][k] for k in sorted(d["options"])]
                      + [d["answer_idx"]])
                for d in _read_jsonl(path)]
    if benchmark == "legalbench-scalr":
        with open(path, "r", encoding="utf-8") as handle:
            return [tuple([r["question"]] + [r[f"choice_{i}"] for i in range(5)]
                          + [r["answer"]])
                    for r in csv.DictReader(handle, delimiter="\t")]
    raise ValueError(f"Unknown benchmark {benchmark!r}")


def check_dataset(benchmark, path):
    """Compare a benchmark file against the fingerprint of the audited one.

    Never repairs a mismatch: re-ordering a file would change the GPQA option
    shuffle, and therefore the prompts that get sent.
    """
    expected, expected_n = DATASET_FINGERPRINTS[benchmark]
    records = dataset_records(benchmark, path)
    got = dataset_fingerprint(records)
    if got == expected and len(records) == expected_n:
        print(f"dataset {benchmark}: {len(records)} records, fingerprint {got} (matches the audited file)")
    else:
        print(f"[warning] dataset {benchmark} differs from the audited file:")
        print(f"          this file: {len(records)} records, fingerprint {got}")
        print(f"          audited:   {expected_n} records, fingerprint {expected}")
        print(f"          Results will not be comparable with the paper's. Both "
              f"endpoints must be collected from this same file.")
    return got


def tool_commit():
    """Short commit of this checkout, or None when it is not a git repository."""
    import subprocess
    try:
        return subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                               "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip() or None
    except Exception:
        return None

def _sha16(value):
    if not value:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def prompt_templates_sha():
    """Fingerprint the four prompt templates.

    A locally edited template silently breaks comparability between two
    collections, and a git commit hash would not show it in a dirty tree.
    """
    joined = "\x1f".join([
        AIME_PROMPT_TEMPLATE, GPQA_PROMPT_TEMPLATE,
        MEDQA_PROMPT_TEMPLATE, LEGALBENCH_PROMPT_TEMPLATE,
    ])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def write_meta(out_dir, args, label, model_id, temperature, seed,
               fingerprint, items_written):
    """Record how this collection was produced, next to its runs.

    The endpoint host and the model id are stored as hashes unless
    --record-endpoint-plaintext is passed: a host name identifies a specific
    reseller, and this file is what people paste into bug reports. The hash
    still answers the only question the analysis needs -- were both sides the
    same endpoint or not.
    """
    meta = {
        "label": label,
        "endpoint_kind": args.endpoint,
        "benchmark": args.benchmark,
        "model": args.model,
        "temperature": temperature,
        "seed": seed,
        "shuffle_seed": SHUFFLE_SEED,
        "num_runs": args.num_runs,
        "limit": args.limit,
        "retries": args.retries,
        "telemetry": bool(args.telemetry),
        "items_written": items_written,
        "dataset_fingerprint": fingerprint,
        "prompt_templates_sha": prompt_templates_sha(),
        "sampling_overrides": getattr(args, "overrides", {}) or None,
        "tool_commit": tool_commit(),
        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    host = urlparse(args.base_url).netloc if args.base_url else None
    if args.record_endpoint_plaintext:
        meta["base_url_host"] = host
        meta["model_id"] = model_id
    else:
        meta["base_url_host_sha256"] = _sha16(host)
        meta["model_id_sha256"] = _sha16(model_id)

    path = os.path.join(out_dir, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect benchmark responses from one endpoint. Run it once "
                    "for --endpoint official and once per endpoint you want to "
                    "audit, then compare with aggregate_results_utility.py.")
    parser.add_argument("--endpoint", choices=list(ENDPOINT_KINDS),
                        help="official = the model vendor's own API; "
                             "shadow = an endpoint you supply")
    parser.add_argument("--benchmark", choices=list(DATASETS))
    parser.add_argument("--model", choices=list(OFFICIAL_MODEL_IDS))
    parser.add_argument("--base-url", default=None,
                        help="shadow only: the OpenAI-compatible URL to audit")
    parser.add_argument("--api-key-env", default=None,
                        help="shadow only: env var holding the key (default SHADOW_API_KEY)")
    parser.add_argument("--model-id", default=None,
                        help="shadow only: the model name on that endpoint "
                             "(see --list-models)")
    parser.add_argument("--label", default=None,
                        help="names the output directory; default 'Official' for "
                             "--endpoint official, required for shadow")
    parser.add_argument("--num-runs", type=int, default=NUM_TRIALS,
                        help=f"trials to average over (default {NUM_TRIALS}, "
                             f"minimum {MIN_TRIALS})")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N questions (for a cheap smoke test)")
    parser.add_argument("--retries", type=int, default=0,
                        help="retry a failed request N times with exponential "
                             "backoff. Default 0, which is what the audit did: a "
                             "failure is recorded and scored")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds between requests (rate limiting)")
    parser.add_argument("--telemetry", action="store_true",
                        help="also record latency and token counts for medqa and "
                             "legalbench-scalr (off by default, as in the audit)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="override the temperature sent. Default: 0.0, or "
                             "omitted for the official GPT-5 family, which rejects it")
    parser.add_argument("--no-temperature", action="store_true",
                        help="send no temperature at all -- for an endpoint that "
                             "rejects the parameter")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the seed sent. Default: 42, or omitted for "
                             "official Gemini, which rejects it")
    parser.add_argument("--no-seed", action="store_true",
                        help="send no seed at all -- for an endpoint that rejects "
                             "the parameter (some reject it with HTTP 400, which "
                             "would otherwise fail every request)")
    parser.add_argument("--data-root", default=BASE_DIR)
    parser.add_argument("--output-root", default=None,
                        help="default: <data-root>/output")
    parser.add_argument("--tag", default="")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace run files that already exist")
    parser.add_argument("--list-models", action="store_true",
                        help="print the model ids an endpoint offers and exit")
    parser.add_argument("--record-endpoint-plaintext", action="store_true",
                        help="write the endpoint host and model id into meta.json "
                             "in the clear. Off by default: a host name names a "
                             "specific reseller, and meta.json travels with "
                             "shared results")
    args = parser.parse_args()

    if args.list_models:
        if not args.base_url:
            raise SystemExit("--list-models needs --base-url")
        for name in list_models(args.base_url, args.api_key_env):
            print(name)
        return

    missing = [n for n in ("endpoint", "benchmark", "model")
               if getattr(args, n) is None]
    if missing:
        parser.error("the following arguments are required: "
                     + ", ".join("--" + n for n in missing))

    is_official = args.endpoint == "official"
    if is_official:
        # "Official" is the default and stays allowed when passed explicitly.
        label = check_label(args.label, allow={"Official"}) if args.label else "Official"
    else:
        if not args.label:
            raise SystemExit(
                "--endpoint shadow needs --label NAME (it names the output "
                "directory, so you can compare several endpoints)."
            )
        label = check_label(args.label)

    if args.num_runs < MIN_TRIALS:
        raise SystemExit(
            f"--num-runs {args.num_runs} is below the minimum of {MIN_TRIALS}. "
            f"Model Equality Testing divides by sum_p c_p(c_p-1), which is zero "
            f"when every question was answered once, and then reports a "
            f"meaningless verdict. The audit protocol is {NUM_TRIALS} runs."
        )

    path = os.path.join(args.data_root, DATASETS[args.benchmark])
    if not os.path.exists(path):
        raise SystemExit(
            f"Dataset not found: {path}\n"
            f"This benchmark is not shipped with the repository -- fetch it "
            f"yourself, then verify it with\n"
            f"  python utility/dataset/prepare_datasets.py --benchmark {args.benchmark}\n"
            f"which prints where the file comes from. See utility/dataset/README.md."
        )
    fingerprint = check_dataset(args.benchmark, path)

    out_root = args.output_root or os.path.join(args.data_root, "output")
    out_dir = os.path.join(out_root, OUTPUT_SUBDIR[args.benchmark],
                           args.model, label)

    suffix = f"_{args.tag}" if args.tag else ""
    stems = [f"{args.benchmark}_{args.model}_{label}_run{run}{suffix}"
             for run in range(1, args.num_runs + 1)]

    clash = [s for s in stems if os.path.exists(os.path.join(out_dir, s + ".csv"))]
    if clash and not args.overwrite:
        raise SystemExit(
            f"{len(clash)} run file(s) already exist in {out_dir}, "
            f"starting with {clash[0]}.csv\n"
            f"Pass --tag NAME to write alongside them, --output-root DIR to "
            f"write elsewhere, or --overwrite to replace them."
        )

    try:
        model_id, temperature, seed = get_model_config(
            args.model, is_official=is_official, model_id_override=args.model_id)
    except ValueError as exc:
        raise SystemExit(str(exc))

    # Overrides, applied after the audit's own choice so the defaults are
    # unchanged. They exist because an endpoint that rejects `seed` or
    # `temperature` answers every request with HTTP 400, and without a way to
    # drop the parameter the run is simply impossible.
    overrides = {}
    if args.no_temperature:
        temperature, overrides["temperature"] = None, "omitted"
    elif args.temperature is not None:
        temperature, overrides["temperature"] = args.temperature, args.temperature
    if args.no_seed:
        seed, overrides["seed"] = None, "omitted"
    elif args.seed is not None:
        seed, overrides["seed"] = args.seed, args.seed

    try:
        client = build_client(args.model, args.endpoint, args.base_url,
                              args.api_key_env)
    except (ValueError, MissingCredential) as exc:
        raise SystemExit(str(exc))

    print(f"benchmark={args.benchmark} model={args.model} endpoint={args.endpoint} "
          f"label={label} model_id={model_id} temperature={temperature} "
          f"seed={seed} runs={args.num_runs}"
          + (f" overrides={overrides}" if overrides else ""))
    if overrides:
        print("[note] sampling parameters were overridden; the other endpoint must "
              "be collected the same way or the comparison is not like-for-like")
    args.overrides = overrides

    os.makedirs(out_dir, exist_ok=True)
    print(f"writing to {out_dir}")

    counts = []
    for stem in stems:
        counts.append(run_once(
            client, args.benchmark, path, model_id, temperature, seed,
            os.path.join(out_dir, stem + ".csv"),
            os.path.join(out_dir, stem + ".txt"),
            limit=args.limit, retries=args.retries, pause=args.pause,
            telemetry=args.telemetry,
        ))

    write_meta(out_dir, args, label, model_id, temperature, seed, fingerprint,
               sum(counts))

    print(f"\n{args.benchmark} | {args.model} | {label} | "
          f"{len(counts)} run(s), {sum(counts)} responses recorded")


if __name__ == "__main__":
    main()
