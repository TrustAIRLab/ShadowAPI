import json, os, argparse, statistics
import numpy as np
from collections import defaultdict

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Result_Safety")

DATASET_MAP = {
    "JBB": "JBB_harmful",
    "AdvBench": "AdvBench_Subset",
    "JBB_harmful": "JBB_harmful",
    "AdvBench_Subset": "AdvBench_Subset",
}

ATTACK_DISPLAY = {
    "None": "No Attack",
    "GCG-T": "GCG",
    "Base64": "Base64",
    "Combination": "Combination",
    "FCS_CoT": "FlipAttack",
}

ATTACK_ORDER = ["None", "GCG-T", "Base64", "Combination", "FCS_CoT"]

# Attacks whose recorded Answer is a decoded base64 payload. Every failure to
# produce one -- no response, a refusal in prose, a reply that complied in plain
# text, a genuine decode error -- used to collapse into the literal string
# "No Matched String", which the judge scores as a refusal. So a large share of
# the score for these two attacks measures base64 round-tripping, not refusal.
DECODED_ATTACKS = ("Base64", "Combination")
NO_DECODE_SENTINEL = "No Matched String"


def parse_filename(fname, labels):
    """Split a result filename into (dataset, model, label, attack, run).

    `labels` are the endpoint labels to look for, and the order they are tried in
    matters: a label that happens to be a prefix of an attack name (say "FCS"
    against "FCS_CoT") would otherwise mis-split. Two rules keep it honest --
    labels are tried longest first, and the marker has to be the first "_" run
    in the remainder, since model names contain no underscore.
    """
    name = fname[:-5] if fname.endswith(".json") else fname
    for prefix, ds in [("AdvBench_Subset_", "AdvBench_Subset"),
                       ("JBB_harmful_", "JBB_harmful")]:
        if name.startswith(prefix):
            rest = name[len(prefix):]
            break
    else:
        return None
    run = None
    if "_Run" in rest:
        parts = rest.rsplit("_Run", 1)
        rest, run = parts[0], int(parts[1])
    for label in sorted(labels, key=len, reverse=True):
        marker = f"_{label}_"
        idx = rest.find(marker)
        if idx == -1 or "_" in rest[:idx]:
            continue
        model = rest[:idx]
        attack = rest[idx + len(marker):]
        return ds, model, label, attack, run
    return None


def discover_labels():
    """Endpoint labels present in RESULT_DIR, from the meta_<label>.json files.

    Falls back to nothing if no meta file exists; pass --official-label /
    --shadow-label explicitly in that case.
    """
    labels = []
    for fname in sorted(os.listdir(RESULT_DIR)):
        if fname.startswith("meta_") and fname.endswith(".json"):
            labels.append(fname[len("meta_"):-len(".json")])
    return labels


def cell_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return None
    scores = [item["Safety_Score"] for item in data]
    answers = [(item.get("Answer") or "") for item in data]
    no_decode = sum(1 for a in answers if a.strip() in ("", NO_DECODE_SENTINEL))
    decoded_scores = [s for s, a in zip(scores, answers)
                      if a.strip() not in ("", NO_DECODE_SENTINEL)]
    statuses = defaultdict(int)
    for item in data:
        st = item.get("Decode_Status")
        if st:
            statuses[st] += 1
    # A transport failure records an empty answer, which the judge scores ~0 --
    # i.e. as a refusal. So a degraded run makes the endpoint look *safer*, the
    # opposite of the bias in the utility module. Worth watching.
    errors = sum(1 for item in data if (item.get("api_error") or ""))
    return {
        "n": len(data),
        "score": float(np.mean(scores)),
        "no_decode_rate": no_decode / len(data),
        "score_if_decoded": float(np.mean(decoded_scores)) if decoded_scores else None,
        "answer_chars_median": statistics.median(len(a) for a in answers),
        "api_error_rate": errors / len(data),
        "decode_status": dict(statuses),
    }


def collect(labels, ds_filter=None, model_filter=None):
    cells = defaultdict(list)
    for fname in sorted(os.listdir(RESULT_DIR)):
        if not fname.endswith(".json") or fname.startswith("meta_"):
            continue
        parsed = parse_filename(fname, labels)
        if parsed is None:
            continue
        ds, model, label, attack, run = parsed
        if ds_filter and ds != ds_filter:
            continue
        if model_filter and model != model_filter:
            continue
        stats = cell_stats(os.path.join(RESULT_DIR, fname))
        if stats is not None:
            cells[(ds, model, label, attack)].append(stats)
    return cells


def mean_sd(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return np.mean(vals), np.std(vals, ddof=0)


def field(runs, name):
    return [r[name] for r in runs]


def compare_meta(official_label, shadow_label):
    def _read(label):
        path = os.path.join(RESULT_DIR, f"meta_{label}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    a, b = _read(official_label), _read(shadow_label)
    if not a or not b:
        return
    if a.get("judge_model") != b.get("judge_model"):
        print(f"  [warning] judge model differs: {a.get('judge_model')} vs "
              f"{b.get('judge_model')} -- StrongREJECT scores are not comparable "
              f"across different judges")
    if a.get("judge_base_url_host_sha256") != b.get("judge_base_url_host_sha256"):
        print(f"  [warning] judge endpoint differs between the two collections")
    for key, label in (("limit", "--limit"), ("num_runs", "--num-runs")):
        if a.get(key) != b.get(key):
            print(f"  [warning] {label} differs: {a.get(key)!r} vs {b.get(key)!r}")


def main():
    global RESULT_DIR
    parser = argparse.ArgumentParser(
        description="Aggregate safety results, and compare one endpoint against "
                    "another when --shadow-label is given.")
    parser.add_argument("--dataset", type=str, default=None, help="JBB or AdvBench")
    parser.add_argument("--model", type=str, default=None, help="e.g. GPT-5-Mini")
    parser.add_argument("--official-label", type=str, default="Official",
                        help="baseline label (default Official)")
    parser.add_argument("--shadow-label", type=str, action="append", default=None,
                        help="endpoint to compare against the baseline; repeat "
                             "for several")
    parser.add_argument("--labels", type=str, default=None,
                        help="comma-separated list of every label present in the "
                             "filenames; only needed when no meta_<label>.json "
                             "files exist")
    parser.add_argument("--input-root", type=str, default=None,
                        help="directory holding the result JSON and "
                             "meta_<label>.json files. Default: "
                             "safety/Result_Safety. Use it to analyse results "
                             "kept outside the repository.")
    args = parser.parse_args()
    if args.input_root:
        RESULT_DIR = os.path.abspath(args.input_root)

    if not os.path.isdir(RESULT_DIR):
        raise SystemExit(
            f"No results yet: {RESULT_DIR} does not exist.\n"
            f"Collect some first, e.g.\n"
            f"  python safety/main.py --endpoint official --datasets AdvBench "
            f"--attacks None --models GPT-5-Mini")

    if args.labels:
        labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    else:
        labels = discover_labels()
        for extra in [args.official_label] + (args.shadow_label or []):
            if extra not in labels:
                labels.append(extra)
    if not labels:
        raise SystemExit(
            "Could not work out which endpoint labels are present. Pass "
            "--labels Official,my-endpoint-1 (the label is the part of each "
            "filename between the model name and the attack name).")

    cells = collect(labels, DATASET_MAP.get(args.dataset) if args.dataset else None,
                    args.model)
    if not cells:
        raise SystemExit(
            f"No result files matched under {RESULT_DIR} for labels {labels}")

    ds_label = {"JBB_harmful": "JBB", "AdvBench_Subset": "AdvBench"}
    datasets = sorted({k[0] for k in cells})
    models = sorted({k[1] for k in cells})
    present_labels = sorted({k[2] for k in cells})

    for ds in datasets:
        for model in models:
            if not any(k[0] == ds and k[1] == model for k in cells):
                continue
            print(f"=== {ds_label.get(ds, ds)} / {model} ===")
            print(f"{'Attack':<14} | {'Endpoint':<16} | {'Runs':>4} | {'Mean':>8} | "
                  f"{'SD':>8} | {'NoDecode':>8} | {'ApiErr':>7}")
            print("-" * 82)
            for attack in ATTACK_ORDER + sorted({k[3] for k in cells} - set(ATTACK_ORDER)):
                for label in present_labels:
                    key = (ds, model, label, attack)
                    if key not in cells:
                        continue
                    runs = cells[key]
                    if len(runs) < 3:
                        print(f"  WARNING: {ds}/{model}/{label}/{attack} has only "
                              f"{len(runs)} run(s)")
                    mean, sd = mean_sd(field(runs, "score"))
                    nd, _ = mean_sd(field(runs, "no_decode_rate"))
                    ae, _ = mean_sd(field(runs, "api_error_rate"))
                    display = ATTACK_DISPLAY.get(attack, attack)
                    print(f"{display:<14} | {label:<16} | {len(runs):>4} | "
                          f"{mean:>8.4f} | {sd:>8.4f} | {nd:>7.1%} | "
                          f"{'    n/a' if ae is None else format(ae, '>6.1%')}")

            if args.shadow_label:
                for shadow in args.shadow_label:
                    print(f"\n  {args.official_label} vs {shadow}:")
                    compare_meta(args.official_label, shadow)
                    for attack in ATTACK_ORDER:
                        bk = (ds, model, args.official_label, attack)
                        sk = (ds, model, shadow, attack)
                        if bk not in cells or sk not in cells:
                            continue
                        bm, _ = mean_sd(field(cells[bk], "score"))
                        sm, _ = mean_sd(field(cells[sk], "score"))
                        line = (f"    {ATTACK_DISPLAY.get(attack, attack):<14}"
                                f"{bm:>8.4f} -> {sm:>8.4f}  ({sm - bm:+.4f})")
                        if attack in DECODED_ATTACKS:
                            bd, _ = mean_sd(field(cells[bk], "score_if_decoded"))
                            sd_, _ = mean_sd(field(cells[sk], "score_if_decoded"))
                            bn, _ = mean_sd(field(cells[bk], "no_decode_rate"))
                            sn, _ = mean_sd(field(cells[sk], "no_decode_rate"))
                            line += (f"   | no-decode {bn:>5.1%} -> {sn:>5.1%}"
                                     f" | if decoded "
                                     f"{'n/a' if bd is None else format(bd, '.4f')}"
                                     f" -> "
                                     f"{'n/a' if sd_ is None else format(sd_, '.4f')}")
                        print(line)
            print()


if __name__ == "__main__":
    main()
