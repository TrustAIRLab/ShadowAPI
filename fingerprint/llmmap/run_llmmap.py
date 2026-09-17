import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.models import ENDPOINT_KINDS, NUM_TRIALS, OFFICIAL_MODEL_IDS, get_model_config


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "output")
TEMPLATES_INDEX = os.path.join(HERE, "reference", "templates", "templates_index.json")

LLMMAP_COMMIT = "f661d550e3409d1bdf7e00b34bcec15296c5215e"

SCHEMA = "shadow-api-audit/llmmap-summary/v1"

FAMILY_PREFIXES = ("GPT", "Gemini", "DeepSeek")


def family_of(model_key):
    for prefix in FAMILY_PREFIXES:
        if model_key.startswith(prefix):
            return prefix
    return "unknown"


def load_endpoints(path):
    """Read the endpoints to probe from a JSON config.

        {
          "official":      {"kind": "official"},
          "my-endpoint-1": {"kind": "shadow",
                            "base_url": "https://example.test/v1",
                            "api_key_env": "SHADOW_API_KEY",
                            "model_ids": {"GPT-5": "gpt-5"}}
        }

    An official entry needs no model_ids -- the published ids are used. A shadow
    entry must map each model you want probed to the name that endpoint serves it
    under.
    """
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict) or not config:
        raise SystemExit(f"{path}: expected a non-empty object of label -> endpoint")

    table = []
    for label, spec in config.items():
        kind = spec.get("kind")
        if kind not in ENDPOINT_KINDS:
            raise SystemExit(
                f"{path}: endpoint {label!r} has kind={kind!r}; "
                f"expected one of {list(ENDPOINT_KINDS)}")
        if kind == "official":
            model_ids = {m: None for m in (spec.get("model_ids") or OFFICIAL_MODEL_IDS)}
        else:
            model_ids = spec.get("model_ids") or {}
            if not model_ids:
                raise SystemExit(
                    f"{path}: shadow endpoint {label!r} needs a model_ids map, e.g. "
                    f'"model_ids": {{"GPT-5": "gpt-5"}} -- the name a model is '
                    f"served under is specific to each endpoint")
            if not spec.get("base_url"):
                raise SystemExit(f"{path}: shadow endpoint {label!r} needs base_url")
        for model_key, model_id_override in model_ids.items():
            if model_key not in OFFICIAL_MODEL_IDS:
                raise SystemExit(
                    f"{path}: endpoint {label!r} names unknown model {model_key!r}. "
                    f"Known: {sorted(OFFICIAL_MODEL_IDS)}")
            table.append({
                "label": label,
                "kind": kind,
                "model_key": model_key,
                "model_id_override": model_id_override,
                "base_url": spec.get("base_url"),
                "api_key_env": spec.get("api_key_env"),
                "family": family_of(model_key),
            })
    return table

def load_llmmap(repo, templates_dir, device):
    repo = os.path.abspath(os.path.expanduser(repo))
    if not os.path.isdir(os.path.join(repo, "LLMmap")):
        raise SystemExit(
            f"{repo} does not look like an LLMmap checkout "
            f"(no LLMmap/ package). See README_LLMmap.md."
        )
    sys.path.insert(0, repo)
    from LLMmap.inference import load_LLMmap

    templates_dir = templates_dir or os.path.join(
        repo, "data", "pretrained_models", "default"
    )
    conf, llmmap = load_LLMmap(templates_dir, device=device)
    if not llmmap.ready:
        raise SystemExit(
            f"{templates_dir} carries no template file; open-set inference needs one. "
            f"Copy reference/templates/templates.json into it."
        )
    check_template_db(llmmap)
    return conf, llmmap, templates_dir


def check_template_db(llmmap):
    with open(TEMPLATES_INDEX, encoding="utf-8") as f:
        index = json.load(f)
    expected = {t["template_key"] for t in index["templates"]}
    loaded = set(llmmap.templates_map)
    if loaded == expected:
        print(f"template database: {len(loaded)} templates, matches "
              f"reference/templates/templates.json")
        return
    print(f"[warning] loaded template database differs from the shipped one "
          f"({len(loaded)} vs {len(expected)} templates)")
    for label, keys in (("only in the loaded database", loaded - expected),
                        ("missing from the loaded database", expected - loaded)):
        if keys:
            shown = sorted(keys)
            tail = "" if len(shown) <= 8 else f", ... ({len(shown) - 8} more)"
            print(f"  {label} ({len(shown)}): {', '.join(shown[:8])}{tail}")


def ask(client, model_id, temperature, seed, query, retries=3, backoff=5.0):
    kwargs = {"model": model_id, "messages": [{"role": "user", "content": query}]}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed

    last = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or "", None
        except Exception as exc:  
            last = f"{type(exc).__name__}: {exc}"
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    return "", last


def probe_once(llmmap, client, model_id, temperature, seed, pause):
    queries = list(llmmap.queries)
    answers, errors = [], []
    for query in queries:
        answer, error = ask(client, model_id, temperature, seed, query)
        answers.append(answer)
        errors.append(error)
        if pause:
            time.sleep(pause)

    distances = np.asarray(llmmap(answers), dtype=float)
    labels = [llmmap.label_map[i] for i in range(len(distances))]
    order = np.argsort(distances)
    return {
        "queries": queries,
        "answers": answers,
        "errors": errors,
        "num_failed_queries": sum(e is not None for e in errors),
        "distances": {labels[i]: float(distances[i]) for i in range(len(labels))},
        "top1_model": labels[order[0]],
        "top1_distance": float(distances[order[0]]),
        "ranking": [
            {"rank": r + 1, "model": labels[i], "distance": float(distances[i])}
            for r, i in enumerate(order)
        ],
    }

def aggregate(trials):
    labels = list(trials[0]["distances"])
    matrix = np.array([[t["distances"][m] for m in labels] for t in trials], dtype=float)
    top1 = labels[int(np.argmin(matrix.mean(axis=0)))]
    column = matrix[:, labels.index(top1)]
    return {
        "top1_model": top1,
        "distance_mean": round(float(column.mean()), 2),
        "distance_std": (
            round(float(column.std(ddof=1)), 2) if len(column) > 1 else None
        ),
        "distance_per_trial": [round(float(v), 2) for v in column],
        "top1_per_trial": [t["top1_model"] for t in trials],
        "top1_stable_across_trials": len({t["top1_model"] for t in trials}) == 1,
    }

def run_live(args, table):
    from config.providers import build_client

    conf, llmmap, templates_dir = load_llmmap(
        args.llmmap_repo, args.templates_dir, args.device
    )
    if args.models:
        table = [t for t in table if t["model_key"] in args.models]
    if args.labels:
        table = [t for t in table if t["label"] in args.labels]
    if not table:
        raise SystemExit("no endpoint matches --models / --labels")

    raw_root = os.path.join(args.out, "raw")
    rows = []
    with open(TEMPLATES_INDEX, encoding="utf-8") as f:
        library_index = {t["template_key"]: t["index"] for t in json.load(f)["templates"]}

    for entry in table:
        model_key, label = entry["model_key"], entry["label"]
        is_official = entry["kind"] == "official"
        model_id, temperature, seed = get_model_config(
            model_key, is_official=is_official,
            model_id_override=entry["model_id_override"])
        print(f"\n=== {model_key} @ {label} ({entry['kind']}) -> {model_id}")
        try:
            client = build_client(model_key, entry["kind"],
                                  entry["base_url"], entry["api_key_env"])
        except Exception as exc:
            print(f"  skipped: {exc}")
            continue

        trials, records = [], []
        for trial in range(1, args.trials + 1):
            try:
                record = probe_once(
                    llmmap, client, model_id, temperature, seed, args.pause
                )
            except Exception:  
                traceback.print_exc()
                break
            record.update(
                {
                    "claimed_model": model_key,
                    "endpoint": label,
                    "requested_model_id": model_id,
                    "temperature": temperature,
                    "seed": seed,
                    "trial": trial,
                    "llmmap_commit": LLMMAP_COMMIT,
                    "templates_dir": templates_dir,
                    "collected_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            )
            path = os.path.join(raw_root, model_key, label, f"trial{trial}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            stored = dict(record)
            if not args.store_answers:
                stored.pop("answers")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(stored, f, indent=2, ensure_ascii=False)

            trials.append(record)
            records.append(os.path.relpath(path, args.out))
            print(
                f"  trial {trial}: {record['top1_model']} "
                f"[D {record['top1_distance']:.2f}] "
                f"({record['num_failed_queries']} failed queries)"
            )

        if not trials:
            continue

        agg = aggregate(trials)
        row = {
            "family": entry["family"],
            "claimed_model": model_key,
            "endpoint": label,
            "endpoint_kind": entry["kind"],
            "is_reference": is_official,
            "requested_model_id": model_id,
            "temperature": temperature,
            "seed": seed,
            "trials_completed": len(trials),
            "top1_model": agg["top1_model"],
            "top1_library_index": library_index.get(agg["top1_model"]),
            "distance_mean": agg["distance_mean"],
            "distance_std": agg["distance_std"],
            "distance_per_trial": agg["distance_per_trial"],
            "top1_per_trial": agg["top1_per_trial"],
            "top1_stable_across_trials": agg["top1_stable_across_trials"],
            "raw_records": records,
        }
        rows.append(row)

    return {
        "schema": SCHEMA,
        "generated": "live",
        "endpoints_config": os.path.abspath(args.endpoints),
        "model_library": os.path.relpath(
            os.path.join(HERE, "reference", "model_library.json"), args.out
        ),
        "protocol": {
            "trials": args.trials,
            "queries_per_trial": len(llmmap.queries),
            "llmmap_commit": LLMMAP_COMMIT,
            "templates_dir": templates_dir,
            "template_library_size": len(llmmap.templates_map),
            "distance_fn": conf.get("distance_fn", "euclidean"),
            "embedding_model": conf.get("embedding_model_id"),
            "statistic": "mean and sample standard deviation (ddof=1) of D over the trials",
            "top1_rule": "arg-min of the per-template distance averaged over the trials",
        },
        "endpoints": rows,
    }


def compare(baseline_path, other_path):
    """Diff two summaries produced by this script: does each endpoint fingerprint
    as the same model as the baseline did?"""
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(other_path, encoding="utf-8") as f:
        other = json.load(f)

    # Key the baseline on (model, endpoint) so a summary holding several
    # endpoints cannot silently collapse to whichever row came last -- that
    # would compare an endpoint against itself and report "agree".
    base_rows = baseline["endpoints"]
    base_official = [e for e in base_rows if e.get("endpoint_kind") == "official"]
    if base_official and len(base_official) != len(base_rows):
        others = sorted({e.get("endpoint", "?") for e in base_rows
                         if e.get("endpoint_kind") != "official"})
        print(f"[warning] the baseline summary also contains non-official "
              f"endpoint(s) {others}; using only its official rows as the "
              f"baseline. Sweep one endpoint per --out directory to avoid this.\n")
        base_rows = base_official
    seen = {}
    for e in base_rows:
        key = e["claimed_model"]
        if key in seen and seen[key].get("endpoint") != e.get("endpoint"):
            raise SystemExit(
                f"The baseline summary has more than one row for {key!r} "
                f"({seen[key].get('endpoint')!r} and {e.get('endpoint')!r}). "
                f"Sweep one endpoint per --out directory so each summary holds a "
                f"single endpoint, then compare the two files.")
        seen[key] = e
    base_by_model = seen

    def fmt(e):
        std = e.get("distance_std")
        mean = e.get("distance_mean")
        s = f"{e['top1_model']} " \
            f"{'%.2f' % mean if mean is not None else 'n/a'}+-" \
            f"{'%.2f' % std if std is not None else ' n/a'}"
        trials = e.get("trials_completed")
        if trials is not None and trials < baseline.get("protocol", {}).get("trials", trials):
            s += f" [{trials} trial(s)]"
        return s

    header = f"{'claimed model @ endpoint':<40} {'this endpoint':<40} {'baseline':<40} agree"
    print(header)
    print("-" * len(header))
    agree = compared = 0
    for row in other["endpoints"]:
        ref = base_by_model.get(row["claimed_model"])
        if ref is None:
            print(f"{row['claimed_model'] + ' @ ' + row.get('endpoint', '?'):<40} "
                  f"{fmt(row):<40} {'(not in baseline)':<40} -")
            continue
        compared += 1
        same = row["top1_model"] == ref["top1_model"]
        agree += same
        print(f"{row['claimed_model'] + ' @ ' + row.get('endpoint', '?'):<40} "
              f"{fmt(row):<40} {fmt(ref):<40} {'yes' if same else 'NO'}")

    if compared:
        print(f"\ntop-1 identity agrees with the baseline on {agree}/{compared} "
              f"of the compared endpoints")
    missing = sorted(set(base_by_model) - {r["claimed_model"] for r in other["endpoints"]})
    if missing:
        print(f"{len(missing)} model(s) in the baseline were not probed here: "
              f"{', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(
        description="Fingerprint endpoints with LLMmap and write one JSON summary. "
                    "Needs a patched upstream LLMmap checkout (--llmmap-repo) and "
                    "an endpoints config (--endpoints); see README_LLMmap.md. Use "
                    "--compare to diff two summaries afterwards."
    )
    parser.add_argument("--llmmap-repo",
                        help="path to a patched upstream LLMmap checkout")
    parser.add_argument("--endpoints", default=None,
                        help="JSON config of endpoints to probe "
                             "(see endpoints.example.json)")
    parser.add_argument("--templates-dir", default=None,
                        help="LLMmap model directory (default: <repo>/data/pretrained_models/default)")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--models", nargs="*", default=None,
                        help="restrict to these model names")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="restrict to these endpoint labels")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pause", type=float, default=0.3,
                        help="seconds between probes (rate limiting)")
    parser.add_argument("--no-store-answers", dest="store_answers", action="store_false",
                        help="omit raw endpoint answers from the per-trial records")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE.json", "OTHER.json"),
                        help="diff two summaries written by this script and exit")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return

    if not args.llmmap_repo or not args.endpoints:
        parser.print_usage()
        raise SystemExit(
            "\nFingerprinting needs both:\n"
            "  --llmmap-repo PATH   a patched upstream LLMmap checkout\n"
            "  --endpoints FILE     which endpoints to probe\n"
            "\nSee fingerprint/llmmap/README_LLMmap.md for the checkout and patch "
            "steps, and endpoints.example.json for the config format.\n"
            "To diff two summaries you already have: "
            "--compare BASELINE.json OTHER.json")

    table = load_endpoints(args.endpoints)
    os.makedirs(args.out, exist_ok=True)
    summary = run_live(args, table)

    path = os.path.join(args.out, "llmmap_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {path}  ({len(summary['endpoints'])} endpoints)")


if __name__ == "__main__":
    main()
