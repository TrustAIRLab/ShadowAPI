import argparse, csv, glob, json, os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UTILITY = os.path.normpath(os.path.join(HERE, "..", "..", "utility", "output"))
SAFETY = os.path.normpath(os.path.join(HERE, "..", "..", "safety", "Result_Safety"))
OUT = os.path.join(HERE, "output", "met_results.csv")

L, B, ALPHA, SEED = 200, 1000, 0.05, 0
PAD = -1
UTILITY_BENCHMARKS = [("AIME 2025", "aime"), ("GPQA", "gpqa")]
SAFETY_BENCHMARKS = [("AdvBench", "AdvBench_Subset"), ("JailbreakBench", "JBB_harmful")]


def encode(text):
    """Unicode codepoints, truncated / right-padded to L with PAD."""
    a = np.frombuffer(str(text).encode("utf-32-le"), dtype=np.int32)[:L]
    return np.concatenate([a, np.full(L - len(a), PAD, dtype=np.int32)])


def utility_cell(sub, model, channel, root=None):
    out = []
    root = root or UTILITY
    for f in sorted(glob.glob(os.path.join(root, sub, model, channel, "*.csv"))):
        df = pd.read_csv(f)
        out += [(int(i), "" if pd.isna(s) else str(s))
                for i, s in zip(df["id"], df["response"])]
    return out


def safety_models(prefix, labels, root=None):
    """Model names appearing in the safety result filenames for these labels."""
    root = root or SAFETY
    tail = re.compile(r"_(?:%s)_.*_Run\d+\.json$"
                      % "|".join(re.escape(l) for l in labels))
    return sorted({tail.sub("", f)[len(prefix) + 1:]
                   for f in os.listdir(root)
                   if f.startswith(prefix + "_") and tail.search(f)})


def safety_cell(prefix, model, channel, root=None):
    root = root or SAFETY
    head = "%s_%s_%s_" % (prefix, model, channel)
    out = []
    for f in sorted(glob.glob(os.path.join(root, head + "*_Run*.json"))):
        attack = os.path.basename(f)[len(head):].rsplit("_Run", 1)[0]
        with open(f, encoding="utf-8") as fh:
            out += [((attack, i), str(r.get("Answer", "") or ""))
                    for i, r in enumerate(json.load(fh))]
    return out


def paired(official, shadow):
    shared = {k for k, _ in official} & {k for k, _ in shadow}
    index = {k: i for i, k in enumerate(sorted(shared))}
    enc = lambda rows: np.array([[index[k]] + list(encode(t))
                                 for k, t in rows if k in shared], dtype=np.int64)
    return enc(official), enc(shadow)


def build(seq1, seq2, chunk=256):
    A = np.vstack([seq1, seq2]).astype(np.int64)
    p, C = A[:, 0], A[:, 1:]
    N, L = len(A), C.shape[1]
    S = (p[:, None] == p[None, :])
    M = np.zeros((N, N), dtype=np.float64)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        M[i:j] = (C[i:j, None, :] == C[None, :, :]).sum(-1)
    M /= L
    M *= S
    return M, S.astype(np.float64), len(seq1), len(seq2)

def stat(M, S, idx1, idx2):
    N = M.shape[0]
    v1 = np.zeros(N); v1[idx1] = 1.0
    v2 = np.zeros(N); v2[idx2] = 1.0
    Mv1, Mv2, Sv1, Sv2 = M @ v1, M @ v2, S @ v1, S @ v2
    n1, n2 = len(idx1), len(idx2)
    sXX = v1 @ Mv1 - n1;  nXX = v1 @ Sv1 - n1
    sYY = v2 @ Mv2 - n2;  nYY = v2 @ Sv2 - n2
    sXY = v1 @ Mv2;       nXY = v1 @ Sv2
    return sXX / nXX - 2 * sXY / nXY + sYY / nYY

def test(seq1, seq2, b=B, seed=SEED):
    M, S, n1, n2 = build(seq1, seq2)
    N = n1 + n2
    obs = stat(M, S, np.arange(n1), np.arange(n1, N))
    rng = np.random.default_rng(seed)
    null = np.empty(b)
    for k in range(b):
        ix = rng.permutation(N)
        null[k] = stat(M, S, ix[:n1], ix[n1:])
    return float(np.mean(null >= obs))


def diagnostics(seq1, seq2):
    """Sample counts behind one cell's statistic. Read-only: nothing here feeds
    back into stat() or test().

    nXX and nYY are sum_p c_p(c_p - 1), the denominators of the within-side terms
    of the MMD estimator. They are zero when a side answered every prompt once,
    which makes the statistic undefined -- hence the two-runs-per-side minimum the
    collectors enforce. A small n_prompts is the other thing to watch: with one or
    two prompts the permutation test has almost no power, so `Pass` carries no
    information.
    """
    from collections import Counter
    p1 = seq1[:, 0].tolist() if len(seq1) else []
    p2 = seq2[:, 0].tolist() if len(seq2) else []
    c1, c2 = Counter(p1), Counter(p2)
    return {
        "n_prompts": len(set(p1) | set(p2)),
        "n_X": len(seq1),
        "n_Y": len(seq2),
        "nXX": sum(c * (c - 1) for c in c1.values()),
        "nYY": sum(c * (c - 1) for c in c2.values()),
    }


def main():
    global L  # encode() reads it; --length overrides the default of 200
    parser = argparse.ArgumentParser(
        description="Model Equality Testing between a baseline endpoint and one or "
                    "more endpoints under audit. Reports the same MMD-with-Hamming "
                    "statistic, permutation p-value and Reject/Pass decision as the "
                    "audit, over whatever runs are on disk.")
    parser.add_argument("--official-label", default="Official",
                        help="baseline label (default Official)")
    parser.add_argument("--shadow-label", action="append", default=None,
                        help="endpoint to test against the baseline; repeat for "
                             "several. Default: every other label found on disk")
    parser.add_argument("--utility-root", default=UTILITY,
                        help="utility/output directory")
    parser.add_argument("--safety-root", default=SAFETY,
                        help="safety/Result_Safety directory")
    parser.add_argument("--out", default=OUT, help="CSV to write")
    parser.add_argument("--length", type=int, default=L,
                        help=f"completion length in codepoints (default {L})")
    parser.add_argument("--permutations", type=int, default=B,
                        help=f"permutations (default {B})")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"rejection threshold (default {ALPHA})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"permutation seed (default {SEED})")
    args = parser.parse_args()
    L = args.length

    rows = []
    fmt = "%-15s %-24s %-16s %9s %8s %8s %7s %7s"
    print(fmt % ("benchmark", "model", "endpoint", "decision", "p", "n_prompts",
                 "nXX", "nYY"))

    def cell(name, model, label, official, shadow):
        seq1, seq2 = paired(official, shadow)
        diag = diagnostics(seq1, seq2)
        p = test(seq1, seq2, b=args.permutations, seed=args.seed)
        decision = "Reject" if p < args.alpha else "Pass"
        print(fmt % (name, model, label, decision, "%.4f" % p,
                     diag["n_prompts"], diag["nXX"], diag["nYY"]), flush=True)
        rows.append([name, model, label, decision, round(p, 6),
                     diag["n_prompts"], diag["n_X"], diag["n_Y"],
                     diag["nXX"], diag["nYY"]])

    def labels_in(present):
        if args.shadow_label:
            return [l for l in args.shadow_label if l in present]
        return [l for l in sorted(present) if l != args.official_label]

    if os.path.isdir(args.utility_root):
        for name, sub in UTILITY_BENCHMARKS:
            base = os.path.join(args.utility_root, sub)
            if not os.path.isdir(base):
                continue
            for model in sorted(m for m in os.listdir(base)
                                if os.path.isdir(os.path.join(base, m))):
                mdir = os.path.join(base, model)
                present = {d for d in os.listdir(mdir)
                           if os.path.isdir(os.path.join(mdir, d))}
                if args.official_label not in present:
                    continue
                official = utility_cell(sub, model, args.official_label,
                                        args.utility_root)
                if not official:
                    continue
                for label in labels_in(present):
                    shadow = utility_cell(sub, model, label, args.utility_root)
                    if not shadow:
                        # An empty label directory -- an interrupted collection,
                        # or --limit 0. Skipping keeps the cells already computed;
                        # the CSV is only written at the end.
                        print(f"[skipped] {name} / {model} / {label}: no run files")
                        continue
                    cell(name, model, label, official, shadow)

    if os.path.isdir(args.safety_root):
        files = os.listdir(args.safety_root)
        present = {f[len("meta_"):-len(".json")] for f in files
                   if f.startswith("meta_") and f.endswith(".json")}
        present |= {args.official_label} | set(args.shadow_label or [])
        for name, prefix in SAFETY_BENCHMARKS:
            for model in safety_models(prefix, present, args.safety_root):
                official = safety_cell(prefix, model, args.official_label,
                                       args.safety_root)
                if not official:
                    continue
                for label in labels_in(present):
                    shadow = safety_cell(prefix, model, label, args.safety_root)
                    if shadow:
                        cell(name, model, label, official, shadow)

    if not rows:
        raise SystemExit(
            f"No comparable cells found.\n"
            f"  utility: {args.utility_root}\n"
            f"  safety:  {args.safety_root}\n"
            f"Collect a baseline and at least one other endpoint first, and check "
            f"--official-label / --shadow-label match the directory and file names.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "model", "endpoint", "decision", "p_value",
                    "n_prompts", "n_X", "n_Y", "nXX", "nYY"])
        w.writerows(rows)
    print("\nwrote %s (%d cells)" % (args.out, len(rows)))
    print("\nReading these results: `Reject` means the two endpoints' response "
          "distributions differ.\n`Pass` means no difference was detected -- it is "
          "not evidence that the endpoint is\nauthentic, and with few prompts "
          "(small n_prompts) the test has almost no power.\nnXX or nYY of 0 means "
          "a side answered every prompt once and the statistic is\nundefined; "
          "collect at least two runs per side.")


if __name__ == "__main__":
    main()
