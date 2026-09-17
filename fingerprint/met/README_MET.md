# Model Equality Testing (MET)

Tests whether an endpoint's responses are drawn from the same distribution as the official endpoint's, per (benchmark, model) cell. Notices differences that do not change whether an answer is right.

Follows Model Equality Testing (<https://github.com/i-gao/model-equality-testing>).

## Structure

```
└── run_met.py              # the test
```

## Settings

| | |
|---|---|
| statistic | MMD² with Hamming kernel (`mmd_hamming`) |
| completion length | `L = 200` Unicode codepoints |
| permutations | `B = 1000`, global label permutation |
| decision | `Reject` if `p < 0.05`, else `Pass` |
| seed | `0` |
| pairing | utility: question `id`; safety: `(attack, row index)` |

## Usage

```bash
python fingerprint/met/run_met.py --shadow-label ep1
```

Reads whatever is on disk under `utility/output/` and `safety/Result_Safety/`. `--official-label` (default `Official`), `--utility-root`, `--safety-root`, `--out`, and `--length` / `--permutations` / `--alpha` / `--seed` expose the statistic's parameters without changing their defaults.

## Two runs per side, minimum

The estimator removes self-pairs, so its within-side denominators are `sum_p c_p(c_p-1)`. Answer every prompt once and those are **zero**: the statistic is `nan` and the p-value machinery turns that into `p = 0.0`, reported as `Reject` — on data that may be identical. This comes from the upstream implementation and is kept rather than patched, so the statistic stays the one the paper used; the collectors refuse `--num-runs < 2` instead. The `nXX` / `nYY` columns let you confirm it: either being 0 means that verdict is meaningless.

`Pass` means no difference was detected, **not** that the endpoint is authentic — with few prompts the test has almost no power, which is what `n_prompts` is for.

Two conventions inherited from upstream: the p-value is `mean(null >= observed)` with no `(1+k)/(1+B)` correction, so it can be exactly 0; the boundary is strict `<`.
