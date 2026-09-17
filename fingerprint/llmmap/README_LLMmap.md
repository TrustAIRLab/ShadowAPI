# LLMmap

Fingerprints each endpoint with LLMmap (<https://github.com/pasquini-dario/LLMmap>) and reports the top-1 identified model and its distance `D`.

## Structure

```
├── run_llmmap.py                              # fingerprinting script
├── config/                                    # endpoint and model configuration
├── reference/                                 # reference library
│   ├── model_library.json / .csv             #   60 models (.json is read)
│   └── templates/{templates,templates_index}.json
├── patches/llmmap-0.2-shadow-api-audit.patch  # diff against upstream LLMmap
└── output/                                    # your summaries
```

42 templates come from the released LLMmap library, 18 were added for the paper. `run_llmmap.py` warns if the loaded template database differs from the shipped one.

## Setup

Run every command from the artifact root.

```bash
ART=$(pwd)

git clone https://github.com/pasquini-dario/LLMmap.git ../LLMmap
cd ../LLMmap
git checkout f661d550e3409d1bdf7e00b34bcec15296c5215e
# --check first: patch would apply a drifted diff with fuzz rather than fail
git apply --check "$ART/fingerprint/llmmap/patches/llmmap-0.2-shadow-api-audit.patch" \
  && git apply "$ART/fingerprint/llmmap/patches/llmmap-0.2-shadow-api-audit.patch"
pip install -r requirements.txt
cp "$ART/fingerprint/llmmap/reference/templates/templates.json" data/pretrained_models/default/
cd "$ART"
```

Verified against commit `f661d55` (LLMmap0.2): the patch applies cleanly, and upstream ships the trained checkpoint (`data/pretrained_models/default/model.pt`, ~12 MB) that open-set inference needs — the templates alone would not be enough.

The `cp` above **replaces** upstream's 52-template file with this repository's 60. That is deliberate: 42 of the 60 are upstream's own vectors (2 under renamed keys) and 18 were added for the paper, giving exactly the 60-model library the paper used. Ten upstream templates are dropped because the paper's library does not include them. Keep a copy of upstream's file if you want to compare against it.

## Usage

```bash
cp endpoints.example.json endpoints.json     # then edit it
python fingerprint/llmmap/run_llmmap.py --llmmap-repo ../LLMmap \
    --endpoints endpoints.json --labels official      --out out/official
python fingerprint/llmmap/run_llmmap.py --llmmap-repo ../LLMmap \
    --endpoints endpoints.json --labels ep1           --out out/ep1
python fingerprint/llmmap/run_llmmap.py --compare \
    out/official/llmmap_summary.json out/ep1/llmmap_summary.json
```

**One endpoint per `--out`**: each sweep writes `llmmap_summary.json` there, so two sweeps into one directory overwrite each other, and `--compare` refuses a baseline holding several endpoints for the same model.

`endpoints.json` gives each shadow endpoint a `base_url`, an `api_key_env` and a `model_ids` map — the name a model is served under is specific to each endpoint.

Also: `--models`, `--trials` (default 3; LLMmap's own repeat count, unrelated to MET's two-runs-per-side rule), `--pause`, `--no-store-answers`.

`--device` is the torch device the LLMmap checkpoint and its embedding model run on — `cpu` (the default) is fine, the model is ~12 MB and only embeds eight short answer traces per trial. Pass `cuda` or `mps` if you have one and prefer it.

## Reading the output

`distance_mean` / `distance_std` are the Euclidean distance to the nearest template, averaged over trials (`ddof=1`). Check `top1_stable_across_trials`: an unstable top-1 means the trace is not landing on any library model, and the identity claim is weak either way. A top-1 matching the claimed model is *consistent with* the claim, not proof — the library is finite, and a model absent from it is reported as whatever it sits closest to.
