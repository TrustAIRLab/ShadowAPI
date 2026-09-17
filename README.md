# ShadowAPI_OpenSource

> **Disclaimer.** This repository contains real jailbreak prompts, including examples of hateful and abusive language, and running it writes harmful model output to disk. Reader discretion is advised. It is intended for research purposes only; any misuse is strictly prohibited.

Audit toolkit for the shadow-API study. You supply the endpoint; the same code collects from it and from the official API, then compares. Three modules:

- **`utility/`** — utility / performance audit on four benchmarks (AIME 2025,
  GPQA-Diamond, MedQA, LegalBench-SCALR).
- **`safety/`** — safety audit (AdvBench / JailbreakBench) under five attack
  settings, scored with the StrongREJECT rubric.
- **`fingerprint/`** — identity audit: `llmmap/` (LLMmap fingerprinting) and
  `met/` (Model Equality Testing over the utility and safety responses).

No results are included, and no third-party service is named. The tool reproduces the *method* of the paper, not its numbers.

## Install

```bash
pip install -r requirements.txt
python utility/dataset/prepare_datasets.py    # AIME ships here; fetch the other three
```

## Credentials

```bash
export OPENAI_API_KEY=... GEMINI_API_KEY=... DEEPSEEK_API_KEY=...
export SHADOW_API_KEY=...                     # the endpoint you audit
```

## Run

One command per endpoint, each collecting 3 runs of the whole benchmark (`--num-runs`, default 3, minimum 2). One run per side is refused because it would not fail loudly — it makes Model Equality Testing report a false `Reject`. Then compare.

```bash
python utility/run_eval.py --endpoint official --benchmark aime2025 --model GPT-5-Mini
python utility/run_eval.py --endpoint shadow   --benchmark aime2025 --model GPT-5-Mini \
    --base-url https://your-endpoint/v1 --model-id NAME-ON-YOUR-ENDPOINT --label ep1

python utility/aggregate_results_utility.py --shadow-label ep1
python fingerprint/met/run_met.py --shadow-label ep1
```

Same shape for `safety/main.py` + `safety/aggregate_results_safety.py`. `--list-models` prints what an endpoint offers, if you don't know `--model-id`.

## Disclaimer / Ethics

An auditing and transparency tool. It names, recommends and endorses **no** third-party service; you supply the endpoint and are responsible for complying with its terms and with those of the official providers. Sharing an `output/` or `Result_Safety/` tree identifies the service you audited — `.gitignore` keeps them out of version control, and the `meta` files store the endpoint host and model id as hashes for the same reason (`--record-endpoint-plaintext` opts out).

## Notes

`utility/config/` and `fingerprint/llmmap/config/` are deliberate byte-identical copies — the original had them duplicated and this release did not restructure it. **Edit one, edit the other.**

## Citation

Preprint: [arXiv:2603.01919](https://arxiv.org/abs/2603.01919).

```bibtex
@inproceedings{ZJCBSZ26,
author = {Yage Zhang and Yukun Jiang and Zeyuan Chen and Michael Backes and Xinyue Shen and Yang Zhang},
title = {{Real Money, Fake Models: Deceptive Model Claims in Shadow APIs}},
booktitle = {{ACM SIGSAC Conference on Computer and Communications Security (CCS)}},
publisher = {ACM},
year = {2026}
}
```

## Third-party material

MIT `LICENSE` covers this repository's own code. Redistributed under their own terms:

- **LLMmap** — MIT, (c) 2024 pasquini-dario, <https://github.com/pasquini-dario/LLMmap>.
  42 of the 60 vectors in `reference/templates/templates.json` are upstream's;
  `patches/` is a modified diff against commit `f661d55`.
- **AdvBench subset** (`safety/harmful_questions/`) — the 50-behaviour set from
  Chao et al., MIT, (c) 2023 PAIR Team, <https://github.com/patrickrchao/JailbreakingLLMs>;
  behaviours from AdvBench, MIT, (c) 2023 Andy Zou, <https://github.com/llm-attacks/llm-attacks>.
- **JailbreakBench behaviours** — embedded verbatim in `safety/GCG_Suffixes/JBB-*.json`;
  MIT, <https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors>.
- **LegalBench / SCALR** — `LEGALBENCH_PROMPT_TEMPLATE` is LegalBench's
  `tasks/scalr/base_prompt.txt` by Faiz Surani and Varun Iyer, CC BY 4.0
  (<https://creativecommons.org/licenses/by/4.0/>), modified only to use Python
  format fields, provided without warranties. <https://github.com/HazyResearch/legalbench>
- **AIME 2025** (`utility/dataset/aime_2025/`) — competition problems,
  (c) Mathematical Association of America; no public licence found.

Fetched at run time, not redistributed: GPQA-Diamond (CC BY 4.0, gated), MedQA, LegalBench (full), JailbreakBench, StrongREJECT templates (MIT).
