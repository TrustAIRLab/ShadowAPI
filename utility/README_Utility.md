# Utility / Performance Audit

Evaluation runner and comparison for four benchmarks: AIME 2025, GPQA-Diamond, MedQA, LegalBench-SCALR.

## Structure

```
├── config/                           # endpoint and model configuration
├── prompts/                          # per-benchmark prompt templates
├── dataset/                          # benchmark question files (+ prepare_datasets.py)
├── run_eval.py                       # response collection
├── aggregate_results_utility.py      # aggregation and official-vs-shadow comparison
└── output/                           # your runs: <benchmark>/<model>/<label>/
```

## Collect

```bash
python utility/run_eval.py --endpoint official --benchmark aime2025 --model GPT-5-Mini
python utility/run_eval.py --endpoint shadow   --benchmark aime2025 --model GPT-5-Mini \
    --base-url https://your-endpoint/v1 --model-id NAME-ON-YOUR-ENDPOINT --label ep1
```

`--benchmark`: `aime2025`, `gpqa-diamond`, `medqa`, `legalbench-scalr`

`--model`: `GPT-4o-Mini`, `GPT-5`, `GPT-5-Mini`, `Gemini-2.0-Flash`, `Gemini-2.5-Flash`, `Gemini-2.5-Pro`, `DeepSeek-V3.2-Chat`, `DeepSeek-V3.2-Reasoner`

`--label`: names the output directory. `[A-Za-z0-9-]+`; defaults to `Official` for `--endpoint official`, required for shadow.

Also: `--limit N`, `--num-runs` (default 3, min 2), `--retries N` (default 0), `--pause`, `--telemetry`, `--list-models`, `--tag`, `--overwrite`.

Sampling parameters default to what the audit sent (`temperature=0`, `seed=42`; omitted where the official endpoint rejects them). Override with `--temperature X` / `--no-temperature` / `--seed N` / `--no-seed`. **You will need `--no-seed` if your endpoint rejects the parameter** — some answer every request with HTTP 400 otherwise. An override is recorded in `meta.json` and the analyser warns when the two sides were collected differently.

## Compare

```bash
python utility/aggregate_results_utility.py --shadow-label ep1 [--csv results/u.csv]
```

Prints mean ± SD per endpoint, and — with `--shadow-label` — the differences. Before comparing it checks that both sides were asked the same questions (via the recorded `item_key`) and that their `meta.json` agree. The check reports `unverified` — not a match — when a run carries no `item_key` or the two sides cover different question ids; the accuracies printed are always each side's own full set, never a recomputation over the shared ids. `--benchmark` / `--model` filter the output. `--input-root DIR` reads a results tree kept outside the repository: pass whatever you gave `run_eval.py --output-root`.

## Columns beyond accuracy

`accuracy` is computed exactly as in the paper. Three others exist because accuracy hides two different failures:

- `unparsed_rate` — no answer could be extracted. A response truncated before it
  states an answer scores as *wrong*, so a gap here means part of an accuracy gap
  is formatting, not capability.
- `accuracy_if_parsed` — accuracy over rows that did produce an answer.
- `response_chars_median` — because AIME's extractor falls back to the last
  number in the text, a truncated AIME response yields a *wrong* answer rather
  than none, and `unparsed_rate` stays low while something is clearly wrong.

## Unchanged from the audit

Answer extractors, `score_response`, `shuffled_options` (`SHUFFLE_SEED = 42`), `load_items` and the four prompt templates are byte-identical, and should stay that way — they define the measurement. `item_key` and `api_error` are recorded but never scored. medqa / legalbench write no telemetry columns unless `--telemetry`. SD uses `ddof=0`.

`--retries` defaults to 0 because retrying removes rows the audit counted. Note the bias is opposite across modules: a dropped call here scores as a wrong answer, while in `safety/` an empty answer scores as a refusal.
