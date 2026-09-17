# Benchmark files

Only AIME 2025 ships with this repository. The other three come with their own terms, so you fetch them yourself.

```bash
python utility/dataset/prepare_datasets.py
```

That checks what is present and prints where to get what is missing. It verifies content *and row order* against the files the audit used, and never modifies anything.

| Benchmark | Path | Records | Source |
|---|---|---|---|
| AIME 2025 | `aime_2025/aime2025.jsonl` | 30 | ships here |
| GPQA-Diamond | `gpqa/gpqa_diamond.csv` | 198 | https://huggingface.co/datasets/Idavidrein/gpqa (gated) |
| MedQA (USMLE) | `MedQA/test_us.jsonl` | 1273 | https://github.com/jind11/MedQA, US test split |
| LegalBench-SCALR | `LegalBench/scalr/test.tsv` | 571 | https://huggingface.co/datasets/nguha/legalbench, `scalr` test split |

Expected formats:

- `aime2025.jsonl` — JSON lines with `question`, `answer`
- `gpqa_diamond.csv` — columns `Question`, `Correct Answer`, `Incorrect Answer 1`, `Incorrect Answer 2`, `Incorrect Answer 3`
- `test_us.jsonl` — JSON lines with `question`, `options` (a dict), `answer_idx`
- `test.tsv` — columns `question`, `choice_0` … `choice_4`, `answer`

## Why row order is checked, not just content

GPQA's answer options are shuffled per item with `random.Random(SHUFFLE_SEED + row_index)`. The row's *position* therefore decides which letter is correct and in what order the options appear. Two downloads with the same 198 questions in a different order ask genuinely different questions, and comparing an official run against a shadow run collected from differently ordered files would show a difference that is nothing to do with the endpoint.

So `prepare_datasets.py` reports a mismatch and stops rather than sorting the file into line. Sorting would silently change the prompts. If your file does not match, you can still use it — just collect **both** endpoints from that same file, and expect your absolute numbers to differ from the paper's.

## A note on GPQA

The released `gpqa_diamond.csv` contains a `Canary String` column and columns naming the question writer and the expert and non-expert validators. It is distributed behind a gate for that reason. Keep your copy out of any public repository; `.gitignore` here already excludes it.

## Differences from the audit's recorded files

The CSVs this repository writes carry two columns the audit's did not: `item_key` (a hash of the prompt as sent, so the analysis can confirm both endpoints were asked the same question) and `api_error`. Neither is scored. `medqa` and `legalbench-scalr` still record no latency or token columns by default, as in the audit; `--telemetry` adds them.
