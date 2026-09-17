import hashlib

# NOTE: this file is duplicated byte-for-byte at fingerprint/llmmap/config/models.py.
# The original repository had the two copies and this release did not restructure
# it, so any edit here must be mirrored there.
#
# The official model identifier for each audited model family. These are the ids
# published by OpenAI / Google / DeepSeek.
#
# A shadow endpoint exposes its own name for the same claimed model (one reseller
# may serve "GPT-5" where the official id is "gpt-5-2025-08-07"), so the shadow id
# is not tabulated here -- it is supplied per run with --model-id.
OFFICIAL_MODEL_IDS = {
    "GPT-4o-Mini": "gpt-4o-mini-2024-07-18",
    "GPT-5": "gpt-5-2025-08-07",
    "GPT-5-Mini": "gpt-5-mini-2025-08-07",
    "Gemini-2.0-Flash": "gemini-2.0-flash",
    "Gemini-2.5-Flash": "gemini-2.5-flash",
    "Gemini-2.5-Pro": "gemini-2.5-pro",
    "DeepSeek-V3.2-Chat": "deepseek-chat",
    "DeepSeek-V3.2-Reasoner": "deepseek-reasoner",
}

ENDPOINT_KINDS = ("official", "shadow")

# Model Equality Testing needs at least two completions per prompt per side: its
# statistic divides by sum_p c_p(c_p - 1), which is zero when every prompt was
# answered once. Three trials is the audit protocol; two is the hard floor.
NUM_TRIALS = 3
MIN_TRIALS = 2

# Sampling parameters the official endpoints reject outright (HTTP 400) rather
# than ignore. Shadow endpoints accepted both in the audit.
OFFICIAL_REJECTS_TEMPERATURE = ("GPT-5", "GPT-5-Mini")
OFFICIAL_REJECTS_SEED_PREFIX = "Gemini"

TEMPERATURE = 0.0
SEED = 42


def get_model_config(model, *, is_official, model_id_override=None):
    """Resolve (model_id, temperature, seed) for one endpoint.

    The parameters returned here are identical to those the audit sent: the
    original table was keyed on (model, channel) and special-cased
    channel == "Official"; this is the same decision expressed as is_official.
    """
    if model not in OFFICIAL_MODEL_IDS:
        raise ValueError(
            f"Unknown model {model!r}. Known: {sorted(OFFICIAL_MODEL_IDS)}"
        )

    if is_official:
        if model_id_override:
            # The official branch keys its parameters off the model name, and the
            # safety runner keys them off substrings of the model id. Allowing an
            # override here could send a parameter the official endpoint rejects.
            raise ValueError(
                "--model-id applies to --endpoint shadow only; the official "
                "endpoint always uses its published model id."
            )
        model_id = OFFICIAL_MODEL_IDS[model]
        temperature = None if model in OFFICIAL_REJECTS_TEMPERATURE else TEMPERATURE
        seed = None if model.startswith(OFFICIAL_REJECTS_SEED_PREFIX) else SEED
    else:
        if not model_id_override:
            raise ValueError(
                f"--endpoint shadow needs --model-id: the name {model!r} is served "
                f"under a different id on every endpoint. Run with --list-models to "
                f"see what your endpoint offers."
            )
        model_id = model_id_override
        temperature = TEMPERATURE
        seed = SEED

    return model_id, temperature, seed


def item_key(rendered_prompt):
    """Content key for one benchmark item: sha256 of the prompt as sent, 16 hex.

    Recorded alongside every response so the analysis scripts can verify that
    both endpoints were asked the same question. It is never used to score or to
    pair samples -- pairing stays on the positional `id`, as in the audit.

    safety/main.py carries an identical three-line copy (it does not import this
    package); the two must agree.
    """
    return hashlib.sha256(str(rendered_prompt).encode("utf-8")).hexdigest()[:16]


def dataset_fingerprint(records):
    """Fingerprint a benchmark file: content and row order, 16 hex.

    `records` is an iterable of field tuples, in file order -- every field that
    load_items() feeds into a prompt or a gold answer. Each record is hashed on
    its own (fields joined by \\x1f), then the hex digests are joined by newlines
    and hashed again.

    Row order is part of the fingerprint on purpose: GPQA shuffles its options
    with random.Random(SHUFFLE_SEED + idx), so a reordered file asks different
    questions even when the text is identical.
    """
    per_item = [
        hashlib.sha256(
            "\x1f".join(str(f).strip() for f in record).encode("utf-8")
        ).hexdigest()
        for record in records
    ]
    return hashlib.sha256("\n".join(per_item).encode("utf-8")).hexdigest()[:16]


# Fingerprints of the four benchmark files used in the audit. prepare_datasets.py
# checks a download against these; a mismatch is reported, never repaired, since
# re-ordering a file changes the prompts that get sent.
DATASET_FINGERPRINTS = {
    "aime2025": ("320b570ce49c7354", 30),
    "gpqa-diamond": ("173e6ee8c0835794", 198),
    "medqa": ("63bb9370af9ab284", 1273),
    "legalbench-scalr": ("a964fec812af1c1f", 571),
}
