import os

from openai import OpenAI

# Official endpoints, selected by the model family prefix.
OFFICIAL_BY_FAMILY_PREFIX = {
    "GPT": {
        "base_url": "https://api.openai.com/v1",
        "env": "OPENAI_API_KEY",
    },
    "Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env": "GEMINI_API_KEY",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "env": "DEEPSEEK_API_KEY",
    },
}

# A shadow endpoint is described entirely by the caller: --base-url and the name
# of the environment variable holding its key. Nothing about any specific
# reseller is recorded in this repository.
DEFAULT_SHADOW_KEY_ENV = "SHADOW_API_KEY"


class MissingCredential(RuntimeError):
    pass


def _require_key(env_var: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise MissingCredential(
            f"Environment variable {env_var} is not set. "
            f"Export it before running (see the Credentials section in README.md)."
        )
    return key


def resolve_endpoint(model_family: str, endpoint: str, base_url=None, api_key_env=None):
    """Return (base_url, env_var) for one endpoint.

    endpoint == "official": both come from the model family.
    endpoint == "shadow":   both come from the caller.
    """
    if endpoint == "official":
        for prefix, spec in OFFICIAL_BY_FAMILY_PREFIX.items():
            if model_family.startswith(prefix):
                return spec["base_url"], spec["env"]
        raise ValueError(
            f"No official endpoint registered for model_family={model_family!r}. "
            f"Known prefixes: {sorted(OFFICIAL_BY_FAMILY_PREFIX)}"
        )

    if endpoint == "shadow":
        if not base_url:
            raise ValueError(
                "--endpoint shadow needs --base-url (the OpenAI-compatible URL of "
                "the endpoint you want to audit, e.g. https://example.test/v1)."
            )
        return base_url, api_key_env or DEFAULT_SHADOW_KEY_ENV

    raise ValueError(f"Unknown endpoint {endpoint!r}. Known: official, shadow")


def build_client(model_family: str, endpoint: str, base_url=None, api_key_env=None) -> OpenAI:
    url, env_var = resolve_endpoint(model_family, endpoint, base_url, api_key_env)
    return OpenAI(api_key=_require_key(env_var), base_url=url)


def list_models(base_url, api_key_env=None):
    """Model ids an OpenAI-compatible endpoint advertises.

    Answers "what do I pass to --model-id?", which nothing else can: a reseller's
    catalogue names are its own.
    """
    client = OpenAI(
        api_key=_require_key(api_key_env or DEFAULT_SHADOW_KEY_ENV),
        base_url=base_url,
    )
    return sorted(m.id for m in client.models.list().data)
