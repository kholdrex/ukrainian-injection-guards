from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"

SEED = 2026
MAX_TOKENS = 200
CARRIER_MAX_TOKENS = 120
N_ATTACK = 400
N_BENIGN = 600
N_CARRIERS = 150

PROMPTSHIELD = "hendzh/PromptShield"
GANDALF = "Lakera/gandalf_ignore_instructions"
LENGTH_TOKENIZER = "protectai/deberta-v3-base-prompt-injection-v2"

MT_PRIMARY = ("Helsinki-NLP/opus-mt-en-uk", "")
MT_SECOND = ("Helsinki-NLP/opus-mt-tc-big-en-zle", ">>ukr<< ")
MT_BACK = ("Helsinki-NLP/opus-mt-uk-en", "")
LLM_TRANSLATORS = ("google/gemini-2.5-flash-lite", "mistralai/mistral-small-3.2-24b-instruct")

CLASSIFIERS = {
    "protectai": "protectai/deberta-v3-base-prompt-injection-v2",
    "mdeberta": "proventra/mdeberta-v3-base-prompt-injection",
    "testsavant": "testsavantai/prompt-injection-defender-base-v0",
}
LLM_GUARDS = {
    "qwen7b": "qwen2.5:7b-instruct",
    "gemma4b": "gemma3:4b",
}

TARGET_FPR = (0.01, 0.05)
BOOTSTRAP = 2000

REVISIONS = {
    "protectai/deberta-v3-base-prompt-injection-v2": "90c9989b1a342275dd0d1a95aad283c04e075671",
    "proventra/mdeberta-v3-base-prompt-injection": "b8a89d3096cf11a71d57b283c854f5ae2ed3df83",
    "testsavantai/prompt-injection-defender-base-v0": "eaa1423e6fc20104b2bbd7d38015351e10efd5d9",
    "Helsinki-NLP/opus-mt-en-uk": "634fdc4b28bf0c8a1d4359cbbcfb29190a09dc0e",
    "Helsinki-NLP/opus-mt-tc-big-en-zle": "708be1d372fe4c358a352f404e6dc9ca0126ba48",
    "Helsinki-NLP/opus-mt-uk-en": "1b9a2a0d1aa2a329d8476d2be5c7b0b27d5397b0",
    "hendzh/PromptShield": "a5234cb1f5cdb256600cab64b8c961195b5e8404",
    "Lakera/gandalf_ignore_instructions": "04737b65e90a6794ec227012e4a255a7def6344b",
}
