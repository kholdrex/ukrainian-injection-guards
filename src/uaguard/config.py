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

TRANSLATOR = "google/gemini-2.5-flash-lite"
SECOND_TRANSLATOR = "mistralai/mistral-small-3.2-24b-instruct"

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
