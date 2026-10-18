# Prompt-injection detectors on Ukrainian inputs

Code, detector scores and analysis for the paper *Transfer of English-calibrated prompt-injection
detectors to Ukrainian inputs* (O. Kholodniak).

Archived on Zenodo: [doi:10.5281/zenodo.23444970](https://doi.org/10.5281/zenodo.23444970).

PromptShield prompts (400 injections, 600 benign, plus 449 benign calibration prompts) and Gandalf injection payloads
embedded in benign carrier texts are translated into Ukrainian with sequence-to-sequence MT and scored by five
detectors. Thresholds are set on English benign calibration prompts (1% and 5% FPR) and frozen; recall, FPR and AUROC
are compared between the English and Ukrainian versions of the same prompts, for a second translator, for
translation back into English before detection, and for injections embedded in English or Ukrainian text.

| Detector | Model |
|---|---|
| `protectai` | protectai/deberta-v3-base-prompt-injection-v2 |
| `mdeberta` | proventra/mdeberta-v3-base-prompt-injection |
| `testsavant` | testsavantai/prompt-injection-defender-base-v0 |
| `qwen7b`, `gemma4b` | qwen2.5:7b-instruct and gemma3:4b in Ollama as guards; score P(YES) / (P(YES) + P(NO)) of the first token |

Translation: Helsinki-NLP/opus-mt-en-uk (primary), Helsinki-NLP/opus-mt-tc-big-en-zle (second translator),
Helsinki-NLP/opus-mt-uk-en (back-translation); sentence by sentence, code lines kept. An earlier attempt with LLM
translators is recorded in the manifest: they often executed the prompt instead of translating it (see
`CHANGELOG.md`).

## Data and models

- [PromptShield](https://huggingface.co/datasets/hendzh/PromptShield) and
  [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions).
  PromptShield is compiled from other datasets; its evaluation splits include conversations from LMSYS-Chat-1M, whose
  licence does not allow redistribution, and the source of each prompt is not recorded. The repository therefore
  contains no prompt texts and no translations of them. `data/manifest.json` keeps what the analysis needs: the
  identifier, label and group of every item, the automatic checks of each translation, the letter classes of every
  text used by the character screens and the hash of every scored input. The texts are rebuilt locally from the
  pinned dataset revisions (see Usage).
- Classifiers and MT models from the Hugging Face Hub (downloaded on first use); local guards in Ollama 0.34.3 on an
  Apple M1 Pro (`qwen2.5:7b-instruct` digest 845dbda0ea48, `gemma3:4b` a2af6cc3eb7f, Q4_K_M).

## Layout

```
src/uaguard/    sampling, translation, evaluation inputs, detectors, metrics, latency, report
data/           manifest.json (text-free record of items, inputs and LLM translations),
                llm_translation_audit.csv (40 rejected LLM translations classified by hand)
results/        scores.jsonl (score, tokens and time per detector and input), summary.json, latency.json, roc.png
tests/          unit tests
PROTOCOL.md     plan written before any detector was run
CHANGELOG.md    changes after the plan
```

`protocol_freeze.sha256` holds the hashes of the protocol, the sample (`data/items.jsonl`) and the code at the commit
tagged `protocol-freeze` (2026-10-03 01:31 UTC), before any detector was run. Later changes to these files are listed
in `CHANGELOG.md`, so the check is run on that commit, with the sample rebuilt by `uaguard sample`:

```bash
uv run --no-sync uaguard sample
mkdir /tmp/uaguard-freeze && git archive protocol-freeze | tar -x -C /tmp/uaguard-freeze
mkdir -p /tmp/uaguard-freeze/data && cp data/items.jsonl /tmp/uaguard-freeze/data/
cd /tmp/uaguard-freeze && grep -v '^[0-9-]*T' protocol_freeze.sha256 | shasum -a 256 -c
```

## Usage

Python 3.11+ and [uv](https://docs.astral.sh/uv/). `requirements.lock.txt` lists the exact package versions used.

```bash
uv venv && uv pip install -r requirements.lock.txt && uv pip install -e ".[dev]" --no-deps
uv run --no-sync pytest
uv run --no-sync uaguard report   # results/summary.json and results/roc.png from the manifest and the stored scores
```

`report` needs no texts. To rebuild them, `uv run --no-sync uaguard sample` downloads the pinned dataset revisions and
writes `data/items.jsonl`, and `uv run --no-sync uaguard translate` writes `data/items_translated.jsonl`,
`data/inputs.jsonl` (every scored text) and the manifest; a fresh run on the CPU of an Apple M1 Pro (40 minutes)
reproduced every stored text.

New measurements (macOS with Apple MPS: the classifiers and the latency measurement use the `mps` device):
`uv run --no-sync uaguard score protectai mdeberta testsavant qwen7b gemma4b` and `uv run --no-sync uaguard latency` on
an otherwise idle machine. The guards need Ollama with the model digests listed above (`ollama list`); the scores
identify a guard by its tag, so resumed scoring with other weights under the same tag would mix them.
`uv run --no-sync uaguard translate-llm` reproduces the LLM translation attempt (needs `OPENROUTER_API_KEY`). Model and
dataset revisions are pinned in `src/uaguard/config.py`. Scoring resumes from `results/`; `report` stops if a detector
has missing or stale scores.

## License

MIT. The repository contains no third-party texts; the datasets keep their own licences.
