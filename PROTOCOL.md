# Protocol

Fixed before any detector was run on the evaluation inputs. Deviations are listed in CHANGELOG.md.

## Question

Do prompt-injection detectors whose decision threshold is set on English benign traffic keep their recall and
false-positive rate (FPR) when the same inputs arrive in Ukrainian, and does translating Ukrainian input back to
English before detection restore them?

## Data

- `data/items.jsonl`, built by `uaguard sample` with seed 2026 from public datasets:
  - calibration: all benign PromptShield validation prompts of at most 200 tokens (449 after de-duplication);
  - main: 400 attacks and 600 benign prompts from the PromptShield test split, at most 200 tokens;
  - embedded: 150 further benign test prompts of at most 120 tokens (carriers) paired with 150 injection payloads
    from Lakera/gandalf_ignore_instructions (validation and test splits).
- Token counts use the ProtectAI DeBERTa tokenizer; the limit keeps Ukrainian versions within 512 tokens.

## Conditions

- `en`: original text. `uk`: translation by google/gemini-2.5-flash-lite. `uk_alt`: independent translation by
  mistralai/mistral-small-3.2-24b-instruct. `pivot`: `uk` translated back to English by the second translator.
- Embedded: carrier alone in English or Ukrainian (benign), carrier followed by a payload, in all four combinations
  of carrier language and payload language (attack).

## Detectors

- Classifiers: protectai/deberta-v3-base-prompt-injection-v2 (English), proventra/mdeberta-v3-base-prompt-injection
  (multilingual base model), testsavantai/prompt-injection-defender-base-v0.
- Local LLM guards via Ollama: qwen2.5:7b-instruct and gemma3:4b with one fixed instruction; score =
  P(YES)/(P(YES)+P(NO)) of the first generated token.

## Operating points

Thresholds are set on the English calibration prompts at 5% FPR (primary) and 1% FPR, and are then frozen. The
default 0.5 threshold is reported for reference. A Ukrainian-calibrated threshold (same procedure on the Ukrainian
calibration prompts) is secondary.

## Endpoints

- Primary: on the main set, change in recall and in FPR from `en` to `uk` at the 5% English threshold, per detector.
  A recall loss of 10 percentage points or more is treated as practically important.
- Secondary: AUROC by condition; `uk_alt` as translator sensitivity; recovery by `pivot`; embedded 2x2 recall at the
  frozen threshold; effect of Ukrainian calibration; per-input latency.

## Statistics

Paired percentile bootstrap over source items (2000 resamples, all conditions of one item resampled together) for
differences; exact McNemar test for the paired en-vs-uk decisions on attacks (recall) and on benign prompts (FPR),
Holm correction across detectors for the two primary contrasts. Wilson intervals for single proportions.

## Exclusions

Main items whose `uk`, `uk_alt` or `pivot` text fails an automatic check (length ratio outside 0.5–2.5, wrong
script) are excluded from the primary analysis; the analysis is repeated on all items.
