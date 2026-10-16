# Deviations from PROTOCOL.md

1. Translators replaced before any detector was run (2026-10-03). The protocol named two LLM translators
   (google/gemini-2.5-flash-lite, mistralai/mistral-small-3.2-24b-instruct). Automatic checks showed that they often
   answered or executed the prompt instead of translating it: 45% of attack prompts and 35% of benign prompts for the
   first model, 24% and 17% for the second, failed the length-ratio or script check; most of the inspected failures
   were not translations (see item 2), so the evaluation inputs are produced by sequence-to-sequence MT models instead:
   Helsinki-NLP/opus-mt-en-uk (`uk`), Helsinki-NLP/opus-mt-tc-big-en-zle (`uk_alt`) and Helsinki-NLP/opus-mt-uk-en
   (`pivot`, applied to `uk`), sentence by sentence with line breaks kept and code-like lines (indented,
   keyword-led or mostly symbols) left untranslated, beam size 2, output length bounded by twice the input length.
   NLLB-200 (600M), tried first as the second translator, was too slow on the shared CPU (over two hours).
   The LLM translations are kept in `data/llm_translation_pilot.jsonl` (`uaguard translate-llm`) and reported as a
   separate observation.

2. Analyses and code changes after the code was re-checked (2026-10-03); no detector was rerun.
   - Sensitivity analysis: the primary MT model sometimes produces Russian words, which the length and script checks
     do not catch (179 of 1,000 main prompts contain letters that exist only in Russian; the second translator: 1).
     `report` repeats the main comparison on prompts whose two Ukrainian versions contain none of these letters
     (`clean_uk`, 772 prompts; Russian words without these letters remain, for example «Ответь мне» in 74 of them)
     and the embedded comparison on pairs that pass a character screen (`embedded_clean`, 134 pairs: no Cyrillic in
     the English texts, Ukrainian-only letters in the Ukrainian payload and no Russian-only letters in the
     Ukrainian texts); one Gandalf payload is Russian and one German.
   - A random sample of 40 rejected LLM translations (seed 7) was classified by hand
     (`data/llm_translation_audit.csv`): `executed` – the output contains a result of the instruction, with or without
     a translation; `partial` – parts of the source left untranslated or dropped; `translation` – a complete
     translation in which only code, names or quoted titles stay in English. Result: 32 executed, 6 partial,
     2 complete translations rejected by the checks.
   - `scores.jsonl` rows carry a hash of the scored text, and `report` uses only scores whose hash matches the
     current input; MT cache files are keyed by model revision and beam size; model and dataset revisions are
     pinned; `translate` always starts from `data/items.jsonl`.

3. Reporting additions (2026-10-16); no detector was rerun and no earlier value changed.
   - `report` writes the FPR of the benign main prompts whose Ukrainian version was not truncated at 512 tokens, in
     English and in Ukrainian at the English 5% threshold (`untruncated_benign` in `summary.json`).
   - `audit_sample` in `report.py` draws the 40 rejected LLM translations of `data/llm_translation_audit.csv`
     (seed 7); a test checks that it reproduces the file.
   - Unused language-code arguments of the MT wrapper, left from the NLLB attempt, removed.

4. Texts removed from the repository (2026-10-16); no detector was rerun and no value changed.
   PromptShield's evaluation splits include LMSYS-Chat-1M conversations, whose licence does not allow
   redistribution, and the source of each prompt is not recorded, so no prompt text or translation is published.
   `data/items.jsonl`, `data/items_translated.jsonl`, `data/inputs.jsonl` and `data/llm_translation_pilot.jsonl` are
   rebuilt locally (`uaguard sample`, `uaguard translate`, `uaguard translate-llm`); `uaguard manifest` writes
   `data/manifest.json` from them, and `report` reads only the manifest and the scores. `results/summary.json` and
   `results/roc.png` are byte-identical with and without the texts, `uaguard sample` rebuilds `data/items.jsonl` with
   the hash in `protocol_freeze.sha256`, and a fresh `uaguard translate` without the MT cache reproduced every stored
   translation.
