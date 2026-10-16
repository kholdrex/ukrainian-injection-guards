# Third-party data

`data/items.jsonl`, `data/items_translated.jsonl`, `data/inputs.jsonl`, `data/llm_translation_pilot.jsonl` and
`data/llm_translation_audit.csv` contain texts from the datasets below and their machine translations into Ukrainian;
`results/scores.jsonl` contains the same texts with the detector scores.

- PromptShield (https://huggingface.co/datasets/hendzh/PromptShield), under the Apache License, Version 2.0
  (`LICENSES/Apache-2.0.txt`).
- Lakera/gandalf_ignore_instructions (https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions), MIT licence
  (`LICENSES/MIT-Gandalf.txt`).

PromptShield is compiled from other public datasets and published attack strategies (see its paper,
https://arxiv.org/abs/2501.15145); the terms of those sources also apply to the texts taken from them.

`data/llm_translation_pilot.jsonl` also contains outputs of google/gemini-2.5-flash-lite and
mistralai/mistral-small-3.2-24b-instruct, obtained through OpenRouter on 2026-10-03.

Content note: the data are prompt-injection attacks and benign prompts as published in these datasets; some of them
contain offensive language or quoted social-media posts. They are included for research on injection detection.

The MIT licence of this repository covers the code.
