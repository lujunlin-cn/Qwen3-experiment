1: # nvidia-nemotron
2: 
3: Training and dataset-building pipeline for NVIDIA Nemotron reasoning experiments. The repo is organized as py:percent scripts in `src/*.py` with matching Kaggle notebooks in `notebooks/*.ipynb`; the notebooks are the GPU-server execution form of the scripts.
4: 
5: ## Published Artifacts
6: 
7: **Dataset**
8: - Kaggle: [`rohitraje0493/nemotron-reasoning`](https://www.kaggle.com/datasets/rohitraje0493/nemotron-reasoning)
9: - Hugging Face: [`the-submitter/nemotron-reasoning`](https://huggingface.co/datasets/the-submitter/nemotron-reasoning)
10: 
11: **SFT**
12: - Best noted run: `v7`, public `0.836`, private `0.852`
13: - Kaggle: [`nemotron-3-nano/transformers/lora-sft/7`](https://www.kaggle.com/models/rohitraje0493/nemotron-3-nano/Transformers/lora-sft/7)
14: - Hugging Face: [`the-submitter/nemotron-lora-sft-v2`](https://huggingface.co/the-submitter/nemotron-lora-sft-v2)
15: 
16: **DPO**
17: - Best noted run: `v9`, public `0.844`, private `0.856`
18: - Kaggle: [`nemotron-3-nano/transformers/lora-dpo/9`](https://www.kaggle.com/models/rohitraje0493/nemotron-3-nano/transformers/lora-dpo)
19: - Hugging Face: [`the-submitter/nemotron-lora-dpo-v7`](https://huggingface.co/the-submitter/nemotron-lora-dpo-v7)
20: 
175: ## [`src/02_update_dataset.py`](src/02_update_dataset.py) — Preference Dataset Update
176: 
177: **Corresponding Notebook**: [`notebooks/02_update_dataset.ipynb`](notebooks/02_update_dataset.ipynb)
178: 
179: Adds DPO preference fields to the base dataset by generating new trajectories for candidate prompts.
180: 
181: **Candidate selection**
182: - Loads the reasoning dataset and ensures preference columns exist.
183: - Selects rows where DPO processing is still pending.
184: - Prioritizes response-missing rows, then a configurable amount/range of existing-response rows.
185: - Supports source ordering and optional remaining-source pass-through with `*_ORDER_REMAINING`.
186: - Supports candidate-level high-quality filtering for missing and existing groups.
187: - Marks selected candidates with `dpo_selected` and updates `dpo_processed` incrementally.
188: 
189: **Generation**
190: - Builds vLLM offline inference with the base Nemotron model and optional LoRA adapter.
191: - Runs `TRAJECTORIES` generations per prompt in batches of `PROMPT_BATCH_SIZE`.
192: - Formats prompts through the tokenizer chat template.
193: - Supports debug CSV backups for generated trajectories and selected preferences.
194: 
195: **Preference selection**
196: - Extracts final answers with both competition-metric boxed extraction and balanced-brace boxed extraction.
197: - Falls back to textual/number answer extraction when boxed answers are absent.
198: - Verifies generated answers using numeric closeness, `math_verify`, and normalized string comparison.
255: - Filters rows with non-empty `prompt` and `final_answer`.
256: - Supports all source controls used in SFT/DPO.
257: - Adds `TRAIN_DPO_AWARE` / `EVAL_DPO_AWARE` ordering that prioritizes examples with useful DPO metadata.
258: - Adds `TRAIN_DPO_SORT_REMAINING` / `EVAL_DPO_SORT_REMAINING` so remaining-source rows preserve the original source mix while swapping indices within each source by DPO priority.
259: 
260: **Reward**
261: - Unified reward combines:
262:   - exact answer verification against `final_answer` using a combination of binary full match, `math.isclose()` with relative tolerance of 1e-2, and `math_verify`.
263:   - fuzzy matching of extracted answers
264:   - fuzzy matching (token set ratio) of full completion against known `reasoning + response`
265:   - boxed-answer format reward
266:   - non-empty `<think>...</think>` reward
267: - Answer extraction uses competition boxed extraction, balanced-brace boxed extraction, and fallback extraction.
268: - Handles completions that contain `</think>` without a leading `<think>` by prepending the expected opening tag.
269: - Sets `importance_sampling_level="sequence"` to enable GSPO-style sequence-level importance sampling.
270: 
271: **Status**
272: - Kept as an experimental Unsloth path.
273: - Notes from Kaggle: vLLM/Unsloth state-dict synchronization issues were encountered with the current Nemotron hybrid model.
