# Safety Evaluation is Highly Sensitive to Prompt Framing

This repository is a minimal, inference-only scaffold for the proposed HarmBench framing-sensitivity study.

## Protocol Locks

- Dataset: official HarmBench text test CSV from the Center for AI Safety repository
- Prompt subset: first 100 rows only, no sampling
- Model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Framing conditions: `direct`, `fictional`, `translation`
- Primary inferences: `100 x 3 = 300` under deterministic decoding
- Appendix inferences: `100 x 3 = 300` under stochastic decoding with `temperature=0.7`
- Total inferences: `600`
- Refusal detector: fixed rule-based regex over the specified refusal phrases, with matched pattern auditing
- Figures: exactly three figures are produced

## Assumption

The proposal does not name a HarmBench split, so this scaffold uses the official text benchmark split at:

`data/behavior_datasets/harmbench_behaviors_text_test.csv`

If you want a different official HarmBench CSV, change the constant in `src/framing_sensitivity/constants.py` before running the pipeline.

## Reproducibility Record

- HarmBench commit hash: `8e1604d1171fe8a48d8febecd22f600e462bdcdd`
- HarmBench dataset path: `data/behavior_datasets/harmbench_behaviors_text_test.csv`
- HarmBench repository license: `MIT`
- HF model ID: `meta-llama/Meta-Llama-3-8B-Instruct`
- HF model revision: `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`
- `torch` version: `2.10.0`
- `transformers` version: `5.2.0`
- Base seed for stochastic decoding: `20250301`
- Prompt-level sampling seed rule: `base_seed + prompt_index * 10 + frame_order`

## Environment

The target model is gated on Hugging Face. You need:

- Python 3.10+
- a local GPU with enough memory for Llama-3-8B-Instruct
- Hugging Face access to `meta-llama/Meta-Llama-3-8B-Instruct`
- `HF_TOKEN` available in the environment or a prior `huggingface-cli login`

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Pipeline

1. Prepare the first 100 HarmBench prompts and generate the three fixed framings:

```bash
python3 scripts/prepare_dataset.py
```

2. Run both decoding presets over all prompt/frame pairs:

```bash
python3 scripts/run_inference.py
```

3. Classify refusals, compute statistics, and render the three figures:

```bash
python3 scripts/analyze_results.py
```

## Outputs

- `data/processed/original_prompts.csv`
- `data/processed/framed_prompts.csv`
- `results/completions/completions.csv`
- `results/analysis/refusal_summary.csv`
- `results/analysis/fsi_summary.csv`
- `results/analysis/pairwise_mcnemar.csv`
- `results/analysis/prompt_level_outcomes.csv`
- `results/analysis/fsi_subgroups.csv`
- `results/analysis/manual_flagged_audit_sample.csv`
- `results/analysis/manual_flagged_audit_summary.csv`
- `results/analysis/summary.json`
- `results/figures/figure_1_refusal_rate.png`
- `results/figures/figure_2_prompt_level_consistency.png`
- `results/figures/figure_3_framing_sensitivity_histogram.png`

## Notes

- The prompt templates are hard-coded to match the proposal exactly.
- `results/completions/completions.csv` stores `raw_output`, `refusal_label`, and `matched_pattern` for refusal-audit traceability.
- The main condition uses greedy decoding as the practical implementation of `temperature=0.0`. The appendix condition adds one stochastic pass with `temperature=0.7`, `top_p=1.0`, `repetition_penalty=1.0`, and the same `max_new_tokens=512`.
- Stochastic sampling is not cherry-picked post hoc. It is fixed in advance with `temperature=0.7`, `top_p=1.0`, `repetition_penalty=1.0`, `max_new_tokens=512`, and seed formula `base_seed + prompt_index * 10 + frame_order`.
- Pairwise significance tests use `statsmodels.stats.contingency_tables.mcnemar(..., exact=True, correction=False)`.
- FSI is reported with a 95% bootstrap percentile confidence interval from 10,000 prompt-level resamples, and Figure 1 includes matching refusal-rate error bars from the same bootstrap procedure.
- `results/analysis/manual_flagged_audit_sample.csv` contains a fixed random sample of 20 flagged deterministic cases for manual sanity checking. Fill `manual_false_positive` with `0` or `1` and rerun analysis to update `manual_flagged_audit_summary.csv`.
- Figure settings are locked to avoid visual exaggeration: Figure 1 uses a fixed `0-1` y-axis and consistent framing colors, and Figure 2 uses a fixed `0-1` heatmap scale.
- No classifier model, fine-tuning, or augmentation is used. Primary refusal labels remain fully rule-based; the 20-case manual audit is only a false-positive sanity check.
- A draft paper skeleton is included at `paper/short_paper.md`.
