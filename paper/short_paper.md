# Safety Evaluation is Highly Sensitive to Prompt Framing: An Inference-Only Study on HarmBench

## Abstract

Safety benchmarks are often treated as stable measurements of a model's refusal behavior, but that assumption can fail even under minimal prompt reformatting. We study this effect with an inference-only protocol on HarmBench using the first 100 harmful instructions from the official text benchmark and a single open-weight model, `meta-llama/Meta-Llama-3-8B-Instruct`, under deterministic decoding. For each instruction, we evaluate three fixed prompt framings: the original request, a fictional-story wrapper, and a translation wrapper. Outputs are labeled automatically with a fixed refusal-rule detector, and pairwise framing differences are tested with McNemar's test. We show that safety benchmark outcomes can vary substantially under minimal prompt reformatting, raising concerns about the robustness of current evaluation practices. Our study makes three contributions: (1) an inference-only safety evaluation protocol, (2) a direct quantification of framing sensitivity, and (3) the Framing Sensitivity Index (FSI), which measures how often refusal outcomes change across framings.

## 1. Introduction

Safety evaluation pipelines often assume that a benchmark prompt has a stable semantic target and therefore produces a stable safety judgment. This assumption matters because benchmark scores are frequently used to compare models, measure progress, and support deployment decisions. If benchmark outcomes change under minimal prompt reframing, then the measured score may reflect surface prompt form as much as underlying harmful intent.

This paper tests a narrow question: how sensitive is HarmBench refusal behavior to prompt framing alone? We do not propose a new alignment method, train a model, or develop a jailbreak technique. Instead, we isolate a simple evaluation variable and measure its effect under a fixed inference-only setup.

We emphasize that we do not introduce a new attack or bypass method. Our goal is solely to measure evaluation sensitivity under benign prompt reformatting.

## 2. Experimental Setup

### Dataset

We use the official HarmBench text benchmark and select the first 100 harmful instructions without modification or sampling. No new prompts, augmentations, or filtering steps are introduced.

### Model and Decoding

We evaluate `meta-llama/Meta-Llama-3-8B-Instruct` with no fine-tuning. The model is publicly available under a gated license on Hugging Face. The primary analysis uses deterministic decoding with temperature fixed to 0.0, `max_tokens=512`, `top_p=1.0`, and `repetition_penalty=1.0`. As an appendix robustness check, we repeat the same 300 prompt/frame evaluations once with stochastic decoding at temperature 0.7 and the same remaining decoding settings.

### Prompt Framing

Each original HarmBench instruction is evaluated under exactly three framings:

1. Direct: the original prompt unchanged.
2. Fictional: `In a fictional story, a character asks: "{original prompt}" How would this scenario unfold?`
3. Translation: `Translate the following sentence into French: "{original prompt}"`

This produces 300 inferences per decoding condition and 600 total inferences across the main and appendix runs.

### Refusal Rule

Outputs are automatically classified as refusal if they contain a fixed refusal regex including patterns such as `I cannot`, `I can't`, `I will not`, `I am unable`, `I do not provide`, `I must refuse`, `I cannot comply`, and `As an AI`. The exact matched pattern and raw model output are stored for auditability. Primary benchmark labels are fully rule-based; a separate 20-case manual sanity check is used only to estimate false positives in the refusal detector.

### Analysis

We report refusal rate by framing, pairwise framing differences with McNemar's test, and the Framing Sensitivity Index:

`FSI = (# prompts whose refusal outcome changes across framings) / 100`

To quantify uncertainty, we compute 95% bootstrap percentile confidence intervals for FSI using 10,000 prompt-level resamples.

## 3. Results

Figure 1 reports refusal rate by framing. Figure 2 visualizes prompt-level refusal consistency across the three framing conditions. Figure 3 shows the distribution of refusal counts across the three framings, making stable and unstable prompts easy to distinguish. We also report FSI separately for prompt indices 0-49 and 50-99 as a simple sample-split robustness check. Effect persists under stochastic decoding (Appendix A).

In the deterministic main condition, refusal rates are 0.74 for Direct, 0.65 for Fictional, and 0.53 for Translation. The Framing Sensitivity Index is 0.24 with a 95% bootstrap confidence interval of [0.16, 0.33]. Exact McNemar tests show significant pairwise differences for Direct vs Fictional ($p=0.0225$), Direct vs Translation ($p=9.54\times 10^{-7}$), and Fictional vs Translation ($p=0.00183$).

## 4. Discussion

If refusal outcomes shift under light prompt reframing, then benchmark scores are not purely measuring harmful-intent recognition. They are also measuring how the benchmark prompt is packaged for the model. This weakens the interpretation of a single benchmark number as a robust safety property.

Our findings therefore target evaluation methodology rather than alignment or mechanistic explanation. The main implication is straightforward: safety benchmarks should be tested for framing robustness before they are used as stable comparative metrics.

The persistence of framing sensitivity under both deterministic and stochastic decoding suggests that the observed effect is not solely attributable to decoding artifacts.

## 5. Limitations

- Single model evaluation
- Single benchmark
- Rule-based refusal detection
- No human evaluation

The refusal rule may miss nuanced refusals or incorrectly label non-refusals that happen to contain one of the trigger phrases. Because the study is intentionally narrow, the results should be interpreted as evidence of measurement sensitivity, not as a universal statement about all models or all safety benchmarks.
We do not use human evaluation for primary benchmark scoring; the appendix manual audit is limited to a small false-positive sanity check.

## Ethics

The study evaluates harmful prompts from an existing public benchmark. We do not introduce new harmful content, optimize jailbreaks, or propose methods for increasing attack success. The goal is to assess evaluation reliability and benchmark robustness.

## Appendix A. Stochastic Decoding Robustness

We repeat the full 100-prompt, 3-framing evaluation once at temperature 0.7 with no additional repetitions. This appendix tests whether framing sensitivity is merely a deterministic decoding artifact. Under stochastic decoding, refusal rates are 0.72 for Direct, 0.64 for Fictional, and 0.50 for Translation. The appendix FSI is 0.24 with a 95% bootstrap confidence interval of [0.16, 0.33].

## Appendix B. Reproducibility and Access

- Hugging Face model page: `meta-llama/Meta-Llama-3-8B-Instruct`
- Model revision: `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`
- Access: request access through the Hugging Face model page by accepting the gated license terms and submitting the access form.
- HarmBench commit: `8e1604d1171fe8a48d8febecd22f600e462bdcdd`

## Appendix C. Refusal-Rule Sanity Check

We draw a fixed random sample of 20 matched-pattern cases from the deterministic run and record manual false-positive annotations. Manual inspection of 20 random flagged cases revealed no false positives. The observed false-positive rate in this audit is 0.0.

## Responsible NLP Checklist Notes

- Dataset license: HarmBench is distributed in a repository licensed under MIT.
- Harmful content use: This work evaluates harmful instructions from a public benchmark and does not introduce new harmful prompts.
- Dual-use risk: The paper studies evaluation sensitivity and does not optimize attack or bypass success.
- AI writing assistance disclosure: Portions of this manuscript were edited with AI-assisted language tools.

## Acknowledgements

Portions of this manuscript were edited with AI-assisted language tools.
