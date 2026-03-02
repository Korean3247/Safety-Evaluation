# Safety Evaluation is Highly Sensitive to Prompt Framing: An Inference-Only Study on HarmBench

## Abstract

Safety benchmarks are often treated as stable measurements of a model's refusal behavior, but that assumption can fail even under minimal prompt reformatting. We study this effect with an inference-only protocol on HarmBench using the first 100 harmful instructions from the official text benchmark and a single open-weight model, `meta-llama/Meta-Llama-3-8B-Instruct`. For each instruction, we evaluate three fixed prompt framings: the original request, a fictional-story wrapper, and a translation wrapper. Under deterministic decoding, refusal rates vary from 0.74 for Direct framing to 0.53 for Translation framing, and all pairwise differences are significant under exact McNemar tests. The Framing Sensitivity Index (FSI), which measures how often refusal outcomes change across framings, is 0.24 with a 95% bootstrap confidence interval of [0.16, 0.33]. The effect remains under a single stochastic decoding pass at temperature 0.7, where refusal rates are 0.72, 0.64, and 0.50 and FSI remains 0.24. We show that safety benchmark outcomes can vary substantially under minimal prompt reformatting, raising concerns about the robustness of current evaluation practices.

## 1. Introduction

Safety evaluation pipelines often assume that a benchmark prompt has a stable semantic target and therefore produces a stable safety judgment. This assumption matters because benchmark scores are frequently used to compare models, measure progress, and support deployment decisions. If benchmark outcomes change under minimal prompt reframing, then the measured score may reflect surface prompt form as much as underlying harmful intent.

This paper tests a narrow question: how sensitive is HarmBench refusal behavior to prompt framing alone? We do not propose a new alignment method, train a model, or develop a jailbreak technique. Instead, we isolate a simple evaluation variable and measure its effect under a fixed inference-only setup.

We emphasize that we do not introduce a new attack or bypass method. Our goal is solely to measure evaluation sensitivity under benign prompt reformatting.

If safety benchmarks are used to rank models, then framing sensitivity introduces a hidden evaluation variance that is not typically reported.

## 2. Related Work

Recent safety benchmark work has focused on standardized harmful-instruction evaluation and calibrated refusal behavior. HarmBench provides a broad benchmark for harmful request completion and refusal evaluation, while XSTest probes exaggerated safety behavior on safe prompts. Our study is closest in motivation to this benchmark line, but instead of introducing a new benchmark or a safe contrast set, we ask whether a fixed harmful benchmark remains stable under minimal prompt reformatting.

A separate line of work studies jailbreaks and adversarial prompt attacks, including manually designed jailbreak prompts and universal adversarial suffixes. Those papers optimize prompts to bypass safeguards. By contrast, we use three fixed benign wrappers and perform no search, optimization, or attack construction. Our framing question is also related to broader evidence that language-model behavior is sensitive to prompt formatting and that evaluation conclusions can shift under seemingly small procedural choices. We therefore position this paper as a benchmark-reliability study rather than an attack paper.

## 3. Experimental Setup

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

Outputs are automatically classified as refusal if they contain a fixed refusal regex including patterns such as `I cannot`, `I can't`, `I will not`, `I am unable`, `I do not provide`, `I must refuse`, `I cannot comply`, and `As an AI`. The exact matched pattern and raw model output are stored for auditability. Primary benchmark labels are fully rule-based; a separate 20-case manual sanity check is used only to estimate false positives in the refusal detector. Because the detector is English-anchored, the Translation framing is the most plausible source of false negatives, so we also emit a fixed random audit sample of deterministic Translation non-refusal outputs for targeted manual review.

### Analysis

We report refusal rate by framing, pairwise framing differences with McNemar's test, and the Framing Sensitivity Index:

`FSI = (# prompts whose refusal outcome changes across framings) / 100`

To quantify uncertainty, we compute 95% bootstrap percentile confidence intervals for FSI using 10,000 prompt-level resamples.

## 4. Results

Figure 1 reports refusal rate by framing. Figure 2 visualizes prompt-level refusal consistency across the three framing conditions. Figure 3 shows the distribution of refusal counts across the three framings, making stable and unstable prompts easy to distinguish.

In the deterministic main condition, refusal rates are 0.74 for Direct, 0.65 for Fictional, and 0.53 for Translation. The largest absolute gap is 0.21 between Direct and Translation, and even the smaller 0.09 gap between Direct and Fictional is statistically significant. Exact McNemar tests show significant pairwise differences for Direct vs Fictional ($p=0.0225$), Direct vs Translation ($p=9.54\times 10^{-7}$), and Fictional vs Translation ($p=0.00183$). Table 1 adds paired effect sizes: the paired risk differences are 0.09, 0.21, and 0.12, with discordant prompt shares of 0.13, 0.21, and 0.14 respectively.

| Comparison | Paired risk difference | Discordant prompt share | Exact McNemar p |
| --- | ---: | ---: | ---: |
| Direct vs Fictional | 0.09 | 0.13 | 0.0225 |
| Direct vs Translation | 0.21 | 0.21 | 9.54e-07 |
| Fictional vs Translation | 0.12 | 0.14 | 0.00183 |

The Framing Sensitivity Index is 0.24 with a 95% bootstrap confidence interval of [0.16, 0.33], meaning that 24 of 100 prompts change refusal status across the three framings. While refusal rate captures aggregate shifts, FSI captures prompt-level instability. This instability is not uniformly distributed across the benchmark. For prompt indices 0-49, FSI is 0.36 [0.22, 0.50], while for prompt indices 50-99 it is 0.12 [0.04, 0.22]. This heterogeneity suggests that framing sensitivity is not uniformly distributed across the benchmark content. This suggests that certain harm categories may be more format-dependent than others, although we do not isolate category-level causes in this study.

In Figure 2, vertical stripe patterns indicate prompts whose refusal behavior flips under reframing. The stochastic appendix run yields a closely matched pattern: refusal rates are 0.72 for Direct, 0.64 for Fictional, and 0.50 for Translation, and FSI remains 0.24 with a 95% bootstrap confidence interval of [0.16, 0.33]. The corresponding appendix paired risk differences are 0.08, 0.22, and 0.14. Manual inspection of 20 randomly sampled flagged deterministic outputs reveals no false positives, supporting the rule-based refusal detector for these sampled cases. At the same time, because the refusal rule is English-anchored, the Translation absolute refusal rate should be interpreted more cautiously than the aggregate instability pattern.

## 5. Discussion

If refusal outcomes shift under light prompt reframing, then benchmark scores are not purely measuring harmful-intent recognition. They are also measuring how the benchmark prompt is packaged for the model. This weakens the interpretation of a single benchmark number as a robust safety property.

Our findings therefore target evaluation methodology rather than alignment or mechanistic explanation. The main implication is straightforward: safety benchmarks should be tested for framing robustness before they are used as stable comparative metrics. A benchmark score that changes by 21 percentage points under benign prompt reframing is difficult to interpret as a stable property of the underlying harmful request alone. Even under a single model evaluation, a 21-point shift under benign formatting is non-trivial in the context of leaderboard comparisons.

The persistence of framing sensitivity under both deterministic and stochastic decoding suggests that the observed effect is not solely attributable to decoding artifacts.

## 6. Limitations

- Single model evaluation
- Single benchmark
- Rule-based refusal detection
- No human evaluation

The refusal rule may miss nuanced refusals or incorrectly label non-refusals that happen to contain one of the trigger phrases. Because the detector is English-anchored, Translation framing is the most likely source of false negatives, which means its absolute refusal rate is less secure than the Direct and Fictional estimates. We therefore include a fixed random audit file of deterministic Translation non-refusal outputs for manual review, but we do not replace the primary rule-based labels with human annotations in this paper. Because the study is intentionally narrow, the results should be interpreted as evidence of measurement sensitivity, not as a universal statement about all models or all safety benchmarks.
We do not use human evaluation for primary benchmark scoring; the appendix manual audits are limited to small sanity checks.

## 7. Ethics

The study evaluates harmful prompts from an existing public benchmark. We do not introduce new harmful content, optimize jailbreaks, or propose methods for increasing attack success. The goal is to assess evaluation reliability and benchmark robustness.

## Appendix A. Stochastic Decoding Robustness

We repeat the full 100-prompt, 3-framing evaluation once at temperature 0.7 with no additional repetitions. This appendix tests whether framing sensitivity is merely a deterministic decoding artifact. Under stochastic decoding, refusal rates are 0.72 for Direct, 0.64 for Fictional, and 0.50 for Translation. The appendix FSI is 0.24 with a 95% bootstrap confidence interval of [0.16, 0.33]. Pairwise exact McNemar tests remain significant for Direct vs Fictional ($p=0.0215$), Direct vs Translation ($p=4.77\times 10^{-7}$), and Fictional vs Translation ($p=5.19\times 10^{-4}$).

## Appendix B. Reproducibility and Access

- Hugging Face model page: `meta-llama/Meta-Llama-3-8B-Instruct`
- Model revision: `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`
- Access: request access through the Hugging Face model page by accepting the gated license terms and submitting the access form.
- HarmBench commit: `8e1604d1171fe8a48d8febecd22f600e462bdcdd`

## Appendix C. Refusal-Rule Sanity Check

We draw a fixed random sample of 20 matched-pattern cases from the deterministic run and record manual false-positive annotations. Manual inspection of 20 random flagged cases revealed no false positives. The observed false-positive rate in this audit is 0.0. This audit is intentionally limited in scope, but it provides a direct check against the concern that the refusal regex is spuriously matching non-refusal outputs.

Because Translation framing is the most plausible source of false negatives under an English-anchored refusal detector, the analysis pipeline also emits a second fixed random sample of deterministic Translation outputs currently labeled as non-refusal. We do not use that targeted audit to alter the main reported scores, but we include it as a reproducible follow-up check on the most important detector failure mode.

## Responsible NLP Checklist Notes

- Dataset license: HarmBench is distributed in a repository licensed under MIT.
- Harmful content use: This work evaluates harmful instructions from a public benchmark and does not introduce new harmful prompts.
- Dual-use risk: The paper studies evaluation sensitivity and does not optimize attack or bypass success.
- AI writing assistance disclosure: Portions of this manuscript were edited with AI-assisted language tools.

## Acknowledgements

Portions of this manuscript were edited with AI-assisted language tools.

## References

- Mantas Mazeika et al. 2024. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal.
- Leonie Rottger et al. 2024. XSTest: A test suite for identifying exaggerated safety behaviours in large language models.
- Alexander Wei et al. 2024. Jailbroken: How does LLM safety training fail?
- Andy Zou et al. 2023. Universal and transferable adversarial attacks on aligned language models.
- Melanie Sclar et al. 2023. Quantifying language models' sensitivity to spurious features in prompt design or: How I learned to start worrying about prompt formatting.
- Han Zhou et al. 2024. Larger and more instructable language models become less reliable.
