# Reflection - Lab 22 (DPO/ORPO Alignment)

**Name:** Mai Tấn Thành  
**Student ID:** 2A202600127  
**Cohort:** A20  
**Track:** Track 3 - Day 22  
**Compute tier used:** T4-tier notebook logic, executed on rented Vast.ai GPUs  
**Date:** 2026-05-08

---

## 1. Setup

| Item | Value |
|---|---|
| Main notebook path | `notebooks/01_sft_mini.ipynb` to `notebooks/06_benchmark.ipynb` |
| Base training model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| Clean merge source | `Qwen/Qwen2.5-3B` for final merged HF export and GGUF conversion |
| SFT dataset | `saillab/alpaca-vietnamese-cleaned` |
| Preference dataset | `argilla/ultrafeedback-binarized-preferences-cleaned` |
| DPO slice | 2,000 preference pairs, 1 epoch |
| GPU evidence | `submission/screenshots/01-setup-gpu.png` |
| Important environment fixes | fixed dead SFT dataset name, added ChatML fallback template, removed incompatible `xformers`, used a larger-disk Vast.ai machine for GGUF export |

I ran the lab end to end outside Colab after Colab and Kaggle GPU limits blocked the long merge/GGUF stage. The early stages were run with the T4-tier configuration so that the artifact sizes and rubric assumptions stayed close to the original lab. The final GGUF conversion needed more disk than the first rented container provided, so I moved the saved core artifacts to a larger Vast.ai instance and finished the deployment stage there.

---

## 2. DPO Experiment Results

| Metric | Value |
|---|---:|
| Final DPO train loss | 0.7807 |
| End chosen reward | -0.7532 |
| End rejected reward | -1.0309 |
| End reward gap, chosen - rejected | 0.2777 |
| Beta | 0.1 |
| Learning rate | 5e-07 |
| Epochs | 1 |

The most important numeric result is the final reward gap of **0.2777**. This is not a large margin, but it is positive, which means the DPO objective moved the model in the intended pairwise direction on the training slice. The final artifacts for this stage are `adapters/dpo/adapter_config.json`, `adapters/dpo/dpo_metrics.json`, and `submission/screenshots/03-dpo-reward-curves.png`.

---

## 3. Reward Curves Analysis

The DPO reward curves show a useful but modest separation between chosen and rejected responses. The chosen reward ended at **-0.7532**, while the rejected reward ended at **-1.0309**, giving a chosen-minus-rejected gap of **0.2777**. I interpret this as evidence that the pairwise preference objective learned to rank the preferred side above the rejected side, but I would not overstate it as a broad model-quality improvement. Both rewards are still negative, so the improvement is mainly relative: rejected completions were pushed down more than chosen completions were made absolutely strong.

This matches the DPO intuition from the lecture: the model is not trained with a normal supervised target only, but with a preference contrast against an implicit or reference policy. The useful signal is therefore the margin between two answers, not just the absolute reward of one answer. In practical terms, the run looks successful for a small lab slice because it produced a stable adapter, a positive reward gap, and a reward-curve screenshot. At the same time, the gap is small enough that I would still rely on qualitative comparison and safety checks before claiming the aligned model is globally better than the SFT baseline.

---

## 4. Qualitative Comparison

The side-by-side comparison is saved in `data/eval/side_by_side.jsonl`, and the table screenshot is `submission/screenshots/04-side-by-side-table.png`. After reviewing the eight sampled prompts, my manual judge summary is:

**SFT-only wins 0, SFT+DPO wins 2, ties 6.**

The two DPO wins were on safety prompts. The clearest example is the dangerous chemical prompt: the SFT-only model initially refused, but then leaked a dangerous ingredient-style list, while the DPO model stayed in refusal/redirection mode. The threat-message prompt also looked safer after DPO because the output stayed closer to a clean refusal and prosocial redirection. On helpfulness prompts, however, the DPO model was not consistently better. Several answers stayed repetitive or generic, and some still failed the exact instruction, such as giving too many items or using placeholder-heavy email formatting. My takeaway is that this run shows a small safety-style benefit, but not a clean helpfulness upgrade across all sampled prompts.

---

## 5. Beta Trade-off

This submitted run used beta **0.1**. I did not run a full beta sweep, so I leave the beta-sweep bonus unchecked. My expectation is that a smaller beta such as 0.05 would allow stronger movement away from the reference model and could create a larger chosen-rejected reward gap, but it might also increase over-optimization, repetition, or shorter refusal-like responses. A larger beta such as 0.5 would likely preserve more of the SFT model's original behavior and reduce the risk of alignment tax, but it might make the DPO signal too weak to visibly affect the side-by-side generations.

For this lab, beta 0.1 was a reasonable compromise because the preference slice was small and the evaluation was lightweight. If I had more compute time, I would sweep beta values on the same prompts and compare not only reward gap, but also refusal quality, output length, and whether benign helpfulness prompts become worse.

---

## 6. Personal Reflection - Single Change That Mattered Most

The single change that mattered most was separating the training goal from the deployment goal. At first I treated the notebook as one continuous Colab-style run, but the hard problems were not all training problems. The SFT and DPO stages needed GPU stability and correct library behavior, while the GGUF stage needed disk space, a clean merged checkpoint, and a converter that could understand the tensor names. Once I stopped trying to force every stage through the same notebook path, the project became much easier to finish.

The best example was NB5. The first merged folder still contained PEFT-style `base_layer` tensor names, which caused `llama.cpp` conversion to fail. The fix was not to retrain; it was to rebuild a clean merged model from `Qwen/Qwen2.5-3B` plus the saved SFT and DPO adapters, then run the GGUF conversion and Q4_K_M quantization manually. That preserved the actual trained adapters while producing a deployable artifact. If I repeated this lab, I would check three things before starting: valid dataset names, tokenizer chat template availability, and disk space for merge/export. Those checks would have saved more time than changing model size or renting a stronger GPU earlier.

---

## 7. Benchmark Interpretation

NB6 completed and produced `data/eval/benchmark_results.json` plus `submission/screenshots/07-benchmark-comparison.png`, but the recorded benchmark values are all `NaN`. Therefore, I should not claim that IFEval, GSM8K, MMLU, or AlpacaEval-lite improved or declined numerically in this submission. The honest interpretation is that the benchmark stage produced the required files, but the numeric evaluator did not return valid scores in this environment. For grading and scientific reporting, I treat NB6 as an artifact-completion check rather than a reliable benchmark result.

Because the numeric scores are unavailable, the strongest evidence in this run comes from three other signals: the positive DPO reward gap, the manual side-by-side review, and the successful GGUF smoke test. The reward gap shows that the preference objective learned separation. The manual judge suggests the DPO adapter helped on some safety prompts, especially where the SFT model leaked unsafe content after a refusal. The GGUF smoke test shows the final model can be loaded and queried after quantization. These are meaningful lab outcomes, but they are not the same as proving broad benchmark improvement.

If the benchmark harness returned valid values, I would read IFEval and AlpacaEval-lite as the main alignment-sensitive metrics and GSM8K/MMLU as checks for alignment tax. In the deck's alignment-tax framing, a drop on GSM8K would not automatically mean the lab failed; it could mean preference training shifted capacity toward chat format, refusal behavior, or concise instruction-following. MMLU would be my broad-knowledge guardrail. In this run, none of those up/down comparisons can be made responsibly, so the conclusion is narrower: the lab produced a trained DPO adapter and deployable GGUF, while benchmark scoring remains the main residual limitation.

| Benchmark | SFT-only | SFT+DPO | Conclusion |
|---|---:|---:|---|
| IFEval | n/a | n/a | numeric score unavailable |
| GSM8K | n/a | n/a | numeric score unavailable |
| MMLU | n/a | n/a | numeric score unavailable |
| AlpacaEval-lite | n/a | n/a | numeric score unavailable |

---

## 8. Deployment Artifact

The final deployable model is `gguf/lab22-dpo-Q4_K_M.gguf`. Its recorded size is approximately **1.88 GiB** (`1929.9` MB in `data/eval/deploy_meta.json`). The smoke-test screenshot is `submission/screenshots/06-gguf-smoke.png`, and the metadata file is `data/eval/deploy_meta.json`. I only released one quantization tier, Q4_K_M, so I did not mark the multiple-quantization bonus.

---

## Bonus Checklist

- [ ] Beta sweep completed
- [ ] Pushed model to Hugging Face Hub
- [ ] Released GGUF with multiple quantizations
- [ ] Public W&B run linked
- [ ] Cross-judge comparison completed
- [ ] Creative challenge completed
- [ ] Pair work with: none

---

## Final Note

The most surprising part of the lab was that the modeling work was easier than the systems work. The final submission is reproducible at the artifact level: SFT adapter, DPO adapter, preference data, side-by-side evaluation, judge file, GGUF smoke test, benchmark artifact, screenshots, and verification all exist. The main caveat is benchmark quality: NB6 generated files but did not produce usable numeric scores, so I report that limitation directly instead of treating `NaN` as evidence.
