# Toxic_Comment_Detection_BERT
# Toxic Comment Classification with DistilBERT: Transfer Learning, Fine-Tuning, Hyperparameter Tuning and SHAP Interpretability

**Assignment 03 · Skills for Industry Competitiveness and Innovation Program (SICIP) · BRAC University**

| | |
|---|---|
| **Name** | Mrittika Devi Mati |
| **Student ID** | `1000129281` |
| **Repository** | `https://github.com/devimrittika/Toxic_Comment_Detection_BERT` |
| **Notebook** | `SICIP_Training_Assignment3.ipynb` |
| **Hardware** | Google Colab, NVIDIA Tesla T4 (fp16) |
| **Software** | PyTorch 2.11.0, Transformers 5.22.0, SHAP 0.54.1 |

---

## 1. Summary

The task is to classify a user comment as **toxic** or **not toxic** using the Civil Comments dataset. The pipeline has four stages:

1. **Transfer learning:** a pre-trained DistilBERT used as a frozen feature extractor (baseline).
2. **Fine-tuning:** all DistilBERT weights unfrozen and trained end-to-end.
3. **Hyperparameter tuning:** a learning-rate search on the fine-tuned model.
4. **Interpretability:** SHAP explanations for four selected test predictions.

**Main results**

| Model | Split | Accuracy | Toxic precision | Toxic recall | Toxic F1 |
|---|---|---:|---:|---:|---:|
| Frozen, class-weighted (baseline) | Validation | 0.8150 | 0.2469 | 0.5799 | 0.3463 |
| Fine-tuned, lr = 2e-5 (before tuning) | Validation | 0.9393 | 0.6426 | 0.6331 | 0.6379 |
| Fine-tuned, lr = 1e-5 (after tuning) | Validation | 0.9413 | 0.6451 | 0.6775 | 0.6609 |
| **Fine-tuned, lr = 1e-5 (final)** | **Test** | **0.9380** | **0.5977** | **0.6582** | **0.6265** |

Fine-tuning roughly doubled the toxic-class F1 compared with the frozen baseline (0.346 → 0.638). Tuning the learning rate added a further, smaller gain (0.638 → 0.661 on validation). The final held-out test F1 is 0.627.

> **Note on metrics.** Only about 8% of comments are toxic, so accuracy is misleading (a model that always predicts "not toxic" scores about 92%). All conclusions below are based mainly on **precision, recall and F1 of the toxic class**.

---

## 2. Dataset and Preprocessing

**Source:** [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments), loaded with `load_dataset("google/civil_comments")`. The full dataset has 1,804,874 / 97,320 / 97,320 train / validation / test rows.

**Label threshold:** the `toxicity` score is converted to a binary label:

- **Toxic (1)** if `toxicity >= 0.5`
- **Not toxic (0)** if `toxicity < 0.5`

**Subset (compute-limited):** a random subset was drawn from each official split with `shuffle(seed=42)`.

| Split | Samples | Not toxic | Toxic | % toxic |
|---|---:|---:|---:|---:|
| Train | 20,000 | 18,387 | 1,613 | 8.1% |
| Validation | 4,000 | 3,662 | 338 | 8.5% |
| Test | 4,000 | 3,684 | 316 | 7.9% |

The training subset is about 1.1% of the full training split.

**Preprocessing:**

- Only the `text` and derived `labels` columns are kept.
- Tokenization with the DistilBERT uncased tokenizer, truncation to **128 tokens**, and **dynamic padding** per batch (`DataCollatorWithPadding`).
- `token_type_ids` are removed because DistilBERT does not use them.

**Class imbalance.** Because the data is heavily imbalanced, a class-weighted cross-entropy loss was used for all models after the first baseline run. Weights are computed from the training label counts as `n / (2 · n_class)`:

| Class | Weight |
|---|---:|
| Not toxic | 0.5439 |
| Toxic | 6.1996 |

This is implemented in a custom `WeightedTrainer` that overrides `compute_loss`.

---

## 3. Model Choice: `distilbert/distilbert-base-uncased`

DistilBERT is a BERT-based model produced by knowledge distillation from BERT-base. I chose it because:

- **It satisfies the "BERT-based" requirement** and has the same tokenizer family and architecture style as BERT (6 transformer layers instead of 12).
- **It is much cheaper than BERT-base** (about 66.96M parameters; roughly 40% smaller and about 60% faster at inference, according to its authors) while retaining most of BERT's language-understanding performance.
- **It fits the compute budget.** The pipeline trains 7 models (2 frozen, 1 fine-tuned, 4 tuning runs) on a free-tier Colab T4 within the assignment's time limit. Tuning with a larger model would have allowed fewer experiments.
- **Uncased text is suitable here.** Capitalization carries little extra signal in this dataset compared with the vocabulary itself. (Shouting in capitals may carry some toxicity signal that the uncased model cannot see; see the limitations.)

---

## 4. Experiments

All models share these settings unless stated otherwise: max length 128, batch size 16 (train) / 32 (eval), 3 epochs, AdamW with the Trainer defaults (linear learning-rate decay, no warm-up, weight decay 0), fp16, seed 42.

### 4.1 Transfer learning: frozen feature extractor (baseline)

All parameters of `model.distilbert` were frozen (`requires_grad = False`). Only the classification head (`pre_classifier` + `classifier`) was trained.

| | Parameters |
|---|---:|
| Total | 66,955,010 |
| Trainable | 592,130 (0.88%) |

Two frozen runs were made, both at lr = 2e-5:

1. **Unweighted loss.** The model collapsed to predicting "not toxic" for every comment. Validation confusion matrix: `[[3662, 0], [338, 0]]`. Accuracy was 91.6% on validation (92.1% on test), but toxic precision, recall and F1 were all **0**. This shows that accuracy alone is not usable here.
2. **Class-weighted loss.** This is the baseline used for comparison, because it is the fair counterpart to the fine-tuned model, which also uses weighted loss.

### 4.2 Fine-tuning (end-to-end)

A fresh pre-trained DistilBERT was loaded, all 66,955,010 parameters were unfrozen (100% trainable), and the model was trained with the same weighted loss and settings (lr = 2e-5, 3 epochs).

### 4.3 Hyperparameter tuning

- **Method:** grid search (4 configurations, each starting from fresh pre-trained weights with the same seed).
- **Hyperparameter searched:** learning rate ∈ {1e-5, 2e-5, 3e-5, 5e-5}. Batch size (16) and epochs (3) were fixed.
- **Selection criterion:** validation toxic-class F1. Within each run, `load_best_model_at_end=True` with `metric_for_best_model="f1"` kept the best of the three epoch checkpoints.
- **Test data was not used for selection.**

**Tuning results (validation set):**

| Learning rate | Batch size | Epochs | Accuracy | Precision | Recall | **F1** | Val. loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1e-5** | 16 | 3 | 0.9413 | 0.6451 | 0.6775 | **0.6609** | 0.6700 |
| 2e-5 | 16 | 3 | 0.9393 | 0.6426 | 0.6331 | 0.6379 | 0.8275 |
| 3e-5 | 16 | 3 | 0.9363 | 0.6125 | 0.6686 | 0.6393 | 0.5219 |
| 5e-5 | 16 | 3 | 0.9385 | 0.6411 | 0.6183 | 0.6295 | 0.5511 |

**Screenshot of the tuning results:**

![Hyperparameter tuning results](screenshots/tuning_results.png)

The best learning rate is **1e-5**.

A useful consistency check: the lr = 2e-5 tuning run reproduces the earlier fine-tuning run exactly (F1 0.6379, precision 0.6426, recall 0.6331), so the pipeline is deterministic given the seed.

---

## 5. Results and Justification

### 5.1 Frozen baseline vs. fine-tuned model (without tuning)

All rows are on the **validation set** so the comparison is like for like.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Frozen, unweighted | 0.9155 | 0.0000 | 0.0000 | 0.0000 |
| Frozen, weighted | 0.8150 | 0.2469 | 0.5799 | 0.3463 |
| Fine-tuned, weighted (lr 2e-5) | 0.9393 | 0.6426 | 0.6331 | 0.6379 |

**Why fine-tuning is much better than the frozen baseline:**

- **The pre-trained representations are not toxicity-specific.** The frozen encoder was trained for masked-language modelling, so its output features encode general syntax and semantics. A head with 0.59M parameters on top can only draw a boundary in that fixed feature space. Fine-tuning lets all six transformer layers adapt so that toxic language (insults, aggressive phrasing) separates from non-toxic language.
- **The frozen model trades precision for recall.** Working back from the reported metrics, the weighted frozen model flags roughly 800 of 4,000 validation comments as toxic and about 600 of them are false alarms. This explains its low accuracy (81.5%) and low precision (0.25). The fine-tuned model flags roughly 330 comments with only about 120 false alarms, so its precision is about 2.6× higher while its recall is also higher (0.63 vs. 0.58).
- **Class weighting is necessary in both settings.** Without it, the frozen model never predicted the toxic class; the weights force the model to treat missed toxic comments as costly.
- **Caveat on fairness.** The frozen baseline used lr = 2e-5, which is a typical fine-tuning rate but low for training only a freshly initialized head. A higher learning rate for the head (e.g. 1e-3) would probably have narrowed the gap somewhat. The conclusion that fine-tuning is better is expected and consistent with the literature, but the *size* of the gap in this report is likely partly inflated by this choice.

### 5.2 Without tuning vs. with tuning

| Model | Split | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Fine-tuned, lr 2e-5 (no tuning) | Validation | 0.9393 | 0.6426 | 0.6331 | 0.6379 |
| Fine-tuned, lr 1e-5 (tuned) | Validation | 0.9413 | 0.6451 | 0.6775 | 0.6609 |

Tuning improved validation F1 by **+0.023** (0.638 → 0.661). Nearly all of it came from **recall** (0.633 → 0.678, about 15 more toxic comments found out of 338), while precision stayed almost unchanged (0.643 → 0.645).

**Interpretation:**

- A smaller learning rate (1e-5) updates the pre-trained weights more gently, which preserves more of the pre-trained knowledge. With only 20,000 training examples and 1,613 toxic ones, large steps (3e-5, 5e-5) risk over-fitting the small toxic class and gave lower F1 and recall.
- The trend is not perfectly monotonic (3e-5 is marginally better than 2e-5), which indicates run-to-run noise on the order of the differences between configurations.
- **Honest assessment:** each configuration was run with one seed, and the validation set has only 338 toxic examples. A difference of 0.02 F1 corresponds to a handful of comments, so the improvement from tuning is modest and may partly be noise. The consistent finding is that lr ≤ 2e-5 works at least as well as larger values for this data.
- Validation loss does **not** track F1 (the best-F1 model has a higher loss than the lr = 3e-5 and 5e-5 runs). The loss is the class-weighted cross-entropy, which is sensitive to confident wrong predictions; F1 was used for selection because it is the metric of interest. Weighted-loss values are also not comparable with the unweighted frozen run's loss.

### 5.3 Final test results (best tuned model, never used for selection)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Not toxic | 0.9704 | 0.9620 | 0.9662 | 3,684 |
| Toxic | 0.5977 | 0.6582 | 0.6265 | 316 |
| Accuracy | | | 0.9380 | 4,000 |
| Macro avg | 0.7841 | 0.8101 | 0.7964 | 4,000 |
| Weighted avg | 0.9410 | 0.9380 | 0.9394 | 4,000 |

**Test confusion matrix:**

| | Predicted not toxic | Predicted toxic |
|---|---:|---:|
| **Actual not toxic** | 3,544 (TN) | 140 (FP) |
| **Actual toxic** | 108 (FN) | 208 (TP) |

Test F1 (0.627) is lower than validation F1 (0.661). This is expected: the configuration and checkpoint were chosen because they scored best on validation, so the validation score is slightly optimistic, and with only 316 toxic test examples the estimate itself is noisy. The model finds about two-thirds of toxic comments (recall 0.66) and about 60% of its toxic flags are correct (precision 0.60). The toxic class is clearly harder than the majority class.

---

## 6. Interpretability: SHAP

### 6.1 Why SHAP

I used **SHAP (SHapley Additive exPlanations)** with the `Partition` explainer and a text masker built from the model's tokenizer, applied to the probability of the *toxic* class.

- **Token-level, signed attributions:** each token gets a value showing how much it pushed the toxic probability up (positive) or down (negative), relative to a baseline where text is masked. This directly answers "which words made the model say toxic?"
- **Model-agnostic and faithful to the prediction:** it explains the actual output of the fine-tuned model (through `predict_proba`) and the contributions sum to the difference between the prediction and the baseline.
- **Compared with attention visualization:** attention weights show where the model *looks*, not how much a token *changes the output*, and they are known to be an unreliable explanation on their own. **Compared with LIME:** SHAP has a more principled attribution (Shapley values) and, for text, the partition method is more stable than LIME's random perturbations.
- **Limitation of my setup:** to keep run-time low, SHAP was run with `max_evals=100`. For short comments this is reasonable, but for the long comment (Example 2) it gives coarse, group-level attributions.

### 6.2 Selected predictions

From the test set, the **most confident** example in each of four categories was selected (true negative, true positive, false positive, false negative), so that the analysis covers both correct and incorrect behaviour.

| # | Category | Actual | Predicted | P(toxic) |
|---|---|---|---|---:|
| 1 | True negative | Not toxic | Not toxic | 0.0018 |
| 2 | True positive | Toxic | Toxic | 0.9974 |
| 3 | False positive | Not toxic | Toxic | 0.9972 |
| 4 | False negative | Toxic | Not toxic | 0.0023 |

### 6.3 Analysis

> **Before submitting:** insert your four SHAP screenshots below and check each analysis against what the highlighted tokens actually show in *your* plots. The paragraphs describe the comments and the likely drivers; replace or confirm the token names in `[brackets]` with the highest-magnitude tokens in your plots.

#### Example 1: True negative (P(toxic) = 0.0018)

> *"I'll stick to this subject. Tax returns can be amended, and apparently this paperwork can too."*

![SHAP example 1](screenshots/shap_example_1.png)

A neutral, factual comment about tax paperwork with no insults or emotionally charged words. The model is extremely confident it is not toxic. The SHAP values are expected to be small and mostly negative (pushing away from toxic), with no token pushing strongly towards toxic. Top contributing tokens in my plot: `[FILL]`. This is the model behaving sensibly: it does not respond to topic words (taxes, paperwork) as toxic cues.

#### Example 2: True positive (P(toxic) = 0.9974)

> *"Halfway? This little inbred moron has been a boil on the butt of humanity ... idiot ... buffoons ... He will kill them anyway with his warlord mentality."*

![SHAP example 2](screenshots/shap_example_2.png)

A long, openly abusive comment. It contains several direct insults (`inbred`, `moron`, `idiot`, `buffoons`) and violent wording. The model's high confidence is consistent with strongly positive SHAP contributions on these insult words. Top contributing tokens in my plot: `[FILL: e.g. moron, idiot, inbred]`. Because the comment is long and SHAP was limited to 100 evaluations, attributions are grouped over multi-word spans, so the plot shows *which phrases* matter rather than fine-grained single-token weights. The model is correct, and the explanation shows it is correct for the right reason (the insults) rather than for an artefact.

#### Example 3: False positive (P(toxic) = 0.9972, label: not toxic)

> *"Wasteful spenders tend to be republicans. GW Bush lefft us with a trillion plus deficit. The Donald is proposing a stupid wall which will do nothing. ... Voodoo."*

![SHAP example 3](screenshots/shap_example_3.png)

This is a harsh political opinion, not a personal attack. It was labelled not toxic by annotators, but the model is almost certain it is toxic. The likely driver is negative-opinion vocabulary such as `stupid` (and the derisive tone around `Voodoo`) that the model associates with toxic comments from training. Top contributing tokens in my plot: `[FILL: e.g. stupid]`. This shows a **limitation of the model**: it relies on lexical cues and does not distinguish *criticising an idea or policy* from *insulting a person*. Political comments are a known source of false positives for toxicity classifiers.

#### Example 4: False negative (P(toxic) = 0.0023, label: toxic)

> *"My comments keep disappearing because they are 'below the civility guideline'. ... (I always try to be respectful) ... WTF?"*

![SHAP example 4](screenshots/shap_example_4.png)

The comment is mostly a polite complaint about moderation and uses phrases such as `respectful` and `civility`, with a single mild profanity at the end (`WTF`). The annotators' toxicity score was at least 0.5, probably because of that expletive, but the model is very confident it is not toxic. The likely explanation is that the large number of polite and negative-SHAP tokens outweighs the single `WTF` token. Top contributing tokens in my plot: `[FILL: e.g. WTF positive but small; respectful, civility negative]`. Two takeaways: (a) the model can be dominated by overall polite context, so one toxic word in a long polite text is missed; (b) this label is itself debatable, so some of the "error" is **label noise** in crowd-annotated data rather than a purely model failure.

### 6.3.1 What the four explanations show together

- The model mostly learned **lexical cues** (insults, profanity, negative evaluation words). This works well for blatant toxicity (Example 2) and neutral text (Example 1).
- It **over-triggers** on harsh but non-personal criticism (Example 3) and **under-triggers** when toxic words are a small part of polite-sounding text (Example 4).
- Together these explain why precision and recall for the toxic class stay around 0.6 to 0.7 even after fine-tuning and tuning.
- Caveat: four examples chosen for confidence are illustrative, not statistically representative, and SHAP values with `max_evals=100` are approximations.

---

## 7. Test on Real-World Text

I tested the best tuned model on six comments of the kind found on discussion and review sites. For each, I gave a **reference label from my own judgement**, then compared it with the model's prediction.

> **Important honesty note:** these six sentences were written by me, not scraped from a live platform, and they are fairly easy. This is a small qualitative check, not a performance estimate.

| # | Input | Predicted | P(toxic) | My expected label | Check |
|---|---|---|---:|---|---|
| 1 | "I appreciate the effort you put into this. Thank you!" | Not toxic | 0.0029 | Not toxic | ✅ Polite thanks; no negative language. |
| 2 | "Your explanation is clear and very helpful." | Not toxic | 0.0024 | Not toxic | ✅ Positive feedback. |
| 3 | "You are an idiot and your ideas are completely worthless." | Toxic | 0.9967 | Toxic | ✅ Direct personal insult ("idiot", "worthless"). |
| 4 | "Nobody asked for your stupid opinion." | Toxic | 0.9963 | Toxic | ✅ Insulting and dismissive towards the person. |
| 5 | "I strongly disagree with your argument, but I respect your opinion." | Not toxic | 0.0026 | Not toxic | ✅ Strong disagreement but expressed respectfully. |
| 6 | "This is one of the worst explanations I have ever read." | Not toxic | 0.0044 | Not toxic | ✅ Harsh criticism of the *content*, not an attack on the person. A borderline case that some readers would call rude. |

**Result:** 6/6 predictions agree with my reference labels.

Observations:

- The model separates **respectful disagreement** (#5) and **criticism of content** (#6) from **personal insults** (#3, #4), which is the distinction that matters most in moderation.
- Confidence is very high everywhere (all above 0.99 for the predicted class), including on the borderline #6. This suggests the model is **overconfident**; the probabilities should not be read as calibrated.
- Because Example 3 in the SHAP analysis shows the model failing on harsh political criticism, a set of harder real-world inputs (sarcasm, implicit insults, political and identity-related text, obfuscated profanity) would likely expose more errors than these six easy inputs do.

---

## 8. Limitations

- **Subset training:** only about 1.1% of the training data (20,000 comments) was used; results on the full dataset would likely be higher.
- **Narrow hyperparameter search:** only the learning rate was searched (4 values, grid search). Batch size, epochs, weight decay, warm-up, maximum length and the decision threshold were not tuned.
- **Single seed per configuration:** differences between learning rates (F1 0.630 to 0.661) are of a size comparable to run-to-run noise, so the "best" learning rate should be read cautiously.
- **Baseline fairness:** the frozen baseline used a fine-tuning-scale learning rate (2e-5) for a randomly initialized head; a higher head learning rate could improve it.
- **Small positive class in evaluation:** 338 validation and 316 test toxic comments make F1 estimates noisy.
- **Label threshold:** a fixed 0.5 threshold turns a continuous, annotator-based score into hard labels. Comments just around 0.5 are inherently ambiguous, and the dataset contains label noise.
- **Real-world test is small and self-written** (6 comments).
- **Uncased model:** capitalization (e.g. shouting) is lost.
- **Fairness not evaluated:** Civil Comments is known to contain identity-term bias (non-toxic comments mentioning certain groups can be flagged as toxic). Subgroup performance was not measured here.
- **SHAP approximation:** SHAP used `max_evals=100`, and only four predictions were explained.

---

## 9. Reproducing the Results

1. Open `SICIP_Training_Assignment3.ipynb` in Google Colab with a GPU runtime (T4 or better).
2. Run all cells from top to bottom. Dependencies are installed in the first cells (`datasets`, `transformers`, `accelerate`, `evaluate`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `shap`), or with `pip install -r requirements.txt`.
3. The dataset is downloaded automatically from the Hugging Face Hub. The subset is drawn with seed 42.
4. Model checkpoints are saved to `./results/` during tuning. They are **not** included in this repository because of their size.

---

## 10. Repository Contents

```
├── SICIP_Training_Assignment3.ipynb   # full pipeline (data, baseline, fine-tuning, tuning, SHAP, real-world test)
├── README.md                          # this report
├── requirements.txt                   # Python dependencies
├── screenshots/                       # images used in this report
│   ├── tuning_results.png
│   ├── shap_example_1.png … shap_example_4.png
│   └── real_world_results.png
└── results_csv/                       # small result tables (optional)
    ├── hyperparameter_tuning_results.csv
    ├── model_comparison.csv
    ├── final_test_classification_report.csv
    └── real_world_testing_results.csv
```

Dataset files, model checkpoints and caches are intentionally **not** included.

---

## 11. References

- Civil Comments dataset: https://huggingface.co/datasets/google/civil_comments
- DistilBERT model card: https://huggingface.co/distilbert/distilbert-base-uncased
- Sanh et al., *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter* (2019)
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP), NeurIPS 2017
- Hugging Face Transformers and Datasets documentation
