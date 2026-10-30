# Credit Risk Assessment using SHAP

An end-to-end, **explainable credit-risk scoring application**. A loan officer fills in an eleven-field application in the browser, a FastAPI service scores it with a calibrated XGBoost pipeline, and the applicant is returned a **default probability**, the **decision threshold** used, and a **Low Risk / High Risk** verdict.

The project deliberately spans three disciplines: the data-science work that produces the model, the API that serves it, and the frontend that presents it. Every layer ships in this repository with no hidden services.

![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-EB5B2E)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikitlearn&logoColor=white)
![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=white)

> **Disclaimer.** Predictions are statistical model estimates, not a lending decision. A real underwriting system requires fairness testing across protected groups, adverse-action reason codes, model-risk governance, and human review. This repository is an educational reference implementation.

---

## Table of contents

- [Results at a glance](#results-at-a-glance)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [The dataset](#the-dataset)
- [The machine-learning pipeline](#the-machine-learning-pipeline)
  - [1. Exploratory data analysis](#1-exploratory-data-analysis)
  - [2. Data validation and cleaning](#2-data-validation-and-cleaning)
  - [3. Feature/target separation and the split](#3-featuretarget-separation-and-the-split)
  - [4. Class imbalance](#4-class-imbalance)
  - [5. Preprocessing pipelines](#5-preprocessing-pipelines)
  - [6. Evaluation metrics](#6-evaluation-metrics)
  - [7. Stratified cross-validation](#7-stratified-cross-validation)
  - [8. Baseline model](#8-baseline-model)
  - [9. XGBoost and hyperparameter tuning](#9-xgboost-and-hyperparameter-tuning)
  - [10. Decision-threshold optimization](#10-decision-threshold-optimization)
  - [11. Probability calibration](#11-probability-calibration)
  - [12. SHAP interpretability](#12-shap-interpretability)
  - [13. Error analysis](#13-error-analysis)
  - [14. Model persistence](#14-model-persistence)
- [Backend](#backend)
- [Frontend](#frontend)
- [API reference](#api-reference)
- [Running locally](#running-locally)
- [Deployment](#deployment)
- [Regenerating the model artifacts](#regenerating-the-model-artifacts)
- [Known limitations](#known-limitations)
- [Technology stack](#technology-stack)

---

## Results at a glance

The shipped model is a sigmoid-calibrated XGBoost pipeline evaluated at a threshold of **0.665585**, chosen as the out-of-fold F1 optimum on the training set.

| Evaluation | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Out-of-fold, training set (calibrated) | 0.6656 | 0.955 | 0.737 | 0.832 | 0.948 | 0.904 |
| Hold-out test set | 0.6656 | 0.958 | 0.741 | 0.836 | 0.953 | 0.910 |

Hold-out confusion matrix (6,305 applications, 1,361 of which defaulted):

|  | Predicted repay | Predicted default |
|---|---|---|
| **Actually repaid** (4,944) | 4,900 (TN) | 44 (FP) |
| **Actually defaulted** (1,361) | 353 (FN) | 1,008 (TP) |

Two properties matter more than accuracy here:

- **Calibration is correct.** Mean predicted default probability across the full dataset is `0.2197` against an actual default rate of `0.2182`. When the model says 22%, roughly 22% of those applicants really do default.
- **The threshold is honest.** It was chosen on out-of-fold training predictions and re-derived on the calibrated probability scale, so it is never selected on the test rows and it matches the scale the API actually serves. It flags 17.1% of applicants as high risk while catching ~74% of true defaulters.

---

## How it works

```
┌──────────────────┐
│  Browser form    │  11 fields, grouped as Applicant / Loan request / Bureau file
│  static/         │  vanilla JS, no framework, no build step
└────────┬─────────┘
         │  POST /predict  (application/json)
         ▼
┌──────────────────┐
│  FastAPI         │  Pydantic validates and coerces the payload
│  main.py         │  models are loaded once at startup, not per request
└────────┬─────────┘
         │  pd.DataFrame([payload])   ← 11 raw, unencoded columns
         ▼
┌──────────────────────────────────────────────┐
│  sklearn Pipeline (deserialized from .pkl)    │
│                                              │
│  ColumnTransformer                           │
│    numeric     → median impute               │
│    categorical → constant impute + one-hot   │
│         │                                    │
│         ▼                                    │
│  XGBClassifier (366 trees, depth 5)          │
│         │                                    │
│         ▼                                    │
│  sigmoid calibration (Platt scaling)         │
└────────┬─────────────────────────────────────┘
         │  predict_proba(X)[:, 1]  →  p(default)
         │  decision = p >= 0.665585
         ▼
┌──────────────────┐
│  JSON response   │  { default_probability, default_prediction,
│                  │    threshold, Result }
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Verdict UI      │  animated SVG gauge, risk stamp, ledger facts
└──────────────────┘
```

The single most important architectural decision: **the entire preprocessing pipeline is serialized as one object**. The API accepts raw values, and encoding, imputation, and calibration state travel with the model. It is therefore impossible to serve a model with a subtly different preprocessing step than the one it was trained on.

---

## Repository structure

```
.
├── Credit_Risk.ipynb          # The full ML lifecycle, sections 1-16, including SHAP
├── credit_risk_dataset.csv    # 32,581 raw loan applications
├── credit_risk_model.pkl      # Calibrated pipeline (preprocessing + XGBoost + Platt)
├── best_threshold.pkl         # 0.665585, out-of-fold F1 optimum on calibrated probs
│
├── main.py                    # FastAPI app: /predict + static hosting
├── requirements.txt           # Serving dependencies, exactly pinned
├── runtime.txt                # python-3.11.9 for the platform build
├── render.yaml                # Render blueprint (build + start commands)
│
├── static/
│   ├── index.html             # Semantic markup, three fieldset groups
│   ├── style.css              # Design tokens, 3D transforms, glassmorphism
│   ├── script.js              # Vanilla JS: fetch, gauge animation, scroll motion
│   └── assets/
│       ├── cards.gif          # Hero visual (animated)
│       └── cards.mp4          # Source video for the GIF
│
├── _migrate_model.py          # One-off: re-save a legacy XGBoost pickle in a modern format
├── _retrain_threshold.py      # Reproducible: recompute threshold + refit calibration
├── _threshold_report.json     # Metrics produced by the above
└── _backup/                   # Local-only pre-fix artifacts (gitignored)
```

Files prefixed with `_` are maintenance scripts, not part of the serving path. They exist because both were genuinely needed during development: one to work around a pickle-format break, one to correct the threshold selection.

---

## The dataset

`credit_risk_dataset.csv` — 32,581 rows × 12 columns, a standard public credit-risk dataset.

| Column | Type | Meaning |
|---|---|---|
| `person_age` | numeric | Applicant age in years |
| `person_income` | numeric | Annual income |
| `person_home_ownership` | categorical | `RENT`, `MORTGAGE`, `OWN`, `OTHER` |
| `person_emp_length` | numeric | Years of employment |
| `loan_intent` | categorical | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` |
| `loan_grade` | categorical | Internal risk grade `A`–`G` |
| `loan_amnt` | numeric | Requested principal |
| `loan_int_rate` | numeric | Interest rate (%) |
| `loan_percent_income` | numeric | Loan amount as a fraction of income |
| `cb_person_default_on_file` | categorical | Prior default on record, `Y`/`N` |
| `cb_person_cred_hist_length` | numeric | Credit history length in years |
| `loan_status` | **target** | `1` = defaulted, `0` = repaid |

So there are **11 features** (7 numeric, 4 categorical) and **1 binary target**.

After cleaning: **31,522 rows**. Split 80/20 with stratification: **25,217 train / 6,305 test**, positive rate 21.6%. The ~78/22 imbalance is the reason almost every modelling decision below looks the way it does.

---

## The machine-learning pipeline

`Credit_Risk.ipynb` is organised as sixteen sections. This is what each one does and the concept behind it.

### 1. Exploratory data analysis

Before touching anything, the data is inspected:

- `df.shape` and `df.info()` — row/column counts and dtypes. Notes that `person_emp_length` is `float64` specifically because it contains nulls, while categoricals are `object`.
- `df.isnull().sum()` — a missing-value audit. `person_emp_length` and `loan_int_rate` have gaps, which determines the imputation strategy chosen later.
- `df['loan_status'].value_counts()` and a count plot — this reveals the 78/22 class imbalance.
- Box plots per numeric column — univariate distribution and outlier inspection.
- `df.corr(numeric_only=True)` as a heatmap — **multicollinearity screening**. `person_age` correlates strongly with `cb_person_cred_hist_length`, and `loan_amnt` with `loan_percent_income`: partly duplicated information. Tree models tolerate redundancy; a linear model with L2 regularisation would spread weight arbitrarily across the correlated pair.

**Concept:** EDA is not decoration. Every later choice — median rather than mean imputation, `stratify=y`, precision/recall rather than accuracy — traces back to an observation made here.

### 2. Data validation and cleaning

```python
df_copy.drop_duplicates(inplace=True)
df_copy = df_copy[df_copy['person_age'].between(18, 100)]
df_copy = df_copy[df_copy['person_emp_length'] <= df_copy['person_age']]
df_copy = df_copy[df_copy['person_emp_length'] <= 60]
df_copy = df_copy[df_copy['loan_amnt'] > 0]
```

**Concepts:**

- **Domain-logic validation beats statistical outlier detection.** An employment length longer than the applicant's age is physically impossible. No z-score or IQR rule would catch that; it requires knowing what the columns mean. Those rows are dropped rather than clipped, because the whole record is untrustworthy.
- **Duplicate removal** prevents the same application appearing in both train and test, which would leak information and inflate scores.
- **Impossible-value bounds** — age 144 and 123-year employment histories are data-entry errors.

Note what is *not* done: no row is capped, winsorised, or scaled here. The heavy lifting is deferred to the pipeline so that it is learned from training data only.

### 3. Feature/target separation and the split

```python
X = df_copy.drop('loan_status', axis=1)
y = df_copy['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

**Concepts:**

- **Supervised binary classification.** Labelled historical outcomes, two classes.
- **Hold-out validation.** 20% of rows are locked away so performance is measured on data the model has never seen. This is the primary defence against overfitting.
- **`stratify=y`.** Preserves the exact default-to-repay ratio in both splits. Without it, random chance could hand one split a much rarer positive class, making metrics incomparable across runs.
- **`random_state=42`.** Reproducibility — every number in this README can be regenerated.
- **Explicit feature-type lists.** `numeric_cols` and `categorical_cols` are declared by name, which later guarantees the API's eleven fields match training exactly. Anything unlisted is dropped by `ColumnTransformer`'s default `remainder='drop'`.

### 4. Class imbalance

```python
neg, pos = np.bincount(y_train)
scale_weights = neg / pos      # → 3.631221
```

**Concepts:**

- **The accuracy trap.** With 22% positives, a model that always predicts "will repay" scores 78% accuracy while being completely useless.
- **Cost-sensitive learning.** `scale_pos_weight` multiplies the gradient contribution of the positive class during boosting, so getting a defaulter wrong costs about 3.6× more than getting a repayer wrong. Logistic Regression gets the equivalent via `class_weight='balanced'`.
- **Why this creates a calibration debt.** Reweighting shifts the model's operating point and inflates its raw scores. A model trained this way can rank applicants well while its absolute probabilities are systematically too high. That debt is repaid in §11.

### 5. Preprocessing pipelines

Two `ColumnTransformer`s, one per model family, wrapped in `Pipeline` objects:

```python
# Logistic Regression — needs scaling
('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                  ('scaler',  StandardScaler())]), numeric_cols)
('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                  ('onehot',  OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)

# XGBoost — identical, minus the scaler
```

**Concepts:**

- **`Pipeline`** chains transforms and an estimator into a single object with three benefits. It applies transformations **inside** each CV fold, so imputation medians and scaler means are learned from training folds only and never leak from validation data. It makes the whole thing one serializable unit. And it makes hyperparameter search addressable via `step__param` names.
- **`ColumnTransformer`** applies different transformers to different column subsets in parallel and concatenates the result.
- **Median imputation** rather than mean: income and loan amount have long right tails, and the median is robust to them.
- **Constant `"Missing"` imputation for categoricals.** Whether an employment length was *recorded* can itself carry information; collapsing it into the mode would destroy that.
- **`StandardScaler` for Logistic Regression only.** Its L2 penalty is scale-sensitive, so a feature measured in lakhs would be regularised differently from one measured in years. Trees compare features to thresholds, so any monotone rescaling leaves them unchanged — scaling XGBoost input would be pure wasted computation.
- **One-hot encoding** avoids the false ordinality a label-encoder implies (`MEDICAL` is not "greater than" `PERSONAL`).
- **`handle_unknown='ignore'`** is the production safety switch: an unseen category at inference becomes an all-zero indicator row instead of raising an exception and returning a 500.

### 6. Evaluation metrics

One reusable function, `evaluate_model(name, model, X, y, threshold)`, reports accuracy, precision, recall, F1, a classification report, a confusion-matrix heatmap, and a precision–recall curve.

| Metric | Definition | Why it matters here |
|---|---|---|
| Accuracy | (TP+TN)/all | Misleading at 78/22. Reported for completeness only. |
| Precision | TP/(TP+FP) | When we flag someone risky, how often are we right? Drives customer friction. |
| Recall | TP/(TP+FN) | Of all real defaulters, how many did we catch? Drives loan losses. |
| F1 | 2PR/(P+R) | Harmonic mean; punishes lopsided precision/recall. |
| ROC-AUC | Area under TPR vs FPR | Ranking quality, threshold-independent. Flattered by huge TN counts. |
| PR-AUC / average precision | Area under the PR curve | **The right summary metric under class imbalance**, because the FPR denominator (thousands of TNs) inflates ROC. |

`predict_proba(X)[:, 1]` is the probability of class 1 — default. Column 0 is the probability of repayment; using the wrong column silently inverts the entire application.

### 7. Stratified cross-validation

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'roc_auc': 'roc_auc', 'accuracy': 'accuracy', 'precision': 'precision',
           'recall': 'recall', 'f1': 'f1'}
cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
```

**Concepts:**

- **Why CV at all.** A single train/validate split gives one noisy estimate; you cannot tell skill from luck in the split. K-fold rotates the held-out fold k times and yields both a mean and a spread.
- **Stratified folds** preserve the class ratio inside each fold, which is essential when the positive class is rare.
- **Multi-metric scoring** computes every metric in one pass instead of refitting per metric.
- **`n_jobs=-1`** parallelises folds across cores.
- **CV runs on `X_train` only.** The test set stays sealed until final reporting. This discipline is what makes the reported numbers believable.

### 8. Baseline model

```python
baseline_model = Pipeline([
    ('preprocessor', preprocessor_lr),
    ('classifier', LogisticRegression(class_weight='balanced'))
])
```

**Concepts:**

- **Baseline discipline.** Never claim a complex model helped without a simple reference point. Logistic Regression is the credit-scorecard workhorse: `sigmoid(w·x + b)`, interpretable via coefficients as log-odds, and calibrated by construction.
- **`class_weight='balanced'`** is sklearn's internal reweighting, roughly equivalent to `scale_pos_weight`.

### 9. XGBoost and hyperparameter tuning

```python
param_grid = {
    "classifier__n_estimators":     randint(150, 600),
    "classifier__max_depth":        randint(3, 9),
    "classifier__learning_rate":    uniform(0.01, 0.5),
    "classifier__subsample":        uniform(0.6, 0.4),
    "classifier__colsample_bytree": uniform(0.6, 0.4),
    "classifier__min_child_weight": randint(1, 10),
    "classifier__gamma":            uniform(0, 5),
}
RandomizedSearchCV(xg_model, param_grid, n_iter=150, cv=cv,
                   scoring="average_precision", n_jobs=-1)
```

This is the densest section conceptually.

- **Gradient boosting.** Hundreds of shallow decision trees are built sequentially, each fitted to the residual errors (negative gradients of the log-loss) of the ensemble so far. The final score is the sum of all trees' contributions. This sequential error-correction is why a boosted ensemble beats any single tree.
- **Decision trees.** Recursive binary splits on feature thresholds, greedily maximising information gain. They capture non-linearity and feature interactions natively and are invariant to monotone feature scaling.
- **Bias–variance, one hyperparameter at a time:**
  - `n_estimators` — boosting rounds. More rounds lower bias, raise variance and cost time.
  - `max_depth` — depth cap. Deeper captures interactions but overfits.
  - `learning_rate` (η) — shrinkage applied to each tree's contribution. Lower η needs more trees but generalises better; `n_estimators` and `learning_rate` must be tuned as a pair.
  - `min_child_weight` — minimum hessian (sum of instance weights) required in a leaf. Interacts directly with `scale_pos_weight`, since weighting changes what a leaf's weight sum means. This is why it must be tuned *after* fixing the class weight.
  - `gamma` (`min_split_loss`) — minimum loss reduction required to accept a split. Low gamma grows complex trees; high gamma prunes aggressively.
- **Stochastic regularisation.** `subsample` (row sampling per tree) and `colsample_bytree` (column sampling per tree) inject randomness that decorrelates trees and reduces variance — the boosting analogue of bagging.
- **Random vs grid search.** Grid search is combinatorial in dimension; seven hyperparameters would be intractable. `RandomizedSearchCV` samples `n_iter=150` configurations, which is empirically near-optimal because only a few hyperparameters actually matter and they need resolution rather than coverage.
- **`scipy.stats` as search space.** `randint(a, b)` is discrete on `[a, b)`. `uniform(loc, scale)` is continuous on `[loc, loc+scale]` — note `uniform(0.01, 0.5)` covers **0.01–0.51**, not 0.01–0.5, because the second argument is a width. A classic off-by-one in search-space definition.
- **`scoring="average_precision"`** was chosen over `roc_auc` deliberately: it optimises the area under the precision–recall curve, which is the honest summary when the positive class is rare and defaulters carry the cost.
- **`classifier__` prefix** is sklearn's pipeline namespacing: `run the search, then call .fit on X_train`.

The winning configuration, recovered from the serialized model:

```
n_estimators=366   max_depth=5          learning_rate=0.1317
subsample=0.8453   colsample_bytree=0.9374   min_child_weight=7
gamma=2.3914       scale_pos_weight=3.6312
```

### 10. Decision-threshold optimisation

```python
threshold_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_prob = cross_val_predict(random_search.best_estimator_, X_train, y_train,
                               cv=threshold_cv, method="predict_proba")[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, train_prob)

# F1 is the harmonic mean of precision and recall: 2 * (P * R) / (P + R)
f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

best_threshold = thresholds[np.argmax(f1[:-1])]
```

**Concepts:**

- **A probability is not a decision.** 0.5 is a default, not a law. Choosing the cut-off is an explicit business decision, and it belongs in a versioned artifact — `best_threshold.pkl` — not hardcoded in the API.
- **Asymmetric error costs.** In credit, a false negative (lending to a defaulter) typically costs orders of magnitude more than a false positive (declining a good applicant, which costs only lost interest). Moving the threshold down buys recall at the price of precision.
- **Out-of-fold selection.** `cross_val_predict` gives every training row a prediction from a model that never saw it. Selecting the threshold on in-sample predictions would pick a cut-off tuned to memorised rows.
- **`f1[:-1]`.** `precision_recall_curve` returns arrays one element longer than `thresholds` — the curve's terminal endpoint has no matching threshold — so the last element must be trimmed before `argmax`.

**The bug this section used to contain.** The original formula was `2*(P - R)/(P + R)` instead of `2*P*R/(P + R)`. With both terms in `[0, 1]` the result is usually negative, so `argmax` landed arbitrarily; the notebook's own saved output showed `Used Threshold: 1.00`, precision 0.00, recall 0.00 — the model flagged **nobody**. It also selected the threshold on `X_test`, which is selection leakage. Both are fixed. On this data the corrected out-of-fold optimum is `0.6938` on raw probabilities.

### 11. Probability calibration

```python
calibrated_model = CalibratedClassifierCV(
    random_search.best_estimator_, method="sigmoid", cv=threshold_cv)
calibrated_model.fit(X_train, y_train)

actual, predicted = calibration_curve(y_test, calibrated_model.predict_proba(X_test)[:, 1],
                                      n_bins=10)
```

**Concepts:**

- **What calibration means.** Among all applicants scored at 0.30, about 30% should actually default. A model can rank perfectly (high AUC) while its numbers are systematically wrong — **ranking quality and probability correctness are separate properties.**
- **Why boosting breaks it.** `scale_pos_weight` plus gradient boosting both distort the absolute scale of the score. The raw probability is a good *ranking signal* but not a good *probability*.
- **Reliability diagram.** Bin predictions into deciles; plot mean predicted probability against observed default frequency; compare to the `y = x` diagonal.
- **Platt scaling / sigmoid** (used here) fits `1/(1+exp(a·s + b))` on top of the model's raw score — two parameters, needs little data, assumes a sigmoid-shaped distortion.
- **Isotonic regression** — the flexible non-parametric alternative: more expressive, but prone to overfitting on small calibration sets.
- **Temperature scaling** — the neural-network variant, a single scalar `T` dividing the logits.
- **`cv=5`** means cross-validated calibration: the calibrator is fitted on out-of-fold predictions, so it is not trained on rows the base model already memorised.
- **Why it is not optional.** Expected loss is `PD × LGD × EAD`. If PD is uncalibrated you cannot price the loan, size loss provisions, set capital, or compare against a competitor's pricing. Calibration is what turns a ranking into a number you can multiply.

**Out-of-fold re-derivation of the threshold.** Because calibration rescales probabilities, a threshold chosen on raw scores no longer sits at the same operating point. The notebook therefore re-derives it on the calibrated scale, still leak-free, using nested CV:

```python
calibrated_oof = cross_val_predict(
    CalibratedClassifierCV(random_search.best_estimator_, method="sigmoid", cv=threshold_cv),
    X_train, y_train, cv=threshold_cv, method="predict_proba")[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, calibrated_oof)
f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_threshold = thresholds[np.argmax(f1[:-1])]      # → 0.665585
```

The outer loop keeps each row out of the evaluation; the inner calibration folds keep it out of the calibrator fit. This value — **0.665585** — is what ships in `best_threshold.pkl`, because it lives on the same probability scale the API serves.

### 12. SHAP interpretability

```python
xgb_classifier = xg_model.named_steps['classifier']
X_test_transformed = xg_model.named_steps['preprocessor'].transform(X_test)
feature_names = xg_model.named_steps['preprocessor'].get_feature_names_out()

explainer = shap.TreeExplainer(xgb_classifier)
shap_values = explainer.shap_values(pd.DataFrame(X_test_transformed, columns=feature_names))

shap.summary_plot(shap_values, X_test_df)                     # global
shap.plots.waterfall(shap.Explanation(values=shap_values[5],   # local
                                      base_values=explainer.expected_value,
                                      data=X_test_df.iloc[5],
                                      feature_names=feature_names))
```

**Concepts:**

- **The regulatory problem.** "The model said no" is not an acceptable answer in consumer lending. Adverse-action notices require specific, ranked reasons — ECOA in the US, GDPR Article 22 in the EU. Explainability is a legal requirement, not a nice-to-have.
- **Shapley values** come from cooperative game theory. Each feature is a player contributing to the payout (the prediction); its Shapley value is its **average marginal contribution across all possible feature orderings**. This gives three properties that make attributions trustworthy: **local accuracy** (contributions sum exactly to prediction − baseline), **missingness** (a feature that never changes the output gets zero), and **consistency** (if a feature becomes more important, its attribution cannot decrease).
- **`TreeExplainer`** computes exact tree SHAP in polynomial rather than exponential time by exploiting tree structure. It runs in microseconds, which is why per-applicant real-time explanations are feasible in a lending app.
- **Global explanation** (`summary_plot`, the beeswarm) ranks features by mean absolute SHAP value and colours each point by feature value, so it shows both *importance* and *direction*.
- **Local explanation** (`waterfall`) decomposes one applicant's score: baseline `E[f(X)]` plus each feature's push, summing to the final log-odds. This is the audit trail for a single decision.
- **Why the pipeline is unwrapped.** SHAP needs the raw fitted tree model and already-preprocessed input, so both halves of the pipeline are extracted via `named_steps`. `get_feature_names_out()` recovers readable names like `cat__loan_grade__E` instead of bare column indices.
- **Why SHAP on the uncalibrated model is acceptable.** Calibration is a monotone transform of the score, so feature rankings and directions survive unchanged; only absolute magnitudes shift.

### 13. Error analysis

```python
false_positive = result_df[(actual == 0) & (predicted == 1)]   # declined a good applicant
false_negative = result_df[(actual == 1) & (predicted == 0)]   # lent to a defaulter
```

**Concepts:**

- **Error costs are not symmetric.** Framing FP and FN in business terms — lost customer versus charged-off loan — is what justifies a non-0.5 threshold at all.
- **Slice-based debugging.** Inspecting *which* rows are misclassified (by age band, income band, grade) reveals segments where the model systematically fails, which you fix with features or targeted data, not with more hyperparameter search.
- **Reading the result.** At the shipped threshold, false negatives (353) still exceed false positives (44) by 8×. That imbalance is intentional: the objective is to catch defaulters, and it is tuned by precision/recall, not by minimising total errors.

### 14. Model persistence

```python
joblib.dump(calibrated_model, "credit_risk_model.pkl")
joblib.dump(best_threshold, "best_threshold.pkl")
```

**Concepts:**

- **Saving the whole pipeline, not just the estimator.** Preprocessing, XGBoost, and the five Platt calibrators serialize as one atomic object. This is why `main.py` passes raw, unscaled, unencoded columns straight from Pydantic.
- **`joblib` over `pickle`.** Same protocol, but optimised for the large NumPy arrays inside sklearn objects.
- **Two artifacts by design.** The threshold is separate so the risk appetite can be retuned without rebuilding the model.
- **The version-lock hazard.** Pickles are tied to library versions. The committed `.pkl` was written by an older XGBoost whose internal container (`Config` + `Model`) is rejected by `xgboost 3.4.1`'s C unserializer with `"input stream corrupted"` / `"Invalid cast, from Null to Object"`. `_migrate_model.py` fixes this without a runtime shim: it patches `Booster.__setstate__`, extracts the nested `{"learner": ...}` document from the legacy container, rebuilds each booster through `load_model()` (a JSON path that still works), verifies against three real CSV rows, and re-dumps in the modern format. The durable fix for any future project is `save_model`/JSON or ONNX instead of pickle.

---

## Backend

`main.py` is 57 lines and touches most of modern Python web serving.

```python
ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model['model'] = joblib.load('credit_risk_model.pkl')
    ml_model['threshold'] = joblib.load('best_threshold.pkl')
    yield
    ml_model.clear()

app = FastAPI(lifespan=lifespan)
```

| Concept | Detail |
|---|---|
| **FastAPI** | ASGI framework on **Starlette** (routing, ASGI) + **Pydantic** (validation). Generates OpenAPI JSON and Swagger UI from type hints alone. |
| **ASGI vs WSGI** | Asynchronous gateway interface: supports `async def` and long-lived connections. Served by `uvicorn`, an ASGI server — not gunicorn's WSGI mode. |
| **`lifespan` context manager** | Modern replacement for the deprecated `@app.on_event("startup")`. The model is deserialized **once at boot** into a module-level dict. Loading a 3 MB pickle per request would be catastrophic; `yield` separates startup from shutdown, and `ml_model.clear()` frees memory on graceful exit. |
| **Cold start** | Because loading happens in `lifespan`, the first request after a free-tier spin-up pays the deserialization cost. The frontend's status pill exists partly to make that visible. |
| **Pydantic `BaseModel`** | Declares the request schema. Bare type hints give coercion and type validation; violations return an automatic **422** with a structured `detail` array before the handler runs. |
| **`pd.DataFrame([data.dict()])`** | A single-row **2-D** frame — sklearn requires `(n_samples, n_features)`. The list wrapper is what makes it 2-D rather than a Series. |
| **`predict_proba(...)[:, 1][0]`** | Class-1 (default) probabilities for all rows, then the only row. |
| **`int(probability >= threshold)`** | Converts `numpy.bool_` → `int` so the response is JSON-serializable without numpy types leaking into the encoder. |
| **Response contract** | Returns probability *and* threshold *and* verdict. A bare verdict would be unauditable. |
| **`StaticFiles(directory="static", html=True)` at `/`** | Serves the SPA and maps `/` → `index.html`. Mounting at the root **after** `@app.post('/predict')` works because routes registered on `app` take precedence over the catch-all mount. |
| **`uvicorn main:app`** | Module `main` must expose a module-level object named `app`. |

Serving dependencies are pinned exactly in `requirements.txt`: `fastapi==0.115.0`, `uvicorn[standard]==0.30.6` (the `[standard]` extra pulls `uvloop` + `httptools` for a faster event loop and HTTP parser), `pydantic==2.9.2` (the v2 Rust-core rewrite), `pandas==2.2.2`, `scikit-learn==1.6.1`, `xgboost==3.4.1`, `joblib==1.4.2`.

`shap`, `matplotlib`, `seaborn`, and `scipy` are deliberately **absent**: they are training-time only, and keeping them out keeps the serving image small and the cold start short.

---

## Frontend

No React, no bundler, no build step, no `node_modules`. Three files served straight from `/static`.

### `index.html`

- **Semantic HTML5**: `<header> <nav> <main> <section> <footer>`, with the form split into three `<fieldset>`/`<legend>` groups mirroring the model's feature families.
- **Progressive enhancement**: a real `<form>` with `<label for>` wiring — usable without JavaScript.
- **Accessibility**: `aria-live="polite"` on the verdict so screen readers announce new results, `role="alert"` on the error note, `aria-hidden` on decorative layers, `aria-label` on icon-only links.
- **Inline SVG**: the nav seal is drawn with `<circle>`/`<path>`; the gauge uses `<defs>`, a `<linearGradient>`, a track circle, a fill arc, and a threshold tick.
- **Web fonts with `preconnect`**: DNS/TLS warm-up before the font CSS request (Instrument Serif, Inter Tight, JetBrains Mono).

### `style.css`

| Technique | Where it is used |
|---|---|
| **Custom properties as design tokens** | `:root` holds colours, spacing, and type scale — one source of truth |
| **Custom properties as a JS↔CSS channel** | JS writes `--ry`, `--rx`, `--mx`, `--my`, `--cx`, `--cy`, `--p`, `--d`; CSS consumes them in `transform` and `animation-delay` |
| **3D transforms** | `perspective: 1400px` on the stage plus `rotateX/rotateY` on the tilt wrapper produces real parallax depth |
| **`mask-image: radial-gradient(...)`** | Feathers the hero GIF's edges into the background — no image editing required |
| **Glassmorphism** | `backdrop-filter: blur(18px) saturate(140%)` with the `-webkit-` prefix |
| **`color-mix(in srgb, ...)`** | Native colour blending, no Sass |
| **Fluid typography** | `clamp(48px, 7.2vw, 108px)` — scales with the viewport, bounded at both ends |
| **`minmax(0, 1fr)` grid columns** | Prevents grid items from refusing to shrink below their content width |
| **`@keyframes`** | `lineUp`, `fadeUp`, `halo`, `floaty`, `chipIn`, `wheel`, `spin`, `verdictIn` |
| **`@media (prefers-reduced-motion: reduce)`** | Disables all motion for users with vestibular sensitivity |

### `script.js`

| Technique | Detail |
|---|---|
| **IIFE module pattern** | `(() => { ... })()` — everything scoped, zero globals, no bundler needed |
| **rAF-throttled scroll** | A `ticking` flag plus `requestAnimationFrame` decouples scroll handling from layout thrash |
| **`{ passive: true }` listeners** | Promises the handler will not call `preventDefault()`, letting the browser composite off the main thread |
| **Scroll-linked animation** | `p = clamp(scrollY / (heroHeight − viewportHeight))` is written to `--p`, so CSS interpolates the hero as you scroll |
| **`IntersectionObserver`** | Reveals sections at 15% visibility, then `unobserve`s them — the performant alternative to scroll-position math |
| **Pointer-driven 3D tilt** | Normalises `clientX/Y` against `getBoundingClientRect()` and maps it to degrees; `data-depth` gives the floating chips differential parallax |
| **Frame-rate-independent smoothing** | `cx += (tx − cx) * 0.08` — linear interpolation toward the cursor each frame |
| **Capability detection, not UA sniffing** | `matchMedia("(prefers-reduced-motion: reduce)")` and `matchMedia("(hover: hover) and (pointer: fine)")` disable tilt on touch devices and for motion-sensitive users |
| **`fetch` + `async/await`** | `POST /predict` with error handling that reads FastAPI's `detail` body and surfaces it |
| **Health probe** | `GET /openapi.json` — free liveness check, since FastAPI publishes it automatically; drives the "service ready / unreachable" pill |
| **Client-side derived feature** | `loan_percent_income = loan_amnt / person_income` recomputed on every keystroke, overridable by the user |
| **SVG stroke animation** | `GAUGE_CIRCUMFERENCE = 540.35` (= 2π × 86, matching `<circle r="86">`); animating `stroke-dashoffset` sweeps the arc. The double `requestAnimationFrame` forces a style flush so the transition replays on every new result |
| **Number tweening** | `easeOutCubic` (`1 − (1 − t)³`) driven by `performance.now()` |
| **Animation restart trick** | `el.style.animation = "none"; void el.offsetWidth; el.style.animation = ""` |
| **Threshold visualisation** | `rotate(thresholdPct * 3.6deg)` — 100% is 360°, so the tick literally shows where the cut-off sits on the dial |

---

## API reference

### `POST /predict`

Request body — all eleven fields required:

```json
{
  "person_age": 30,
  "person_income": 600000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 100000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

Response:

```json
{
  "default_probability": 0.022256371977586637,
  "default_prediction": 0,
  "threshold": 0.665585081060881,
  "Result": "Low Risk"
}
```

That payload is exactly what the form pre-fills on load, so it doubles as a smoke test: it should come back at roughly 2.2% — comfortably below the cut-off.

| Field | Meaning |
|---|---|
| `default_probability` | Calibrated probability of default, `0.0`–`1.0` |
| `default_prediction` | `1` if `default_probability >= threshold`, else `0` |
| `threshold` | The decision cut-off in force, exposed for auditability |
| `Result` | Human-readable verdict: `High Risk` or `Low Risk` |

**Example.** A young, low-income, high-grade-risk applicant with a prior default:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"person_age":22,"person_income":30000,"person_home_ownership":"RENT",
       "person_emp_length":1.0,"loan_intent":"MEDICAL","loan_grade":"E",
       "loan_amnt":20000,"loan_int_rate":18.5,"loan_percent_income":0.67,
       "cb_person_default_on_file":"Y","cb_person_cred_hist_length":2}'
```

```json
{
  "default_probability": 0.9475519362799207,
  "default_prediction": 1,
  "threshold": 0.665585081060881,
  "Result": "High Risk"
}
```

### `GET /`

Serves `static/index.html` and the rest of the frontend.

### `GET /openapi.json`, `GET /docs`

Auto-generated OpenAPI schema and interactive Swagger UI.

---

## Running locally

```bash
git clone https://github.com/GiteshRao/Credit-Risk-Assesment-using-SHAP.git
cd Credit-Risk-Assesment-using-SHAP

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Start the service:

```bash
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000**.

Full training dependencies (not needed to serve, needed to run the notebook):

```bash
pip install jupyter matplotlib seaborn scipy shap
jupyter notebook Credit_Risk.ipynb
```

---

## Deployment

`render.yaml` is an Infrastructure-as-Code blueprint, so no dashboard clicking is required:

```yaml
services:
  - type: web
    name: credit-ledger
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    autoDeploy: true
```

| Detail | Why |
|---|---|
| `--host 0.0.0.0` | Bind all interfaces; binding loopback makes the container unreachable from the platform proxy |
| `$PORT` | The platform injects the port at runtime; hardcoding it fails health checks |
| `runtime.txt` → `python-3.11.9` | Pins the interpreter for reproducible builds |
| `plan: free` | The instance sleeps when idle, so expect a cold start of roughly 30 s; the frontend status pill reflects this |
| `autoDeploy: true` | Every push to `main` redeploys |

Any platform that runs `uvicorn main:app` will host this unchanged — Railway, Fly.io, Heroku, or a plain Docker image.

---

## Regenerating the model artifacts

`_retrain_threshold.py` reproduces both `.pkl` files without re-running the 150-iteration hyperparameter search (it recovers the winning configuration from the serialized pipeline):

```bash
python _retrain_threshold.py
```

It:

1. Reproduces the notebook's cleaning and stratified 80/20 split exactly.
2. Recomputes the threshold from **out-of-fold, uncalibrated** predictions on the training set (reference: `0.6938`).
3. Runs **nested CV** — `cross_val_predict` wrapping `CalibratedClassifierCV` — to obtain out-of-fold **calibrated** probabilities.
4. Derives the shipped threshold from those (`0.665585`).
5. Refits `CalibratedClassifierCV` on all 25,217 training rows.
6. Backs up the previous artifacts to `_backup/`, writes the new ones, reloads them, and asserts the round trip.
7. Emits `_threshold_report.json` with every metric.

To retrain from scratch instead, run the notebook top to bottom — `Credit_Risk.ipynb` ends by writing both artifacts.

---

## Known limitations

Documented deliberately rather than hidden, because these are the things a production hand-off would need to address first.

1. **`data.dict()` is deprecated in Pydantic v2** and should be `data.model_dump()` in `main.py`.
2. **`CORSMiddleware` is imported but never registered.** Harmless while the API is same-origin only; a latent bug the moment it is called from another domain.
3. **No server-side range validation.** The form carries `novalidate`, and the Pydantic model uses bare types with no `Field(ge=..., le=...)` constraints. A negative income or an age of 200 would be scored rather than rejected.
4. **`loan_grade` and other categoricals are plain `str`**, so the API accepts eg `"Z"`. `OneHotEncoder(handle_unknown='ignore')` keeps that from crashing, but it silently becomes an all-zero indicator row. `Literal[...]` types would close this.
5. **SHAP in the notebook explains `xg_model`** (the untuned `fit`), while the deployed model is the tuned, calibrated pipeline. The explanations therefore describe a slightly different estimator than the one making decisions. Point §14 at `random_search.best_estimator_` for exactness.
6. **The notebook's §8 model still passes `n_estimator=300`** (missing `s`); XGBoost silently ignores unknown parameters, so that baseline CV ran with the default 100 rounds.
7. **Pickle is a version lock.** It works today at `xgboost 3.4.1` / `scikit-learn 1.6.1`, and `_migrate_model.py` exists because it already broke once. XGBoost's native `save_model` plus explicit preprocessing, or ONNX, would remove the class of problem.
8. **No automated tests.** A single `pytest` case posting a known row and asserting the probability would have caught the threshold bug, the deprecated `.dict()` call, and any future preprocessing drift.
9. **No fairness evaluation.** Credit models are regulated; real deployment needs disparate-impact testing across protected groups, reason codes derived from SHAP, and documented model-risk governance.

---

## Technology stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python 3.11.9 (runtime), 3.12 in dev venv | Everything server-side |
| Notebook | Jupyter, NumPy, pandas, Matplotlib, Seaborn | EDA, modelling, visualisation |
| Modelling | scikit-learn 1.6.1 | Pipelines, `ColumnTransformer`, CV, tuning, metrics, calibration |
| Modelling | XGBoost 3.4.1 | Gradient-boosted trees |
| Modelling | SciPy | Hyperparameter search distributions |
| Explainability | SHAP | Tree SHAP global + local attributions |
| Serialisation | joblib | Pipeline and threshold artifacts |
| API | FastAPI 0.115.0 | Routing, OpenAPI generation |
| API | Uvicorn 0.30.6 | ASGI server |
| Validation | Pydantic 2.9.2 | Request schema and coercion |
| Data | pandas 2.2.2 | Single-row inference frames |
| Frontend | HTML5, CSS3, vanilla ES2020+ | No framework, no build step |
| Frontend | SVG, `requestAnimationFrame`, `IntersectionObserver`, `fetch` | Gauge, motion, reveals, API calls |
| Deployment | Render (`render.yaml`) | Blueprint-driven PaaS |
| Tooling | Git, GitHub, virtualenv | Version control and environment isolation |

---

## Acknowledgements

Built as an end-to-end study in explainable machine learning for credit risk: the modelling choices are driven by the economics of lending — asymmetric error costs, rare defaults, and the regulatory need to explain every decision — rather than by leaderboard metrics alone.
