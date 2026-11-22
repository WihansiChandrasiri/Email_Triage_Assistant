# University IT Email Triage Assistant

Can I classify synthetic university IT helpdesk emails more accurately than a keyword baseline and draft a reply for a staff member to review?

This repository contains the classification analysis. The keyword baseline performed better than the machine learning model on the held-out test set. Reply drafting and human approval are planned next steps and are not evaluated here.

## Data and method

The dataset contains 200 synthetic emails in four categories: Account Access, Network Issue, Software Support, and General Inquiry. Each of 40 fictional scenarios has five closely related versions. No real student or staff emails are included.

I split the data by scenario, with 32 scenarios (160 emails) for training and eight unseen scenarios (40 emails) for testing. The split is stratified by category and uses random seed 42. Grouping versions of each scenario prevents near-duplicate emails from appearing on both sides of the split.

The keyword baseline counts matching terms, breaks ties by category order, and defaults to General Inquiry when no term matches. The machine learning model uses TF-IDF unigrams and bigrams with logistic regression. Both methods use only the subject and body. I used five-fold grouped cross-validation on the training data, then fitted the model on all training emails for the held-out test.

## Results

| Evaluation | Method | Correct | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Training data | Fixed keyword baseline | 135/160 | 84.4% | 0.845 |
| Grouped cross-validation | TF-IDF + logistic regression | 92/160 | 57.5% | 0.538 |
| Held-out test | Fixed keyword baseline | 35/40 | 87.5% | 0.867 |
| Held-out test | TF-IDF + logistic regression | 30/40 | 75.0% | 0.750 |

On the held-out test, the baseline led by 12.5 percentage points in accuracy. The learned model misclassified all five versions of two test scenarios: an IT orientation question and a PDF export problem.

## Error analysis and limitations

The [error review](results/model_error_review.csv) documents ten distinct wrong predictions, one per scenario. Two came from the held-out test and eight from grouped cross-validation. Each row records the correct label, the model prediction, evidence from the email, and a possible reason for the mistake. These reasons are interpretations, not measured feature contributions.

The test set contains only two scenarios per category. Because each scenario has five similar versions, one misunderstood situation can account for five wrong predictions. Results on these synthetic emails may not reflect performance on real helpdesk messages.

## Reproduce the analysis

Install Python, Jupyter, pandas, and scikit-learn. One way to install the packages is:

```bash
python -m pip install notebook pandas scikit-learn
```

From the project folder, open `email_triage.ipynb` in Jupyter and run all cells in order. The notebook reads `data/raw/helpdesk_emails_200.csv`, creates the training and test CSV files in `data/processed/`, and displays the baseline, cross-validation, and held-out test results.

The completed explanations are in `results/model_error_review.csv`. Rerunning the notebook does not overwrite that file if it already exists.

## Project status

The classification and evaluation stage is complete. Drafting replies and requiring a staff member to review and approve them before sending remain to be implemented and evaluated.
