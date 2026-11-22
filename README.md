# University IT Email Triage Assistant

Can I classify synthetic university IT helpdesk emails more accurately than a keyword baseline and draft a reply for a staff member to review?

I evaluated the classification stage. The keyword baseline was more accurate on the held-out test set. Reply drafting and human approval are the next stage and have not been evaluated in this repository.

## Data and method

I used 200 synthetic subject/body pairs across four categories: Account Access, Network Issue, Software Support, and General Inquiry. Each of 40 fictional scenarios has five closely related versions. No real student or staff emails are included.

I split by scenario with seed 42 and stratified by category. Training used 32 scenarios (160 emails). Testing used eight unseen scenarios (40 emails). The fixed baseline counts distinct matching terms, resolves ties by rule order, and defaults to General Inquiry when nothing matches. The learned classifier uses TF-IDF unigrams and bigrams with logistic regression. Only the subject and body enter the model. I used five-fold grouped cross-validation within the training split, fitting the vocabulary separately in each fold, then fitted the pipeline on all training emails for the final test.

## Results

| Evaluation | Method | Correct | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Development | Fixed keyword baseline | 135/160 | 84.4% | 0.845 |
| Development, out-of-fold | TF-IDF + logistic regression | 92/160 | 57.5% | 0.538 |
| Held-out test | Fixed keyword baseline | 35/40 | 87.5% | 0.867 |
| Held-out test | TF-IDF + logistic regression | 30/40 | 75.0% | 0.750 |

The baseline led by 12.5 percentage points on test accuracy. The learned model missed all five versions of two scenarios: a question about IT orientation and a report of figures disappearing during PDF export.

## Error analysis and limitations

The [error review](results/model_error_review.csv) covers ten distinct model mistakes, one per scenario. Two come from the held-out test and eight from development cross-validation. Each row includes the correct and predicted categories, a phrase from the email, and a possible explanation. For example, “Can visitors use campus Wi-Fi” asks about policy, but the model predicted Network Issue. The explanations are interpretations, not measured feature contributions.

Only two scenarios per category were held out. Five similar messages can therefore produce five similar mistakes. These scores describe performance on this synthetic dataset and give limited evidence about real inboxes.

## Reproduce the evaluation

From the repository root, run:

```bash
python -m pip install -r requirements.txt
python evaluate.py
```

The script reads [the synthetic dataset](data/raw/helpdesk_emails_200.csv) and regenerates [metrics](results/metrics.csv) and [held-out predictions](results/test_predictions.csv). The reviewed error annotations are kept separately.
