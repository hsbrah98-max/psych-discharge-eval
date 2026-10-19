# `psych-discharge-eval`: Automated Clinical Verification for Inpatient Psychiatric Discharge Summaries

**Author:** Harman Brah, MD

An automated clinical evaluation harness and zero-inference fact-checker for language models generating psychiatric hospital discharge summaries (Hospital Course narratives and discharge medication reconciliations).

## The Clinical Problem

Language models applied to Electronic Health Record (EHR) progress notes frequently prioritize smooth narrative synthesis over verbatim accuracy. In acute inpatient psychiatry, small hallucinations produce catastrophic medical and legal errors:
1. **Discharge Medication Dosage Shifts:** Altering titrations (e.g., writing *Lithium 300 mg daily* instead of *Lithium carbonate 600 mg BID*, or discharging a patient on in-house PRN sedatives that were discontinued).
2. **Involuntary Legal Hold Inaccuracies:** Failing to document 5150 hold expiration or voluntary conversion, creating legal liability.
3. **Laboratory & Therapeutic Monitoring Omissions:** Omitting mandatory therapeutic drug levels (Lithium, Clozapine ANC/REMS requirements).
4. **Speculative Diagnostic Inferences:** Hallucinating unconfirmed diagnoses (e.g. labeling unipolar major depression as Bipolar disorder).

## Framework & Architecture

The **Zero-Inference Extractive Protocol** restricts the language model to explicit, verbatim source note extraction.

- `cases/synthetic_inpatient_cases.json`: Synthetic inpatient EHR packets (intake H&P, interval notes, discharge plan) with ground-truth clinical facts.
- `verifier.py`: Automated clinical validation engine:
  - Exact match and regex verification of discharge medications.
  - Section-aware screening to ensure discontinued inpatient PRNs are never added to discharge orders.
  - Involuntary legal hold transition tracking.
- `evaluate.py`: Benchmark harness comparing unconstrained model outputs against zero-inference guardrailed summaries.

## Quick Start

```bash
# Run built-in self-test comparing unconstrained baseline vs. zero-inference guardrail
python3 evaluate.py --self-test

# Grade candidate model discharge summaries from a JSON file
python3 evaluate.py --responses my_model_summaries.json
```
