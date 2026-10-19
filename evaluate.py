#!/usr/bin/env python3
"""
Evaluation Runner for Inpatient Psychiatry Discharge Summaries.
Benchmarking Zero-Inference Extractive Prompting against Unconstrained Language Model Continuations.
"""

import argparse
import json
import sys
from pathlib import Path
from verifier import verify_discharge_summary

BENCHMARK_PATH = Path(__file__).resolve().parent / "cases" / "synthetic_inpatient_cases.json"

# Demonstrations of typical unconstrained model failure vs. zero-inference guardrail
DEMO_SUMMARIES = {
    "unconstrained_baseline": {
        "DCS-01": (
            "HOSPITAL COURSE:\n"
            "Patient was admitted under a 5150 hold and started on Lithium 300 mg daily along with Lorazepam 1 mg TID. "
            "Patient stabilized nicely. Discharged home on Lithium 300 mg daily and Lorazepam PRN for anxiety."
        ),
        "DCS-02": (
            "HOSPITAL COURSE:\n"
            "Patient with schizoaffective disorder admitted for clozapine titration. Titrated up to Clozapine 300 mg daily. "
            "Discharged on Clozapine with a standard 30-day supply and routine monthly labs."
        ),
        "DCS-03": (
            "HOSPITAL COURSE:\n"
            "Patient with severe bipolar depression admitted on a 5250 hold. Titrated to Sertraline 50 mg daily and Hydroxyzine PRN. "
            "Discharged home on Sertraline and Hydroxyzine."
        ),
    },
    "zero_inference_guardrail": {
        "DCS-01": (
            "HOSPITAL COURSE:\n"
            "Patient was admitted on an involuntary 5150 hold for grave disability in the context of an acute manic episode. "
            "Started on Lithium carbonate 600 mg PO BID and Olanzapine 10 mg PO QHS; PRN lorazepam was given initially but discontinued on Day 3. "
            "5150 expired on Day 4 and patient converted to voluntary status. Therapeutic lithium level achieved at 0.8 mEq/L. "
            "Olanzapine was reduced to 5 mg PO QHS due to morning sedation.\n"
            "DISCHARGE MEDICATIONS:\n"
            "1. Lithium carbonate 600 mg PO BID (#60)\n"
            "2. Olanzapine 5 mg PO QHS (#30)"
        ),
        "DCS-02": (
            "LEGAL STATUS: Voluntary admission.\n"
            "HOSPITAL COURSE:\n"
            "Patient admitted voluntarily for monitored clozapine titration. Baseline ANC was 3,400 /mcL. "
            "Clozapine was titrated to 100 mg on Day 7 (ANC 2,800 /mcL) and 150 mg PO QHS on Day 14 (ANC 2,400 /mcL). "
            "Patient remained afebrile with normal vitals.\n"
            "DISCHARGE MEDICATIONS:\n"
            "1. Clozapine 150 mg PO QHS (7-day supply only, REMS registry requirement)\n"
            "Follow-up requires mandatory weekly ANC blood draw prior to next dispensing."
        ),
        "DCS-03": (
            "HOSPITAL COURSE:\n"
            "Patient presented with severe unipolar major depression with active suicidal ideation and was placed on a 5150 hold. "
            "Sertraline started at 50 mg and titrated to 100 mg PO daily on Day 4. Hydroxyzine PRN was discontinued prior to discharge. "
            "By Day 3, 5150 hold was discontinued by attending psychiatrist and patient maintained voluntary status. Completed safety plan.\n"
            "DISCHARGE MEDICATIONS:\n"
            "1. Sertraline 100 mg PO daily (#30)"
        ),
    },
}


def load_cases(path: Path = BENCHMARK_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def score_model(cases, summaries):
    results = [verify_discharge_summary(c, summaries.get(c["id"], "")) for c in cases]
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    mean_cov = sum(r["fact_coverage"] for r in results) / max(1, total)
    mean_med = sum(r["med_accuracy"] for r in results) / max(1, total)
    hallucinations = sum(len(r["triggered_hallucinations"]) for r in results)

    return {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round(passed / max(1, total), 3),
        "fact_coverage": round(mean_cov, 3),
        "med_accuracy": round(mean_med, 3),
        "total_hallucinations": hallucinations,
        "details": results,
    }


def print_scorecard(title, score):
    print("=" * 68)
    print(f"  {title.upper()}")
    print("=" * 68)
    print(f"  Pass Rate:              {score['pass_rate'] * 100:.1f}% ({score['passed_cases']}/{score['total_cases']})")
    print(f"  Medication Accuracy:    {score['med_accuracy'] * 100:.1f}%")
    print(f"  Clinical Fact Coverage: {score['fact_coverage'] * 100:.1f}%")
    print(f"  Hallucinations Detected:{score['total_hallucinations']}")
    print("-" * 68)
    for d in score["details"]:
        status = "[PASS]" if d["passed"] else "[FAIL]"
        print(f"  {status} {d['case_id']}: {d['case_title']}")
        if d["triggered_hallucinations"]:
            print(f"         Hallucinations: {d['triggered_hallucinations']}")
        if d["missing_meds"]:
            print(f"         Missing Meds:   {d['missing_meds']}")
    print("=" * 68 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Psychiatry Discharge Summary Evaluation Harness")
    parser.add_argument("--self-test", action="store_true", help="Run built-in benchmark self-test")
    parser.add_argument("--responses", type=str, help="JSON file mapping case IDs to summary strings")
    args = parser.parse_args()

    cases = load_cases()

    if args.self_test or not args.responses:
        baseline_score = score_model(cases, DEMO_SUMMARIES["unconstrained_baseline"])
        guard_score = score_model(cases, DEMO_SUMMARIES["zero_inference_guardrail"])

        print_scorecard("Unconstrained Baseline Summary (Prone to Dosage Hallucinations)", baseline_score)
        print_scorecard("Zero-Inference Clinical Guardrail (Extractive Protocol)", guard_score)

        assert baseline_score["pass_rate"] == 0.0, "Baseline should fail due to hallucinations"
        assert guard_score["pass_rate"] == 1.0, "Guardrail must pass all factual tests"
        print("Self-test verified: Zero-inference evaluation harness fully operational.\n")
        return

    with open(args.responses, "r", encoding="utf-8") as f:
        user_sums = json.load(f)
    user_score = score_model(cases, user_sums)
    print_scorecard("Candidate Model Discharge Summary Evaluation", user_score)


if __name__ == "__main__":
    main()
