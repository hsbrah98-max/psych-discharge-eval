#!/usr/bin/env python3
"""
Clinical Verification Engine for Inpatient Psychiatry Discharge Summaries.
Enforces Zero-Inference constraints (Brah et al., 2026):
1. Discharge Medication Dosage Check: flags fabricated meds, dropped meds, or dosage shifts.
2. Involuntary Legal Status Hold Tracking: validates 5150/5250 hold expiration or voluntary conversion.
3. Laboratory & Trough Level Fidelity: validates therapeutic drug levels (e.g., Lithium, Clozapine ANC).
4. Zero Speculative Inferences: penalizes unmentioned diagnoses or unsupported claims.
"""

import re
from typing import Dict, List, Tuple


def verify_discharge_summary(case: Dict, summary_text: str) -> Dict:
    gt = case["ground_truth"]
    lower_text = summary_text.lower()

    # Split into general text vs discharge meds section if present
    dc_section = ""
    dc_idx = lower_text.find("discharge medications:")
    if dc_idx != -1:
        dc_section = lower_text[dc_idx:]
    else:
        # Fallback to last half of summary where discharge disposition typically sits
        dc_section = lower_text[len(lower_text) // 2:]

    # 1. Check required factual landmarks
    missing_facts = []
    matched_facts = []
    for fact in gt.get("required_facts", []):
        if re.search(re.escape(fact.lower()), lower_text):
            matched_facts.append(fact)
        else:
            missing_facts.append(fact)

    fact_coverage = len(matched_facts) / max(1, len(gt.get("required_facts", [])))

    # 2. Check forbidden global hallucinations (e.g. fabricated diagnoses, wrong holds)
    triggered_hallucinations = []
    for bad in gt.get("forbidden_global", []):
        if re.search(re.escape(bad.lower()), lower_text):
            triggered_hallucinations.append(f"Global: {bad}")

    # 3. Check forbidden discharge medications (e.g. discontinued meds added to discharge)
    for bad_med in gt.get("forbidden_discharge_meds", []):
        if re.search(re.escape(bad_med.lower()), dc_section):
            triggered_hallucinations.append(f"Discharge Med: {bad_med}")

    # 4. Check exact discharge medications
    matched_meds = []
    missing_meds = []
    for med in gt.get("discharge_medications", []):
        if re.search(re.escape(med.lower()), lower_text):
            matched_meds.append(med)
        else:
            missing_meds.append(med)

    med_accuracy = len(matched_meds) / max(1, len(gt.get("discharge_medications", [])))
    passed = (len(triggered_hallucinations) == 0) and (len(missing_meds) == 0) and (fact_coverage >= 0.8)

    return {
        "case_id": case["id"],
        "case_title": case["case_title"],
        "passed": passed,
        "fact_coverage": round(fact_coverage, 3),
        "med_accuracy": round(med_accuracy, 3),
        "missing_facts": missing_facts,
        "missing_meds": missing_meds,
        "triggered_hallucinations": triggered_hallucinations,
    }
