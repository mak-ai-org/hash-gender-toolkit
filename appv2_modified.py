# -*- coding: utf-8 -*-
"""MSRH Gender Equity Auditor

Seven-stage gender-bias audit for maternal, sexual, and reproductive-health
AI systems.

Framework alignment:
- UNESCO Recommendation on the Ethics of AI, Principle 10: fairness and
  non-discrimination, with a gender-responsive lifecycle lens.
- Fairlearn-inspired MetricFrame-style subgroup metrics and trade-off review.
- IBM AI Fairness 360-inspired statistical-parity, disparate-impact,
  equal-opportunity, average-odds, and mitigation diagnostics.

Important: this application implements transparent metric approximations with
pandas/numpy. It does not claim to be an official Fairlearn or AIF360
certification, clinical validation, or regulatory approval.
"""

import io
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
except ImportError:
    FPDF = None


# -----------------------------------------------------------------------------
# Configuration and framework definitions
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="MSRH Gender Equity Auditor — UNESCO / Fairlearn / AIF360",
    layout="wide",
    page_icon="⚖️",
)

UNESCO_PRINCIPLE_10 = (
    "AI actors should promote social justice and safeguard fairness and "
    "non-discrimination through an inclusive lifecycle approach. AI systems "
    "should avoid reinforcing discriminatory outcomes, include affected "
    "communities, consider women and girls and vulnerable groups, and provide "
    "effective remedy."
)

UNESCO_GENDER_REQUIREMENTS = [
    "Define a transversal gender perspective in the ethical impact assessment.",
    "Protect the rights, safety, integrity, dignity, and agency of women and girls.",
    "Identify and proactively redress gender stereotypes and discriminatory bias.",
    "Include affected women, girls, and gender-diverse communities in review.",
    "Consider intersectional and access divides: age, language, disability, geography, and socioeconomic position.",
    "Provide effective complaint, remedy, escalation, rollback, and accountability mechanisms.",
]

THRESHOLDS = {
    "demographic_parity_difference": 0.10,
    "disparate_impact_ratio": 0.80,
    "equal_opportunity_difference": 0.10,
    "average_odds_difference": 0.10,
    "pronoun_consistency": 0.95,
    "max_false_negative_rate_gap": 0.10,
}

# These are screening thresholds, not universal clinical or legal rules.

STAGES = [
    "1. Scope and gender impact",
    "2. Data representation and participation",
    "3. Model fairness measurement",
    "4. Mitigation and trade-offs",
    "5. Output equity and clinical safety",
    "6. Governance, remedy, and monitoring",
    "7. Evidence report and deployment decision",
]


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def init_state():
    defaults = {
        "scope_evidence": {},
        "data_evidence": {},
        "data_results": None,
        "model_results": None,
        "mitigation_results": None,
        "output_results": None,
        "governance_evidence": {},
        "audit_pdf": None,
        "dataset": None,
        "deployment_status": "NOT ASSESSED",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def safe_text(value):
    return "" if pd.isna(value) else str(value)


def nonempty_columns(df):
    return ["(not provided)"] + list(df.columns)


def selected_column(choice):
    return None if choice == "(not provided)" else choice


def value_or_unknown(series):
    return series.fillna("Unknown / not reported").astype(str).replace({"": "Unknown / not reported"})


def normalize_gender(value):
    text = safe_text(value).strip().lower()
    if not text or text in {"nan", "none", "unknown", "prefer not to say", "not reported"}:
        return "Unknown / not reported"
    if re.search(r"\b(trans woman|cis woman|woman|women|female|girl)\b", text):
        return "Women / female"
    if re.search(r"\b(trans man|cis man|man|men|male|boy)\b", text):
        return "Men / male"
    if re.search(r"non.?binary|genderqueer|agender|trans", text):
        return "Gender-diverse"
    return str(value).strip()


def infer_gender_from_text(text):
    text = safe_text(text).lower()
    female = bool(re.search(r"\b(she|her|hers|woman|women|female|girl|pregnant|maternal|lactating)\b", text))
    male = bool(re.search(r"\b(he|him|his|man|men|male|boy)\b", text))
    if female and not male:
        return "Women / female"
    if male and not female:
        return "Men / male"
    return "Unknown / not reported"


def age_band(value):
    text = safe_text(value).lower()
    match = re.search(r"\b(\d{1,3})\b", text)
    if match:
        age = int(match.group(1))
        if age < 20:
            return "Adolescent (<20)"
        if age < 35:
            return "Adult 20–34"
        if age < 50:
            return "Adult 35–49"
        return "Adult 50+"
    if any(token in text for token in ["adolescent", "teen", "youth"]):
        return "Adolescent (<20)"
    return "Unknown / not reported"


def clinical_category(text):
    text = safe_text(text).lower()
    if any(term in text for term in ["gonorrhea", "syphilis", "chlamydia", "sti", "std"]):
        return "STI / sexual health"
    if any(term in text for term in ["hiv", "aids"]):
        return "HIV"
    if any(term in text for term in ["pregnan", "maternal", "prenatal", "postpartum", "birth"]):
        return "Maternal health"
    if any(term in text for term in ["pelvic", "menstrual", "period", "cramp", "ectopic"]):
        return "Reproductive pain"
    if any(term in text for term in ["violence", "assault", "rape", "coercion"]):
        return "Violence / safeguarding"
    return "General MSRH"


def prepare_dataset(df, question_col, answer_col, gender_col=None, age_col=None, language_col=None):
    out = df.copy()
    out["Question_Text"] = out[question_col].map(safe_text)
    out["Answer_Text"] = out[answer_col].map(safe_text)
    if gender_col:
        out["Gender_Group"] = out[gender_col].map(normalize_gender)
        out["Gender_Basis"] = "declared metadata"
    else:
        out["Gender_Group"] = out["Question_Text"].map(infer_gender_from_text)
        out["Gender_Basis"] = "text inference — exploratory only"
    out["Age_Band"] = out[age_col].map(age_band) if age_col else out["Question_Text"].map(age_band)
    out["Language_Group"] = value_or_unknown(out[language_col]) if language_col else "Not provided"
    out["Clinical_Category"] = out["Question_Text"].map(clinical_category)
    out["Answer_Word_Count"] = out["Answer_Text"].map(lambda x: len(x.split()) if x.strip() else 0)
    out["Detailed_Response"] = (out["Answer_Word_Count"] >= 10).astype(int)
    out["Has_Safety_Netting"] = out["Answer_Text"].str.lower().str.contains(
        r"seek urgent|emergency|immediately|call emergency|go to hospital|health worker", regex=True
    ).astype(int)
    out["Has_Stigmatizing_Terms"] = out["Answer_Text"].str.lower().str.contains(
        r"promiscuous|dirty|bad woman|deserve|shame|immoral", regex=True
    ).astype(int)
    return out


# -----------------------------------------------------------------------------
# UNESCO Stage 1: scope and gender impact
# -----------------------------------------------------------------------------

def render_scope_stage():
    st.header("Stage 1 — Scope and Gender Impact")
    st.caption("UNESCO lifecycle gate: define purpose, beneficiaries, risks, and affected groups before measuring model fairness.")
    st.info(UNESCO_PRINCIPLE_10)
    with st.form("scope_form"):
        purpose = st.text_area("Intended purpose", placeholder="Example: support MSRH triage and referral; not autonomous diagnosis.")
        beneficiaries = st.text_area("Intended beneficiaries and affected communities")
        prohibited = st.text_area("Prohibited uses and high-risk decisions")
        gender_risks = st.text_area("Gender and intersectional risks identified")
        human_decision = st.checkbox("A qualified human retains final determination for high-impact or life-affecting decisions.")
        gender_impact = st.checkbox("A gender-responsive ethical impact assessment has been completed.")
        community = st.checkbox("Affected women, girls, and gender-diverse communities participated in design or review.")
        submitted = st.form_submit_button("Save Stage 1 evidence")
    if submitted:
        st.session_state.scope_evidence = {
            "purpose": purpose, "beneficiaries": beneficiaries, "prohibited": prohibited,
            "gender_risks": gender_risks, "human_decision": human_decision,
            "gender_impact": gender_impact, "community": community,
        }
        st.success("Stage 1 evidence saved.")
    evidence = st.session_state.scope_evidence
    if evidence:
        missing = [label for label, value in {
            "intended purpose": evidence.get("purpose"),
            "beneficiaries": evidence.get("beneficiaries"),
            "gender risks": evidence.get("gender_risks"),
            "human final determination": evidence.get("human_decision"),
            "gender impact assessment": evidence.get("gender_impact"),
            "community participation": evidence.get("community"),
        }.items() if not value]
        if missing:
            st.warning("UNESCO scope evidence incomplete: " + ", ".join(missing))
        else:
            st.success("Stage 1 evidence is complete for review.")


# -----------------------------------------------------------------------------
# Stage 2: data representation and participation
# -----------------------------------------------------------------------------

def representation_table(df, group_col="Gender_Group"):
    grouped = df.groupby(group_col, dropna=False)
    result = grouped.agg(
        Sample_Count=(group_col, "size"),
        Mean_Answer_Words=("Answer_Word_Count", "mean"),
        Detailed_Response_Rate=("Detailed_Response", "mean"),
        Safety_Netting_Rate=("Has_Safety_Netting", "mean"),
        Stigmatizing_Term_Rate=("Has_Stigmatizing_Terms", "mean"),
    ).reset_index()
    result["Detailed_Response_Rate"] *= 100
    result["Safety_Netting_Rate"] *= 100
    result["Stigmatizing_Term_Rate"] *= 100
    return result.round(2)


def chi_square_gap(df, group_col, outcome_col):
    clean = df[[group_col, outcome_col]].dropna()
    table = pd.crosstab(clean[group_col], clean[outcome_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"chi2": np.nan, "p_value": np.nan, "status": "INSUFFICIENT EVIDENCE"}
    chi2, p_value, _, _ = chi2_contingency(table)
    return {"chi2": round(float(chi2), 4), "p_value": round(float(p_value), 5),
            "status": "REVIEW REQUIRED" if p_value < 0.05 else "NO SIGNAL DETECTED"}


def render_data_stage(df):
    st.header("Stage 2 — Data Representation, Participation, and Provenance")
    st.caption("UNESCO inclusion and gender requirements plus AIF360-inspired dataset-level diagnostics.")
    basis = df["Gender_Basis"].iloc[0] if len(df) else "not available"
    if basis != "declared metadata":
        st.warning("Gender groups are inferred from text. This is exploratory only and cannot establish deployment evidence.")
    else:
        st.success("Gender groups use a declared metadata field. Verify consent, provenance, and permitted use.")
    table = representation_table(df)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("Small groups are not evidence of fairness. They are evidence of limited measurement power.")
    st.bar_chart(table.set_index("Gender_Group")[["Detailed_Response_Rate", "Safety_Netting_Rate"]])
    gap = chi_square_gap(df, "Gender_Group", "Detailed_Response")
    st.write(f"AIF360-inspired response-completeness diagnostic: χ²={gap['chi2']}, p={gap['p_value']} — **{gap['status']}**")
    with st.expander("UNESCO data, participation, and provenance checklist"):
        checklist = {}
        for i, item in enumerate([
            "Collection purpose, consent, provenance, and permitted secondary use are documented.",
            "Representation is compared with the intended deployment population.",
            "Gender-diverse, adolescent, disabled, rural, multilingual, and marginalized groups are considered.",
            "Affected communities participated in data or system review.",
            "Cleaning and labeling procedures were checked for gender stereotypes and proxy harms.",
            "Small-cell privacy and disclosure risks are controlled.",
        ]):
            checklist[item] = st.checkbox(item, key=f"data_unesco_{i}")
        if st.button("Save Stage 2 evidence"):
            st.session_state.data_evidence = checklist
            st.success("Stage 2 evidence saved.")


def build_binary_outcome(df, outcome_col=None):
    if outcome_col and outcome_col in df.columns:
        values = df[outcome_col]
        if values.nunique(dropna=True) == 2:
            codes, _ = pd.factorize(values)
            return pd.Series(codes, index=df.index).astype(int)
    # Demonstration fallback: urgency proxy from question text. Must be replaced
    # with validated labels for a real model audit.
    text = df["Question_Text"].str.lower()
    urgent = text.str.contains(
        r"severe|excruciating|heavy bleeding|faint|unconscious|emergency|ectopic|assault|suicid", regex=True
    )
    return urgent.astype(int)


def paired_group_metrics(y_true, y_pred, groups):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = pd.Series(groups).fillna("Unknown / not reported").astype(str).to_numpy()
    rows = []
    for group in sorted(pd.unique(groups)):
        mask = groups == group
        tp = int(((y_true == 1) & (y_pred == 1) & mask).sum())
        tn = int(((y_true == 0) & (y_pred == 0) & mask).sum())
        fp = int(((y_true == 0) & (y_pred == 1) & mask).sum())
        fn = int(((y_true == 1) & (y_pred == 0) & mask).sum())
        n = int(mask.sum())
        pos = int(((y_true == 1) & mask).sum())
        neg = int(((y_true == 0) & mask).sum())
        rows.append({
            "Group": group, "N": n,
            "Selection_Rate": float(y_pred[mask].mean()) if n else np.nan,
            "TPR_Recall": tp / pos if pos else np.nan,
            "FPR": fp / neg if neg else np.nan,
            "FNR_Missed_Positive": fn / pos if pos else np.nan,
            "Accuracy": (tp + tn) / n if n else np.nan,
        })
    return pd.DataFrame(rows)


def fairness_summary(group_metrics):
    gm = group_metrics.dropna(subset=["Selection_Rate"])
    if gm.empty:
        return {}
    selection = gm["Selection_Rate"]
    tpr = gm["TPR_Recall"].dropna()
    fpr = gm["FPR"].dropna()
    fnr = gm["FNR_Missed_Positive"].dropna()
    dpd = float(selection.max() - selection.min()) if len(selection) else np.nan
    dir_ratio = float(selection.min() / selection.max()) if selection.max() > 0 else 1.0
    tpr_diff = float(tpr.max() - tpr.min()) if len(tpr) else np.nan
    fpr_diff = float(fpr.max() - fpr.min()) if len(fpr) else np.nan
    fnr_gap = float(fnr.max() - fnr.min()) if len(fnr) else np.nan
    return {
        "demographic_parity_difference": dpd,
        "disparate_impact_ratio": dir_ratio,
        "equal_opportunity_difference": tpr_diff,
        "average_odds_difference": (tpr_diff + fpr_diff) / 2 if not np.isnan(tpr_diff) and not np.isnan(fpr_diff) else np.nan,
        "false_negative_rate_gap": fnr_gap,
    }


def render_metric_table(summary):
    rows = []
    definitions = {
        "demographic_parity_difference": ("Demographic parity difference", "lower is better", THRESHOLDS["demographic_parity_difference"]),
        "disparate_impact_ratio": ("Disparate impact ratio", "higher is better", THRESHOLDS["disparate_impact_ratio"]),
        "equal_opportunity_difference": ("Equal opportunity difference", "lower is better", THRESHOLDS["equal_opportunity_difference"]),
        "average_odds_difference": ("Average odds difference", "lower is better", THRESHOLDS["average_odds_difference"]),
        "false_negative_rate_gap": ("False-negative-rate gap", "lower is better", THRESHOLDS["max_false_negative_rate_gap"]),
    }
    for key, (label, direction, threshold) in definitions.items():
        value = summary.get(key, np.nan)
        if np.isnan(value):
            status = "INSUFFICIENT EVIDENCE"
        elif direction == "higher is better":
            status = "SCREENING PASS" if value >= threshold else "REVIEW REQUIRED"
        else:
            status = "SCREENING PASS" if value <= threshold else "REVIEW REQUIRED"
        rows.append({"Metric": label, "Value": round(value, 4) if not np.isnan(value) else "—", "Screening rule": f"{direction}; threshold {threshold}", "Status": status})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Stage 3: model measurement
# -----------------------------------------------------------------------------

def render_model_stage(df):
    st.header("Stage 3 — Model Fairness Measurement")
    st.caption("Fairlearn-inspired MetricFrame and AIF360-inspired group fairness metrics.")
    outcome_options = ["(question-text proxy)"] + list(df.columns)
    outcome_choice = st.selectbox("Validated binary outcome column (optional)", outcome_options, key="model_outcome")
    outcome_col = None if outcome_choice == "(question-text proxy)" else outcome_choice
    y_true = build_binary_outcome(df, outcome_col)
    # Demonstration prediction proxy. In production, replace with model outputs.
    x = df["Question_Text"].str.lower()
    y_pred = x.str.contains(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid", regex=True).astype(int)
    if st.button("Run baseline Fairlearn / AIF360-inspired measurement", key="run_model"):
        gm = paired_group_metrics(y_true, y_pred, df["Gender_Group"])
        summary = fairness_summary(gm)
        st.session_state.model_results = {"group_metrics": gm, "summary": summary, "outcome_col": outcome_col}
    result = st.session_state.model_results
    if result:
        st.subheader("Disaggregated performance")
        st.dataframe(result["group_metrics"].round(4), use_container_width=True, hide_index=True)
        st.subheader("Fairness metric dashboard")
        st.dataframe(render_metric_table(result["summary"]), use_container_width=True, hide_index=True)
        st.info("These are transparent local approximations of Fairlearn/AIF360 measures. They require validated labels, model predictions, subgroup counts, and uncertainty analysis before clinical use.")
        if any(v == "REVIEW REQUIRED" for v in render_metric_table(result["summary"])["Status"]):
            st.error("Baseline model: review required. Do not interpret aggregate accuracy as evidence of gender fairness.")


# -----------------------------------------------------------------------------
# Stage 4: mitigation and trade-offs
# -----------------------------------------------------------------------------

def mitigation_variants(df):
    x = df["Question_Text"].str.lower()
    baseline = x.str.contains(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid", regex=True).astype(int).to_numpy()
    # Transparent demonstrations of threshold/abstention alternatives. These
    # are not substitutes for a validated mitigation implementation.
    conservative = x.str.contains(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid|pain|bleeding", regex=True).astype(int).to_numpy()
    abstaining = baseline.copy()
    low_confidence = ~x.str.contains(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid", regex=True)
    abstaining[low_confidence] = 0
    return {"Baseline": baseline, "Conservative safety rule": conservative, "Abstention-aware rule": abstaining}


def render_mitigation_stage(df):
    st.header("Stage 4 — Mitigation, Retesting, and Trade-offs")
    st.caption("AIF360-inspired pre/in/post-processing comparison with Fairlearn-style utility/fairness trade-off review.")
    y_true = build_binary_outcome(df)
    variants = mitigation_variants(df)
    rows = []
    for name, pred in variants.items():
        gm = paired_group_metrics(y_true, pred, df["Gender_Group"])
        summary = fairness_summary(gm)
        accuracy = float((np.asarray(y_true) == pred).mean())
        rows.append({"Variant": name, "Accuracy": accuracy, **summary})
    table = pd.DataFrame(rows)
    st.dataframe(table.round(4), use_container_width=True, hide_index=True)
    st.line_chart(table.set_index("Variant")[["Accuracy", "demographic_parity_difference", "false_negative_rate_gap"]])
    st.warning("A mitigation is not automatically acceptable because one fairness metric improves. Review false negatives, calibration, clinical utility, subgroup size, and operational feasibility.")
    selected = st.selectbox("Select variant for documented human review", list(variants), key="selected_variant")
    rationale = st.text_area("Document mitigation rationale, trade-offs, and clinical review", key="mitigation_rationale")
    if st.button("Save mitigation review", key="save_mitigation"):
        st.session_state.mitigation_results = {"table": table, "selected": selected, "rationale": rationale}
        st.success("Mitigation review saved. This does not approve deployment.")


# -----------------------------------------------------------------------------
# Stage 5: output equity and clinical safety
# -----------------------------------------------------------------------------

PRONOUN_MAP = {
    "he": "she", "she": "he", "him": "her", "her": "him", "his": "hers", "hers": "his",
    "himself": "herself", "herself": "himself",
}


def pronoun_only_counterfactual(text):
    pattern = re.compile(r"(?<!\w)(" + "|".join(sorted(PRONOUN_MAP, key=len, reverse=True)) + r")(?!\w)", re.I)
    def replace(match):
        original = match.group(0)
        swapped = PRONOUN_MAP.get(original.lower(), original)
        if original.isupper():
            return swapped.upper()
        if original[0].isupper():
            return swapped.capitalize()
        return swapped
    return pattern.sub(replace, safe_text(text))


def render_output_stage(df):
    st.header("Stage 5 — Output Equity, Counterfactuals, and Clinical Safety")
    st.caption("UNESCO harm prevention and remedy lens; pronoun-only testing is one diagnostic, not a certification.")
    st.markdown("Test types should be separated: pronoun-only, gender identity, age, pregnancy status, language, and intersectional pairs. The default test changes pronouns only.")
    if st.button("Run pronoun-only paired test", key="run_output"):
        records = []
        for text in df["Question_Text"]:
            cf = pronoun_only_counterfactual(text)
            # Demonstration output proxy; replace with the actual system output.
            original = int(bool(re.search(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid", text.lower())))
            counterfactual = int(bool(re.search(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid", cf.lower())))
            records.append({"Original": text, "Pronoun-only pair": cf, "Original decision": original, "Counterfactual decision": counterfactual, "Stable": original == counterfactual})
        result = pd.DataFrame(records)
        rate = float(result["Stable"].mean()) if len(result) else 0.0
        st.session_state.output_results = {"records": result, "consistency": rate}
    result = st.session_state.output_results
    if result:
        rate = result["consistency"]
        st.metric("Pronoun-only decision consistency", f"{rate:.1%}")
        if rate < THRESHOLDS["pronoun_consistency"]:
            st.error("Review required: pronoun-only changes altered decisions.")
        else:
            st.success("No decision change observed in the pronoun-only sample; continue with clinical safety and intersectional tests.")
        st.dataframe(result["records"], use_container_width=True, hide_index=True)
        st.warning("For MSRH deployment, add validated tests for missed emergencies, unsafe advice, stigmatizing language, safeguarding, adolescent confidentiality, and referral equity.")


# -----------------------------------------------------------------------------
# Stage 6: governance, remedy, monitoring
# -----------------------------------------------------------------------------

def render_governance_stage():
    st.header("Stage 6 — Governance, Remedy, and Continuous Monitoring")
    st.caption("UNESCO requires effective remedy against discriminatory or biased algorithmic outcomes.")
    items = [
        "Users are informed when AI is used.",
        "Qualified human oversight is required for high-impact decisions.",
        "Healthcare workers can override recommendations.",
        "Emergency and safeguarding escalation pathways are available.",
        "Users can report harmful, discriminatory, or stigmatizing outputs.",
        "Complaints are reviewed for gender and intersectional impact.",
        "Corrective actions, owners, due dates, and retest dates are tracked.",
        "Performance and fairness are monitored by gender and vulnerable subgroup.",
        "Language, geography, disability, age, and access divides are monitored.",
        "The system has rollback, suspension, and incident-response procedures.",
        "Model, prompt, policy, dataset, and audit versions are recorded.",
        "Affected communities participate in periodic review.",
    ]
    evidence = {}
    for i, item in enumerate(items):
        evidence[item] = st.checkbox(item, key=f"gov_{i}")
    if st.button("Save governance evidence", key="save_governance"):
        st.session_state.governance_evidence = evidence
        st.success("Stage 6 evidence saved.")
    count = sum(st.session_state.governance_evidence.values()) if st.session_state.governance_evidence else 0
    st.progress(count / len(items))
    st.write(f"Governance evidence: {count}/{len(items)} controls documented.")


# -----------------------------------------------------------------------------
# Stage 7: evidence report and deployment decision
# -----------------------------------------------------------------------------

def deployment_decision():
    blockers = []
    scope = st.session_state.scope_evidence
    if not scope or not scope.get("purpose") or not scope.get("gender_risks"):
        blockers.append("Scope or gender-impact evidence incomplete")
    if not scope.get("gender_impact", False):
        blockers.append("Gender-responsive ethical impact assessment not evidenced")
    if st.session_state.data_results is None:
        blockers.append("Data representation audit not run")
    if st.session_state.model_results is None:
        blockers.append("Baseline fairness measurement not run")
    if st.session_state.output_results is None:
        blockers.append("Output counterfactual audit not run")
    if st.session_state.mitigation_results is None:
        blockers.append("Mitigation and trade-off review not documented")
    governance = st.session_state.governance_evidence
    if len(governance) < 12 or not all(governance.values()):
        blockers.append("Governance, remedy, and monitoring controls incomplete")
    if st.session_state.model_results:
        statuses = render_metric_table(st.session_state.model_results["summary"])["Status"].tolist()
        if "REVIEW REQUIRED" in statuses or "INSUFFICIENT EVIDENCE" in statuses:
            blockers.append("Critical fairness metric requires review or has insufficient evidence")
    if st.session_state.output_results and st.session_state.output_results["consistency"] < THRESHOLDS["pronoun_consistency"]:
        blockers.append("Pronoun-only counterfactual inconsistency")
    return blockers


class AuditPDF(FPDF if FPDF else object):
    if FPDF:
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, "MSRH GENDER EQUITY AUDIT REPORT", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "UNESCO Principle 10 / Fairlearn-inspired / AIF360-inspired", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

def pdf_safe_text(value):
    replacements = {
        "—": "-",
        "–": "-",
        "‑": "-",
        "−": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
        "•": "-",
        "→": "->",
        "×": "x",
        "≤": "<=",
        "≥": ">=",
        "χ": "chi",
        "²": "2",
        "⚖️": "[fairness]",
    }

    text = str(value)

    for source, target in replacements.items():
        text = text.replace(source, target)

    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf(blockers):
    if FPDF is None:
        return None
    pdf = AuditPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, f"Generated: {datetime.now(timezone.utc).isoformat()}\nThis report is an evidence summary, not clinical validation, regulatory approval, or official Fairlearn/AIF360 certification.")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "UNESCO fairness and gender lens", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, UNESCO_PRINCIPLE_10)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Deployment decision", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    status = "REVIEW REQUIRED — deployment blocked pending evidence, mitigation, and retest" if blockers else "EVIDENCE COMPLETE FOR HUMAN GOVERNANCE REVIEW"
    
    pdf.multi_cell(0, 5, pdf_safe_text(status), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    if blockers:
        pdf.ln(2)
        pdf.multi_cell(0, 5, pdf_safe_text("Open blockers:\n- " + "\n- ".join(blockers)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for title, content in [
        ("Stage 1 — Scope", st.session_state.scope_evidence),
        ("Stage 2 — Data evidence", st.session_state.data_evidence),
        ("Stage 3 — Model results", st.session_state.model_results),
        ("Stage 4 — Mitigation", st.session_state.mitigation_results),
        ("Stage 5 — Output audit", st.session_state.output_results),
        ("Stage 6 — Governance", st.session_state.governance_evidence),
    ]:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, pdf_safe_text(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 8)
        text = str(content)
        pdf.multi_cell(0, 4, pdf_safe_text(text[:5000]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    return bytes(pdf.output())


def render_report_stage():
    st.header("Stage 7 — Evidence Report and Deployment Decision")
    blockers = deployment_decision()
    if blockers:
        st.error("REVIEW REQUIRED: deployment blocked pending mitigation and retest.")
        st.write("Open evidence blockers:")
        for blocker in blockers:
            st.warning(blocker)
        st.session_state.deployment_status = "REVIEW REQUIRED"
    else:
        st.warning("EVIDENCE COMPLETE FOR HUMAN GOVERNANCE REVIEW — not automatic clinical approval.")
        st.session_state.deployment_status = "EVIDENCE COMPLETE FOR HUMAN REVIEW"
    if st.button("Generate audit report", key="generate_report"):
        st.session_state.audit_pdf = build_pdf(blockers)
        st.success("Evidence report generated.")
    if st.session_state.audit_pdf:
        st.download_button("Download gender-equity audit report", st.session_state.audit_pdf, "msrh_gender_equity_audit_report.pdf", "application/pdf")


# -----------------------------------------------------------------------------
# Application shell
# -----------------------------------------------------------------------------

init_state()
st.title("MSRH Gender Equity Auditor")
st.write("A seven-stage audit structured by UNESCO fairness and non-discrimination requirements, with Fairlearn- and AIF360-inspired metric approximations for gender-bias analysis.")
st.warning("Research and governance tool only. Do not use this application as an autonomous clinical decision-maker or as a certification authority.")

uploaded = st.file_uploader("Upload MSRH Q&A CSV", type=["csv"])
default_path = "test_patient_data.csv"
raw_df = None
if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
elif os.path.exists(default_path):
    raw_df = pd.read_csv(default_path)

if raw_df is None:
    st.info("Upload a CSV dataset to begin Stage 2 onward. You can still review the seven-stage structure below.")
    for i, stage in enumerate(STAGES, 1):
        st.write(f"{i}. {stage}")
else:
    st.session_state.dataset = raw_df
    columns = list(raw_df.columns)
    q_col = st.selectbox("Question column", columns, index=0)
    a_col = st.selectbox("Answer column", columns, index=min(1, len(columns) - 1))
    choices = nonempty_columns(raw_df)
    gender_choice = st.selectbox("Gender identity / gender audit field", choices, help="Use documented metadata where available; text inference is exploratory only.")
    age_choice = st.selectbox("Age or age-band field", choices)
    language_choice = st.selectbox("Language / locale field", choices)
    prepared = prepare_dataset(raw_df, q_col, a_col, selected_column(gender_choice), selected_column(age_choice), selected_column(language_choice))
    st.session_state.dataset = prepared

    tabs = st.tabs(STAGES)
    with tabs[0]:
        render_scope_stage()
    with tabs[1]:
        render_data_stage(prepared)
        if st.button("Save data audit snapshot", key="save_data_snapshot"):
            st.session_state.data_results = representation_table(prepared)
            st.success("Data audit snapshot saved.")
    with tabs[2]:
        render_model_stage(prepared)
    with tabs[3]:
        render_mitigation_stage(prepared)
    with tabs[4]:
        render_output_stage(prepared)
    with tabs[5]:
        render_governance_stage()
    with tabs[6]:
        render_report_stage()

st.markdown("---")
st.caption("Framework references: UNESCO Recommendation on the Ethics of Artificial Intelligence, Principle 10 and gender policy area; Fairlearn assessment and mitigation concepts; IBM AI Fairness 360 metric and mitigation concepts.")
