# -*- coding: utf-8 -*-
"""HASH MSRH Gender Equity Auditor (PyShiny UI)

Seven-stage gender-bias audit toolkit for Maternal, Sexual, and Reproductive
Health (MSRH) Q&A datasets and clinical AI systems.
"""

from datetime import datetime, timezone
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from shiny import App, reactive, render, ui

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
except ImportError:
    FPDF = None


# =============================================================================
# HASH Brand Assets & Screening Thresholds
# =============================================================================

HASH_LOGO_URL = "https://hash.theacademy.co.ug/wp-content/themes/openstrap/images/logo.png"

HASH_PARTNERS = [
    {
        "name": "Mak-AI",
        "url": "https://air.ug/",
        "logo": "https://air.ug/wp-content/uploads/2025/06/Mak-AI-06.png",
    },
    {
        "name": "Infectious Diseases Institute (IDI)",
        "url": "https://idi.mak.ac.ug/",
        "logo": "https://idi.mak.ac.ug/wp-content/uploads/2024/09/IDI-MAK-Logo.png",
    },
]

GENDER_EQUITY_PRINCIPLE = (
    "Clinical AI systems should promote health equity and safeguard fairness and "
    "non-discrimination across the entire lifecycle. Systems must avoid reinforcing "
    "discriminatory outcomes, engage affected communities, protect women, girls, and "
    "vulnerable groups, and provide clear mechanisms for oversight and remedy."
)

DATA_PROVENANCE_ITEMS = [
    "Collection purpose, consent, provenance, and permitted secondary use are documented.",
    "Representation is compared with the intended deployment population.",
    "Gender-diverse, adolescent, disabled, rural, multilingual, and marginalized groups are considered.",
    "Affected communities participated in data or system review.",
    "Cleaning and labeling procedures were checked for gender stereotypes and proxy harms.",
    "Small-cell privacy and disclosure risks are controlled.",
]

GOVERNANCE_ITEMS = [
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

THRESHOLDS = {
    "demographic_parity_difference": 0.10,
    "disparate_impact_ratio": 0.80,
    "equal_opportunity_difference": 0.10,
    "average_odds_difference": 0.10,
    "pronoun_consistency": 0.95,
    "max_false_negative_rate_gap": 0.10,
}

PRONOUN_MAP = {
    "he": "she", "she": "he",
    "him": "her", "her": "him",
    "his": "hers", "hers": "his",
    "himself": "herself", "herself": "himself",
}


# =============================================================================
# Core Audit & Analytics Functions
# =============================================================================

def safe_text(value):
    return "" if pd.isna(value) else str(value)


def nonempty_columns(df):
    return ["(not provided)"] + list(df.columns)


def selected_column(choice):
    return None if not choice or choice == "(not provided)" else choice


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
            return "Adult 20-34"
        if age < 50:
            return "Adult 35-49"
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
    if gender_col and gender_col in out.columns:
        out["Gender_Group"] = out[gender_col].map(normalize_gender)
        out["Gender_Basis"] = "declared metadata"
    else:
        out["Gender_Group"] = out["Question_Text"].map(infer_gender_from_text)
        out["Gender_Basis"] = "text inference - exploratory only"
    out["Age_Band"] = out[age_col].map(age_band) if (age_col and age_col in out.columns) else out["Question_Text"].map(age_band)
    out["Language_Group"] = value_or_unknown(out[language_col]) if (language_col and language_col in out.columns) else "Not provided"
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
    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 5),
        "status": "REVIEW REQUIRED" if p_value < 0.05 else "NO SIGNAL DETECTED",
    }


def build_binary_outcome(df, outcome_col=None):
    if outcome_col and outcome_col in df.columns:
        values = df[outcome_col]
        if values.nunique(dropna=True) == 2:
            codes, _ = pd.factorize(values)
            return pd.Series(codes, index=df.index).astype(int)
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
            "Group": group,
            "N": n,
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
        rows.append({
            "Metric": label,
            "Value": round(value, 4) if not np.isnan(value) else "-",
            "Screening rule": f"{direction}; threshold {threshold}",
            "Status": status,
        })
    return pd.DataFrame(rows)


def mitigation_variants(df):
    x = df["Question_Text"].str.lower()
    baseline = x.str.contains(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid", regex=True).astype(int).to_numpy()
    conservative = x.str.contains(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid|pain|bleeding", regex=True).astype(int).to_numpy()
    abstaining = baseline.copy()
    low_confidence = ~x.str.contains(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid", regex=True)
    abstaining[low_confidence] = 0
    return {
        "Baseline": baseline,
        "Conservative safety rule": conservative,
        "Abstention-aware rule": abstaining,
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


# =============================================================================
# Unicode-Safe PDF Evidence Report
# =============================================================================

class AuditPDF(FPDF if FPDF else object):
    if FPDF:
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 8, "HASH MSRH GENDER EQUITY AUDIT REPORT", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Hub for Artificial Intelligence in Maternal, Sexual and Reproductive Health", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")


def pdf_safe_text(value):
    replacements = {
        "—": "-", "–": "-", "‑": "-", "−": "-",
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "…": "...", "•": "-", "→": "->", "×": "x",
        "≤": "<=", "≥": ">=", "χ": "chi", "²": "2",
    }
    text = str(value)
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_pdf(blockers, scope_ev, data_ev, model_res, mit_res, out_res, gov_ev):
    if FPDF is None:
        return None
    pdf = AuditPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0, 5,
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
        "This report is an internal audit evidence summary and does not constitute clinical validation or regulatory approval."
    )
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Guiding Equity Principle", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, GENDER_EQUITY_PRINCIPLE)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Deployment Decision", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    status = "REVIEW REQUIRED - deployment blocked pending evidence, mitigation, and retest" if blockers else "EVIDENCE COMPLETE FOR HUMAN GOVERNANCE REVIEW"
    pdf.multi_cell(0, 5, pdf_safe_text(status), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if blockers:
        pdf.ln(2)
        pdf.multi_cell(0, 5, pdf_safe_text("Open blockers:\n- " + "\n- ".join(blockers)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for title, content in [
        ("Stage 1 - Scope", scope_ev),
        ("Stage 2 - Data evidence", data_ev),
        ("Stage 3 - Model results", model_res),
        ("Stage 4 - Mitigation", mit_res),
        ("Stage 5 - Output audit", out_res),
        ("Stage 6 - Governance", gov_ev),
    ]:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, pdf_safe_text(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(0, 4, pdf_safe_text(str(content)[:5000]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    return bytes(pdf.output())


# =============================================================================
# Native PyShiny UI (No Custom CSS, No Gradients, No Pill Badges)
# =============================================================================

def partners_section_ui():
    partner_cols = [
        ui.a(
            ui.card(
                ui.img(src=p["logo"], alt=p["name"], title=p["name"], height="60px", style="object-fit: contain;"),
                class_="text-center p-3 h-100 d-flex align-items-center justify-content-center",
            ),
            href=p["url"],
            target="_blank",
            rel="noopener noreferrer",
            title=p["name"],
            class_="text-decoration-none",
        )
        for p in HASH_PARTNERS
    ]
    return ui.card(
        ui.card_header("Partners"),
        ui.layout_columns(*partner_cols, col_widths=(6, 6)),
        class_="mt-4",
    )


def overview_tab_ui():
    return ui.nav_panel(
        "Overview",
        ui.card(
            ui.card_header("About the MSRH Gender Bias Measurement Tool"),
            ui.p(
                "This toolkit helps AI developers, clinical researchers, and health governance teams evaluate "
                "Maternal, Sexual, and Reproductive Health (MSRH) question-answering models and datasets for "
                "gender representation gaps, subgroup disparities, mitigation trade-offs, and clinical safety."
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Who It Is For"),
                    ui.p(
                        "Built for AI developers, clinical researchers, digital health implementers, and "
                        "governance teams auditing AI triage, health-information, or decision-support tools before deployment.",
                        class_="mb-0",
                    ),
                ),
                ui.card(
                    ui.card_header("Expected CSV Format"),
                    ui.p(
                        "Upload a CSV containing at least a Question/Prompt column and an Answer/Response column. "
                        "Optional Gender and validated Outcome columns enable subgroup parity and mitigation testing.",
                        class_="mb-0",
                    ),
                ),
                col_widths=(6, 6),
            ),
            ui.h5("How the 7-Stage Audit Works", class_="mt-3 mb-2"),
            ui.tags.ol(
                ui.tags.li(ui.strong("Stage 1 — Scope & Gender Impact: "), "Document intended use, prohibited decisions, and gender risks."),
                ui.tags.li(ui.strong("Stage 2 — Data Representation: "), "Inspect gender balance, response completeness, safety-netting, and provenance."),
                ui.tags.li(ui.strong("Stage 3 — Model Fairness: "), "Measure subgroup selection rates, recall, false-negative rates, and parity gaps."),
                ui.tags.li(ui.strong("Stage 4 — Mitigation & Trade-offs: "), "Compare mitigation strategies and accuracy vs. fairness trade-offs."),
                ui.tags.li(ui.strong("Stage 5 — Output Equity: "), "Run pronoun-swap counterfactual checks and clinical safety verification."),
                ui.tags.li(ui.strong("Stage 6 — Governance & Monitoring: "), "Record human oversight, complaint channels, and post-deployment monitoring."),
                ui.tags.li(ui.strong("Stage 7 — Audit Report: "), "Review readiness blockers and export a structured PDF audit report."),
            ),
            ui.div(
                ui.input_action_button("start_audit", "Start Audit (Go to Stage 1)", class_="btn-primary me-2"),
                ui.input_action_button("load_demo_overview", "Load Sample Demo & Start", class_="btn-outline-primary"),
                class_="mt-3",
            ),
            class_="mt-3",
        ),
    )


app_ui = ui.page_fixed(
    ui.tags.head(ui.tags.title("MSRH Gender Bias Measurement Tool")),
    # Simple Native Header with HASH Logo
    ui.div(
        ui.a(
            ui.img(src=HASH_LOGO_URL, alt="HASH Project Logo", height="48px"),
            href="https://hash.theacademy.co.ug/",
            target="_blank",
        ),
        ui.div(
            ui.h2("MSRH Gender Bias Measurement Tool", class_="mb-0"),
            ui.p(
                "Hub for Artificial Intelligence in Maternal, Sexual and Reproductive Health (HASH)",
                class_="text-muted mb-0",
            ),
        ),
        class_="d-flex align-items-center gap-3 py-3 border-bottom mb-3",
    ),
    ui.div(
        "Research and governance tool only. Do not use this application as an autonomous clinical decision-maker.",
        class_="alert alert-warning",
    ),
    # Dataset Upload Card
    ui.card(
        ui.card_header("Dataset Ingestion"),
        ui.layout_columns(
            ui.input_file("csv_upload", "Upload MSRH Q&A CSV", accept=[".csv"], width="100%"),
            ui.div(
                ui.tags.label("Sample Data", class_="control-label d-block mb-2"),
                ui.div(
                    ui.input_action_button("load_demo", "Load Sample Demo Dataset", class_="btn-primary"),
                    ui.input_action_button("clear_demo", "Clear Dataset", class_="btn-outline-secondary"),
                    class_="d-flex flex-wrap gap-2",
                ),
            ),
            col_widths=(7, 5),
        ),
        ui.output_ui("dataset_banner"),
        ui.output_ui("column_mapper_ui"),
    ),
    # Overview + 7-Stage Navigation Tabs
    ui.navset_tab(
        overview_tab_ui(),
        ui.nav_panel(
            "1. Scope",
            ui.card(
                ui.card_header("Stage 1 - Scope and Gender Impact"),
                ui.p("Define intended purpose, beneficiaries, gender risks, and affected groups before measuring model fairness.", class_="text-muted"),
                ui.div(GENDER_EQUITY_PRINCIPLE, class_="alert alert-info"),
                ui.input_text_area("scope_purpose", "Intended purpose", placeholder="Example: support MSRH triage and referral; not autonomous diagnosis.", width="100%"),
                ui.input_text_area("scope_beneficiaries", "Intended beneficiaries and affected communities", width="100%"),
                ui.input_text_area("scope_prohibited", "Prohibited uses and high-risk decisions", width="100%"),
                ui.input_text_area("scope_risks", "Gender and intersectional risks identified", width="100%"),
                ui.input_checkbox("scope_human", "A qualified human retains final determination for high-impact or life-affecting decisions."),
                ui.input_checkbox("scope_gia", "A gender-responsive ethical impact assessment has been completed."),
                ui.input_checkbox("scope_community", "Affected women, girls, and gender-diverse communities participated in design or review."),
                ui.input_action_button("save_stage1", "Save Stage 1 evidence", class_="btn-primary"),
                ui.output_ui("stage1_status"),
                class_="mt-3",
            ),
        ),
        ui.nav_panel(
            "2. Data",
            ui.card(
                ui.card_header("Stage 2 - Data Representation, Participation, and Provenance"),
                ui.p("Evaluate subgroup representation, response completeness, safety-netting, and data provenance.", class_="text-muted"),
                ui.output_ui("stage2_content"),
                class_="mt-3",
            ),
        ),
        ui.nav_panel(
            "3. Fairness",
            ui.card(
                ui.card_header("Stage 3 - Model Fairness Measurement"),
                ui.p("Disaggregated subgroup metrics and group parity screening.", class_="text-muted"),
                ui.output_ui("stage3_content"),
                class_="mt-3",
            ),
        ),
        ui.nav_panel(
            "4. Mitigation",
            ui.card(
                ui.card_header("Stage 4 - Mitigation, Retesting, and Trade-offs"),
                ui.p("Compare mitigation variants and review accuracy vs. fairness trade-offs.", class_="text-muted"),
                ui.output_ui("stage4_content"),
                class_="mt-3",
            ),
        ),
        ui.nav_panel(
            "5. Output Equity",
            ui.card(
                ui.card_header("Stage 5 - Output Equity, Counterfactuals, and Clinical Safety"),
                ui.p("Test whether pronoun swaps alter triage decisions and verify clinical safety safeguards.", class_="text-muted"),
                ui.output_ui("stage5_content"),
                class_="mt-3",
            ),
        ),
        ui.nav_panel(
            "6. Governance",
            ui.card(
                ui.card_header("Stage 6 - Governance, Remedy, and Continuous Monitoring"),
                ui.p("Document human oversight, complaint mechanisms, and ongoing equity monitoring.", class_="text-muted"),
                ui.input_checkbox_group("gov_checks", "Governance, remedy, and monitoring controls:", choices=GOVERNANCE_ITEMS, width="100%"),
                ui.input_action_button("save_stage6", "Save governance evidence", class_="btn-primary"),
                ui.output_ui("stage6_status"),
                class_="mt-3",
            ),
        ),
        ui.nav_panel(
            "7. Report",
            ui.card(
                ui.card_header("Stage 7 - Evidence Report and Deployment Decision"),
                ui.output_ui("stage7_content"),
                ui.download_button("download_pdf", "Download gender-equity audit report (PDF)", class_="btn-primary mt-2"),
                class_="mt-3",
            ),
        ),
        id="main_tabs",
    ),
    partners_section_ui(),
    ui.div(
        ui.p(
            "HASH Project • Academy for Health Innovation in Uganda • ",
            ui.a("hash.theacademy.co.ug", href="https://hash.theacademy.co.ug/", target="_blank"),
            class_="text-muted small text-center my-3",
        ),
    ),
)


def server(input, output, session):
    raw_df_val = reactive.Value(None)
    demo_mode = reactive.Value(False)

    scope_evidence = reactive.Value({})
    data_evidence = reactive.Value({})
    data_results = reactive.Value(None)
    model_results = reactive.Value(None)
    mitigation_results = reactive.Value(None)
    output_results = reactive.Value(None)
    governance_evidence = reactive.Value({})

    @reactive.effect
    @reactive.event(input.csv_upload)
    def _on_upload():
        file_infos = input.csv_upload()
        if file_infos:
            df = pd.read_csv(file_infos[0]["datapath"])
            raw_df_val.set(df)
            demo_mode.set(False)
            data_results.set(None)
            model_results.set(None)
            mitigation_results.set(None)
            output_results.set(None)

    @reactive.effect
    @reactive.event(input.load_demo)
    def _on_load_demo():
        default_path = "test_patient_data.csv"
        if os.path.exists(default_path):
            raw_df_val.set(pd.read_csv(default_path))
            demo_mode.set(True)
            data_results.set(None)
            model_results.set(None)
            mitigation_results.set(None)
            output_results.set(None)

    @reactive.effect
    @reactive.event(input.start_audit)
    def _on_start_audit():
        ui.update_navs("main_tabs", selected="1. Scope")

    @reactive.effect
    @reactive.event(input.load_demo_overview)
    def _on_load_demo_overview():
        default_path = "test_patient_data.csv"
        if os.path.exists(default_path):
            raw_df_val.set(pd.read_csv(default_path))
            demo_mode.set(True)
            data_results.set(None)
            model_results.set(None)
            mitigation_results.set(None)
            output_results.set(None)
        ui.update_navs("main_tabs", selected="1. Scope")

    @reactive.effect
    @reactive.event(input.clear_demo)
    def _on_clear():
        raw_df_val.set(None)
        demo_mode.set(False)
        data_results.set(None)
        model_results.set(None)
        mitigation_results.set(None)
        output_results.set(None)

    @render.ui
    def dataset_banner():
        df = raw_df_val.get()
        if df is None:
            return ui.div(
                "No dataset loaded yet. Upload a CSV file above or click Load Sample Demo Dataset to begin Stages 2-5.",
                class_="alert alert-info mt-3 mb-0",
            )
        if demo_mode.get():
            return ui.div(
                f"Demo Mode active using test_patient_data.csv ({len(df)} records).",
                class_="alert alert-info mt-3 mb-0",
            )
        return ui.div(
            f"Loaded uploaded dataset ({len(df)} records, {len(df.columns)} columns).",
            class_="alert alert-success mt-3 mb-0",
        )

    @render.ui
    def column_mapper_ui():
        df = raw_df_val.get()
        if df is None:
            return None
        cols = list(df.columns)
        choices = nonempty_columns(df)
        return ui.div(
            ui.hr(),
            ui.layout_columns(
                ui.input_select("q_col", "Question column", choices=cols, selected=cols[0]),
                ui.input_select("a_col", "Answer column", choices=cols, selected=cols[min(1, len(cols) - 1)]),
                ui.input_select("gender_col", "Gender identity / audit field", choices=choices),
                ui.input_select("age_col", "Age or age-band field", choices=choices),
                ui.input_select("lang_col", "Language / locale field", choices=choices),
                col_widths=(3, 3, 2, 2, 2),
            ),
        )

    @reactive.calc
    def prepared_df():
        df = raw_df_val.get()
        if df is None:
            return None
        q_col = input.q_col() if "q_col" in input else df.columns[0]
        a_col = input.a_col() if "a_col" in input else df.columns[min(1, len(df.columns) - 1)]
        if q_col not in df.columns or a_col not in df.columns:
            return None
        g_col = selected_column(input.gender_col()) if "gender_col" in input else None
        ag_col = selected_column(input.age_col()) if "age_col" in input else None
        l_col = selected_column(input.lang_col()) if "lang_col" in input else None
        return prepare_dataset(df, q_col, a_col, g_col, ag_col, l_col)

    # --- Stage 1 ---
    @reactive.effect
    @reactive.event(input.save_stage1)
    def _save_stage1():
        scope_evidence.set({
            "purpose": input.scope_purpose(),
            "beneficiaries": input.scope_beneficiaries(),
            "prohibited": input.scope_prohibited(),
            "gender_risks": input.scope_risks(),
            "human_decision": input.scope_human(),
            "gender_impact": input.scope_gia(),
            "community": input.scope_community(),
        })

    @render.ui
    def stage1_status():
        ev = scope_evidence.get()
        if not ev:
            return None
        missing = [
            label for label, val in {
                "intended purpose": ev.get("purpose"),
                "beneficiaries": ev.get("beneficiaries"),
                "gender risks": ev.get("gender_risks"),
                "human final determination": ev.get("human_decision"),
                "gender impact assessment": ev.get("gender_impact"),
                "community participation": ev.get("community"),
            }.items() if not val
        ]
        if missing:
            return ui.div("Scope evidence incomplete: " + ", ".join(missing), class_="alert alert-warning mt-3 mb-0")
        return ui.div("Stage 1 evidence is complete for review.", class_="alert alert-success mt-3 mb-0")

    # --- Stage 2 ---
    @render.ui
    def stage2_content():
        df = prepared_df()
        if df is None:
            return ui.div("Upload a dataset or load the demo dataset above to inspect representation and provenance.", class_="alert alert-secondary")
        basis = df["Gender_Basis"].iloc[0] if len(df) else "not available"
        basis_alert = (
            ui.div("Gender groups are inferred from text. This is exploratory only and cannot establish deployment evidence.", class_="alert alert-warning")
            if basis != "declared metadata"
            else ui.div("Gender groups use a declared metadata field. Verify consent, provenance, and permitted use.", class_="alert alert-success")
        )
        rep = representation_table(df)
        gap = chi_square_gap(df, "Gender_Group", "Detailed_Response")

        ev = data_evidence.get()
        saved_snap = data_results.get()
        return ui.div(
            basis_alert,
            ui.HTML(rep.to_html(classes="table table-sm table-striped table-hover align-middle", index=False)),
            ui.p("Small groups are not evidence of fairness. They are evidence of limited measurement power.", class_="text-muted small"),
            ui.div(
                ui.tags.strong("Response-completeness diagnostic: "),
                f"chi2 = {gap['chi2']}, p = {gap['p_value']} - ",
                ui.tags.strong(gap["status"]),
                class_="alert alert-light border",
            ),
            ui.tags.details(
                ui.tags.summary("Data, participation, and provenance checklist", class_="fw-semibold mb-2"),
                ui.input_checkbox_group("data_checks", None, choices=DATA_PROVENANCE_ITEMS, width="100%"),
                ui.input_action_button("save_stage2_ev", "Save Stage 2 checklist", class_="btn-primary btn-sm mt-1"),
                ui.div(f"Saved {sum(ev.values())}/{len(DATA_PROVENANCE_ITEMS)} items.", class_="text-success small mt-2") if ev else None,
            ),
            ui.div(
                ui.input_action_button("save_data_snapshot", "Save data audit snapshot", class_="btn-primary mt-3"),
                ui.span(" Snapshot saved for Stage 7.", class_="text-success fw-semibold ms-2") if saved_snap is not None else None,
            ),
        )

    @reactive.effect
    @reactive.event(input.save_stage2_ev)
    def _save_stage2_ev():
        selected = set(input.data_checks() or [])
        data_evidence.set({item: (item in selected) for item in DATA_PROVENANCE_ITEMS})

    @reactive.effect
    @reactive.event(input.save_data_snapshot)
    def _save_data_snap():
        df = prepared_df()
        if df is not None:
            data_results.set(representation_table(df))

    # --- Stage 3 ---
    @render.ui
    def stage3_content():
        df = prepared_df()
        if df is None:
            return ui.div("Upload a dataset or load the demo dataset above to measure baseline fairness.", class_="alert alert-secondary")
        orig_df = raw_df_val.get()
        outcome_options = ["(question-text proxy)"] + list(orig_df.columns)
        res = model_results.get()
        selected_outcome = res["outcome_col"] if (res and res.get("outcome_col")) else "(question-text proxy)"

        blocks = [
            ui.input_select("model_outcome", "Validated binary outcome column (optional)", choices=outcome_options, selected=selected_outcome),
            ui.input_action_button("run_model", "Run baseline measurement", class_="btn-primary mb-3"),
        ]
        if res is not None:
            metric_tbl = render_metric_table(res["summary"])
            blocks.extend([
                ui.h5("Disaggregated performance"),
                ui.HTML(res["group_metrics"].round(4).to_html(classes="table table-sm table-striped align-middle", index=False)),
                ui.h5("Fairness metric summary", class_="mt-3"),
                ui.HTML(metric_tbl.to_html(classes="table table-sm table-bordered align-middle", index=False)),
                ui.div(
                    "These screening metrics require validated labels, model predictions, subgroup counts, and uncertainty analysis before clinical use.",
                    class_="alert alert-info mt-2",
                ),
            ])
            if any(v == "REVIEW REQUIRED" for v in metric_tbl["Status"]):
                blocks.append(ui.div("Baseline model: review required. Do not interpret aggregate accuracy as evidence of gender fairness.", class_="alert alert-danger"))
        return ui.div(*blocks)

    @reactive.effect
    @reactive.event(input.run_model)
    def _run_model():
        df = prepared_df()
        if df is None:
            return
        choice = input.model_outcome()
        outcome_col = None if choice == "(question-text proxy)" else choice
        y_true = build_binary_outcome(df, outcome_col)
        x = df["Question_Text"].str.lower()
        y_pred = x.str.contains(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid", regex=True).astype(int)
        gm = paired_group_metrics(y_true, y_pred, df["Gender_Group"])
        summary = fairness_summary(gm)
        model_results.set({"group_metrics": gm, "summary": summary, "outcome_col": outcome_col})

    # --- Stage 4 ---
    @render.ui
    def stage4_content():
        df = prepared_df()
        if df is None:
            return ui.div("Upload a dataset or load the demo dataset above to compare mitigation variants.", class_="alert alert-secondary")
        y_true = build_binary_outcome(df)
        variants = mitigation_variants(df)
        rows = []
        for name, pred in variants.items():
            gm = paired_group_metrics(y_true, pred, df["Gender_Group"])
            summary = fairness_summary(gm)
            accuracy = float((np.asarray(y_true) == pred).mean())
            rows.append({"Variant": name, "Accuracy": accuracy, **summary})
        table = pd.DataFrame(rows).round(4)
        mit_saved = mitigation_results.get()
        return ui.div(
            ui.HTML(table.to_html(classes="table table-sm table-striped align-middle", index=False)),
            ui.div(
                "A mitigation is not automatically acceptable because one fairness metric improves. Review false negatives, calibration, clinical utility, subgroup size, and operational feasibility.",
                class_="alert alert-warning mt-2",
            ),
            ui.input_select("selected_variant", "Select variant for documented human review", choices=list(variants.keys())),
            ui.input_text_area("mitigation_rationale", "Document mitigation rationale, trade-offs, and clinical review", width="100%"),
            ui.input_action_button("save_mitigation", "Save mitigation review", class_="btn-primary mt-1"),
            ui.div(f"Saved mitigation review for '{mit_saved['selected']}'. This does not approve deployment.", class_="alert alert-success mt-2 mb-0") if mit_saved else None,
        )

    @reactive.effect
    @reactive.event(input.save_mitigation)
    def _save_mit():
        df = prepared_df()
        if df is None:
            return
        y_true = build_binary_outcome(df)
        variants = mitigation_variants(df)
        rows = []
        for name, pred in variants.items():
            gm = paired_group_metrics(y_true, pred, df["Gender_Group"])
            summary = fairness_summary(gm)
            accuracy = float((np.asarray(y_true) == pred).mean())
            rows.append({"Variant": name, "Accuracy": accuracy, **summary})
        mitigation_results.set({
            "table": pd.DataFrame(rows),
            "selected": input.selected_variant(),
            "rationale": input.mitigation_rationale(),
        })

    # --- Stage 5 ---
    @render.ui
    def stage5_content():
        df = prepared_df()
        if df is None:
            return ui.div("Upload a dataset or load the demo dataset above to run counterfactual testing.", class_="alert alert-secondary")
        res = output_results.get()
        blocks = [
            ui.p("Test types should be separated: pronoun-only, gender identity, age, pregnancy status, language, and intersectional pairs. The default test changes pronouns only."),
            ui.input_action_button("run_output", "Run pronoun-only paired test", class_="btn-primary mb-3"),
        ]
        if res is not None:
            rate = res["consistency"]
            blocks.append(
                ui.value_box(
                    "Pronoun-only decision consistency",
                    f"{rate:.1%}",
                    theme="success" if rate >= THRESHOLDS["pronoun_consistency"] else "danger",
                )
            )
            if rate < THRESHOLDS["pronoun_consistency"]:
                blocks.append(ui.div("Review required: pronoun-only changes altered decisions.", class_="alert alert-danger mt-2"))
            else:
                blocks.append(ui.div("No decision change observed in the pronoun-only sample; continue with clinical safety and intersectional tests.", class_="alert alert-success mt-2"))
            blocks.append(ui.HTML(res["records"].to_html(classes="table table-sm table-striped align-middle", index=False)))
            blocks.append(ui.div("For MSRH deployment, add validated tests for missed emergencies, unsafe advice, stigmatizing language, safeguarding, adolescent confidentiality, and referral equity.", class_="alert alert-warning mt-2"))
        return ui.div(*blocks)

    @reactive.effect
    @reactive.event(input.run_output)
    def _run_output():
        df = prepared_df()
        if df is None:
            return
        records = []
        for text in df["Question_Text"]:
            cf = pronoun_only_counterfactual(text)
            original = int(bool(re.search(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid", text.lower())))
            counterfactual = int(bool(re.search(r"severe|excruciating|heavy bleeding|faint|emergency|ectopic|assault|suicid", cf.lower())))
            records.append({
                "Original": text,
                "Pronoun-only pair": cf,
                "Original decision": original,
                "Counterfactual decision": counterfactual,
                "Stable": original == counterfactual,
            })
        result = pd.DataFrame(records)
        rate = float(result["Stable"].mean()) if len(result) else 0.0
        output_results.set({"records": result, "consistency": rate})

    # --- Stage 6 ---
    @reactive.effect
    @reactive.event(input.save_stage6)
    def _save_stage6():
        selected = set(input.gov_checks() or [])
        governance_evidence.set({item: (item in selected) for item in GOVERNANCE_ITEMS})

    @render.ui
    def stage6_status():
        gov = governance_evidence.get()
        count = sum(gov.values()) if gov else 0
        pct = int((count / len(GOVERNANCE_ITEMS)) * 100)
        return ui.div(
            ui.div(
                ui.div(
                    class_="progress-bar",
                    role="progressbar",
                    style=f"width: {pct}%;",
                    **{"aria-valuenow": str(pct), "aria-valuemin": "0", "aria-valuemax": "100"},
                ),
                class_="progress mt-3",
            ),
            ui.p(f"Governance evidence: {count}/{len(GOVERNANCE_ITEMS)} controls documented.", class_="mt-2 fw-semibold"),
        )

    # --- Stage 7 ---
    def compute_blockers():
        blockers = []
        scope = scope_evidence.get()
        if not scope or not scope.get("purpose") or not scope.get("gender_risks"):
            blockers.append("Scope or gender-impact evidence incomplete")
        if not scope or not scope.get("gender_impact", False):
            blockers.append("Gender-responsive ethical impact assessment not evidenced")
        if data_results.get() is None:
            blockers.append("Data representation audit not run")
        mod = model_results.get()
        if mod is None:
            blockers.append("Baseline fairness measurement not run")
        else:
            statuses = render_metric_table(mod["summary"])["Status"].tolist()
            if "REVIEW REQUIRED" in statuses or "INSUFFICIENT EVIDENCE" in statuses:
                blockers.append("Critical fairness metric requires review or has insufficient evidence")
        out = output_results.get()
        if out is None:
            blockers.append("Output counterfactual audit not run")
        elif out["consistency"] < THRESHOLDS["pronoun_consistency"]:
            blockers.append("Pronoun-only counterfactual inconsistency")
        if mitigation_results.get() is None:
            blockers.append("Mitigation and trade-off review not documented")
        gov = governance_evidence.get()
        if len(gov) < len(GOVERNANCE_ITEMS) or not all(gov.values()):
            blockers.append("Governance, remedy, and monitoring controls incomplete")
        return blockers

    @render.ui
    def stage7_content():
        blockers = compute_blockers()
        if blockers:
            items = [ui.div(b, class_="alert alert-warning py-2 mb-2") for b in blockers]
            return ui.div(
                ui.div("REVIEW REQUIRED: deployment blocked pending mitigation and retest.", class_="alert alert-danger fw-bold"),
                ui.p("Open evidence blockers:", class_="fw-semibold"),
                *items,
            )
        return ui.div(
            "EVIDENCE COMPLETE FOR HUMAN GOVERNANCE REVIEW - not automatic clinical approval.",
            class_="alert alert-success fw-bold",
        )

    @render.download(filename="msrh_gender_equity_audit_report.pdf")
    def download_pdf():
        blockers = compute_blockers()
        pdf_bytes = build_pdf(
            blockers,
            scope_evidence.get(),
            data_results.get(),
            model_results.get(),
            mitigation_results.get(),
            output_results.get(),
            governance_evidence.get(),
        )
        if pdf_bytes:
            yield pdf_bytes


app = App(app_ui, server)
