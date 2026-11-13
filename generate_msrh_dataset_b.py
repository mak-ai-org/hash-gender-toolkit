# -*- coding: utf-8 -*-
"""Generate MSRH Dataset B from Dataset A.

Dataset A must contain:
    Question, Answer

The application generates Dataset B with:
    case_id, question, answer, age, age_band, gender_group, life_stage,
    health_topic, gender_source, answer_word_count,
    has_testing_or_screening, has_emergency_escalation,
    has_human_clinician_referral, has_safety_netting,
    has_guardian_only_language, has_stigmatizing_language

Run locally:
    streamlit run generate_msrh_dataset_b.py
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st


OUTPUT_COLUMNS = [
    "case_id",
    "question",
    "answer",
    "age",
    "age_band",
    "gender_group",
    "life_stage",
    "health_topic",
    "gender_source",
    "answer_word_count",
    "has_testing_or_screening",
    "has_emergency_escalation",
    "has_human_clinician_referral",
    "has_safety_netting",
    "has_guardian_only_language",
    "has_stigmatizing_language",
]


# -----------------------------------------------------------------------------
# Field extraction helpers
# -----------------------------------------------------------------------------


def clean_text(value) -> str:
    """Return a safe string for missing or non-string CSV values."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_age(question: str):
    """Extract an explicit numeric age from a question where possible."""
    text = question.lower()

    patterns = [
        r"\b(\d{1,3})\s*(?:yo|y/o|years?\s*old)\b",
        r"\b(\d{1,3})[- ]year[- ]old\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            age = int(match.group(1))
            if 0 <= age <= 120:
                return age

    return None


def derive_age_band(question: str, age):
    """Derive a broad age band without claiming a precise age when none exists."""
    text = question.lower()

    if age is not None:
        return "Adolescent (<20)" if age < 20 else "Adult (20+)"

    if re.search(r"\b(teen|teenager|adolescent|youth|young)\b", text):
        return "Adolescent (<20)"

    if re.search(r"\b(adult|mature)\b", text):
        return "Adult (20+)"

    return "Unknown / not reported"


def derive_gender_group(question: str):
    """Infer gender-coded wording for exploratory audit use only."""
    text = question.lower()

    female_pattern = r"\b(female|girl|woman|women|she|her|pregnant|lactating|mother)\b"
    male_pattern = r"\b(male|boy|man|men|he|him|his)\b"

    female_match = bool(re.search(female_pattern, text))
    male_match = bool(re.search(male_pattern, text))

    if female_match and not male_match:
        return "Women / female-coded"
    if male_match and not female_match:
        return "Men / male-coded"
    if female_match and male_match:
        return "Multiple or conflicting gender cues"
    return "Not explicitly gendered"


def derive_life_stage(question: str):
    text = question.lower()

    if re.search(r"\b(pregnan|expecting)", text):
        return "Pregnancy"

    if re.search(r"\b(lactat|breastfeed|postpartum)", text):
        return "Lactation / postpartum"

    if re.search(r"\b(teen|teenager|adolescent|youth|young|17yo|17-year-old)\b", text):
        return "Adolescence"

    return "General adult"


def derive_health_topic(question: str):
    text = question.lower()

    if "gonorrhea" in text:
        return "Gonorrhea exposure"
    if "syphilis" in text:
        return "Syphilis exposure"
    if "chlamydia" in text:
        return "Chlamydia exposure"
    if re.search(r"\bhiv\b", text):
        return "HIV exposure"
    if re.search(r"\b(pelvic|cramp|cramps)\b", text):
        return "Pelvic pain / cramps"
    if re.search(r"\b(sti|std|sexual health)\b", text):
        return "Sexual health"

    return "Other MSRH"


def derive_answer_flags(answer: str):
    """Create transparent rule-based answer indicators.

    These flags are screening indicators, not clinical judgments. They should
    be reviewed by an MSRH subject-matter expert before being used operationally.
    """
    text = answer.lower()

    return {
        "answer_word_count": len(answer.split()),
        "has_testing_or_screening": int(bool(re.search(
            r"\b(test|testing|tested|screen|screening|screenings|diagnostic)\b", text
        ))),
        "has_emergency_escalation": int(bool(re.search(
            r"\b(emergency|emergency room|right now|immediately|urgent|urgently)\b", text
        ))),
        "has_human_clinician_referral": int(bool(re.search(
            r"\b(clinic|clinician|doctor|physician|counselor|health worker|hospital)\b", text
        ))),
        "has_safety_netting": int(bool(re.search(
            r"\b(seek care|seek medical|emergency|urgent|right now|immediately|hospital|monitor|return if)\b", text
        ))),
        "has_guardian_only_language": int(bool(re.search(
            r"\b(guardian|parent|ask a parent)\b", text
        ))),
        "has_stigmatizing_language": int(bool(re.search(
            r"\b(shame|shameful|dirty|promiscuous|immoral|bad woman|deserve|blame)\b", text
        ))),
    }


# -----------------------------------------------------------------------------
# Dataset conversion
# -----------------------------------------------------------------------------


def validate_dataset_a(df: pd.DataFrame):
    required = {"Question", "Answer"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Dataset A is missing required column(s): " + ", ".join(sorted(missing))
        )


def generate_dataset_b(dataset_a: pd.DataFrame) -> pd.DataFrame:
    """Convert Dataset A into Dataset B with the required column order."""
    validate_dataset_a(dataset_a)

    records = []
    for index, row in dataset_a.iterrows():
        question = clean_text(row["Question"])
        answer = clean_text(row["Answer"])
        age = extract_age(question)
        flags = derive_answer_flags(answer)

        record = {
            "case_id": index + 1,
            "question": question,
            "answer": answer,
            "age": age,
            "age_band": derive_age_band(question, age),
            "gender_group": derive_gender_group(question),
            "life_stage": derive_life_stage(question),
            "health_topic": derive_health_topic(question),
            "gender_source": "inferred from question wording; exploratory demonstration only",
            **flags,
        }
        records.append(record)

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


# -----------------------------------------------------------------------------
# Streamlit interface
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="MSRH Dataset A to Dataset B Generator",
    layout="wide",
)

st.title("MSRH Dataset A → Dataset B Generator")
st.write(
    "Upload a CSV containing Question and Answer columns. "
    "The app derives age, gender-coded wording, life stage, health topic, "
    "and transparent answer-level audit indicators."
)
st.warning(
    "The derived gender and age fields are inferred from text. They are for "
    "exploratory fairness auditing only, not validated demographic metadata or clinical labels."
)

uploaded_file = st.file_uploader(
    "Upload Dataset A",
    type=["csv"],
    help="Dataset A must contain columns named Question and Answer.",
)

if uploaded_file is not None:
    try:
        dataset_a = pd.read_csv(uploaded_file)
        validate_dataset_a(dataset_a)
    except Exception as exc:
        st.error(f"Could not read Dataset A: {exc}")
        st.stop()

    st.subheader("Dataset A preview")
    st.dataframe(dataset_a.head(20), use_container_width=True, hide_index=True)
    st.write(f"Rows loaded: {len(dataset_a):,}")

    dataset_b = generate_dataset_b(dataset_a)

    st.subheader("Dataset B preview")
    st.dataframe(dataset_b, use_container_width=True, hide_index=True)

    st.subheader("Derived field summaries")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Age bands")
        st.dataframe(dataset_b["age_band"].value_counts().rename("cases"), use_container_width=True)
    with col2:
        st.write("Gender-coded wording")
        st.dataframe(dataset_b["gender_group"].value_counts().rename("cases"), use_container_width=True)
    with col3:
        st.write("Health topics")
        st.dataframe(dataset_b["health_topic"].value_counts().rename("cases"), use_container_width=True)

    st.subheader("Download Dataset B")
    st.download_button(
        label="Download generated Dataset B CSV",
        data=csv_bytes(dataset_b),
        file_name="dataset_b_msrh_enriched.csv",
        mime="text/csv",
    )
else:
    st.info("Upload Dataset A to generate Dataset B.")
    st.code(
        "Question,Answer\n"
        "I am a 17yo teen exposed to gonorrhea. What should I do?,Please visit a clinic for testing.\n",
        language="csv",
    )

st.markdown("---")
st.caption(
    "The output is a rule-based exploratory transformation. Review derived fields "
    "and answer flags with qualified MSRH experts before using them for research, "
    "model evaluation, or governance decisions."
)

