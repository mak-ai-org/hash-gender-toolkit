# MSRH Algorithmic Gender Equity Lifecycle Auditor

An interactive auditing toolkit for Maternal, Sexual, and Reproductive Health (MSRH) datasets and clinical AI models. This application helps researchers, data scientists, and healthcare compliance officers audit datasets for systemic demographic bias, mitigate gender feature leakage during model training, and evaluate output consistency across counterfactual pronoun swaps.

---

##  Key Features

The toolkit guides users through a comprehensive **three-phase algorithmic lifecycle audit**:

### 1. Phase 1: Dataset Evaluation (Pre-training Audit)
- **Demographic Cohort Extraction**: Automatically identifies target groups (e.g., adolescent female, maternal female, lactating female, adult male) and clinical categories from unstructured health queries.
- **Differential Completeness & Missingness Testing**: Employs Chi-Square tests of independence (`scipy.stats.chi2_contingency`) to detect systematic disparities in answer volume and descriptive detail across demographic groups.
- **Pre-training Self-Assessment Checklist**: Evaluates problem definition, representation balance, data quality, and sourcing transparency.
- **Dynamic Bias Justifications**: Generates human-readable statistical justifications and actionable recommendations for detected disparities.

### 2. Phase 2: Bias Mitigation during Training (In-processing Audit)
- **Feature Projection Penalty**: Tunes a penalty weight ($\lambda$) to project clinical text representations onto the nullspace of demographic bias directions.
- **Standardized Fairness Benchmarks**: Real-time evaluation aligned with Fairlearn and IBM AI Fairness 360 (AIF360), tracking Model Accuracy, Demographic Parity Difference, Disparate Impact Ratio (EEOC 80% Rule), Equal Opportunity Difference (Sensitivity Parity), and Gender Feature Leakage.
- **Pareto Frontier Analysis**: Traces the accuracy vs. equity trade-off across penalty values to establish an optimal operating point for clinical triage.
- **Config Persistence**: Saves calibrated training hyperparameters directly into the audit pipeline.

### 3. Phase 3: Clinical Output Fairness (Post-processing Audit)
- **Counterfactual Fairness Testing**: Automatically swaps demographic terms and gender pronouns (e.g., *she/he*, *adolescent/adult*) while preserving clinical context to assess decision invariance.
- **Decision Consistency Rate**: Identifies decision discrepancies in clinical urgency triage ratings (1–5 scale).
- **Post-processing Governance Checklist**: Tracks deployment safety protocols, monitoring mechanisms, and redress pipelines.
- **Export Certified PDF Report**: Automatically generates an audit certificate and compliance report (`PDF`) documenting findings, gaps, and readiness scores across all phases.

---

##  Requirements

- Python 3.9+
- Python packages:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `scipy`
  - `fpdf2`

---

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd hash-gender-toolkit
   ```

2. **(Optional) Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS / WSL:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

##  Running the Application

Launch the Streamlit app with:

```bash
streamlit run appv2.py
```

The application will start locally and can be accessed in your browser at:
```
http://localhost:8501
```

---

##  Usage Guide

1. **Upload Dataset**:
   - Ingest a CSV containing MSRH patient questions and responses via the sidebar or global file uploader.
   - Select the respective **Question** and **Answer** columns.
2. **Execute Phase 1 (Dataset Evaluation)**:
   - Complete the manual self-assessment checklist.
   - Click **Run Dataset Bias Audit** to compute Chi-Square statistics, detect completeness disparities, and view justifications.
3. **Execute Phase 2 (Active Training Audit)**:
   - Adjust the **Bias Correction Strength** slider ($\lambda$).
   - Inspect the real-time accuracy and leakage metrics, then click **Save Current Bias Correction Setup**.
4. **Execute Phase 3 (Output Fairness & Export)**:
   - Complete the post-processing checklist.
   - Click **Run Final Fairness Consistency Audit** to perform pronoun-swapped counterfactual testing.
   - Click **Download Certified Audit PDF** to export the official audit certificate.

---

##  Project Structure

```text
hash-gender-toolkit/
├── appv2.py              # Main Streamlit application and audit logic
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```
