import pandas as pd
import numpy as np

def run_tests():
    print("=== 1. DATASET LOADED ===")
    df = pd.read_csv("test_patient_data.csv")
    print(f"Total rows: {len(df)}")
    assert len(df) == 16, "Expected 16 rows"

    # Import functions from appv2
    import appv2
    print("appv2 module loaded successfully.")

    print("\n=== 2. TESTING EXTRACT FEATURES (PHASE 1) ===")
    proc_df = appv2.extract_features_from_qa(df, "Question", "Answer")
    print("Extracted cohorts:\n", proc_df["Audited_Cohort"].value_counts().to_dict())
    assert "Audited_Cohort" in proc_df.columns
    assert "Detailed_Response_Provided" in proc_df.columns

    print("\nChi2 Missingness Analysis:")
    tab_res = appv2.analyze_differential_missingness(proc_df, "Audited_Cohort", ["Detailed_Response_Provided"])
    print(tab_res)
    assert not tab_res.empty

    print("\n=== 3. TESTING IN-PROCESSING AUDIT (PHASE 2) ===")
    for l_test in [0.0, 0.4, 0.8, 1.0]:
        m = appv2.run_in_processing_audit(df, "Question", "Answer", l_test)
        print(f"lambda={l_test:.1f} | Acc={m['accuracy']:.1f}% | Leakage={m['leakage']:.1f}% | DPD={m['dp_diff']:.3f} | DIR={m['dir_ratio']:.3f} | EOD={m['eq_opp_diff']:.3f}")
        assert 0.0 <= m["accuracy"] <= 100.0
        assert 0.0 <= m["leakage"] <= 100.0
        assert 0.0 <= m["dp_diff"] <= 1.0
        assert 0.0 <= m["dir_ratio"] <= 1.0
        assert 0.0 <= m["eq_opp_diff"] <= 1.0

    print("\n=== 4. TESTING PARETO FRONTIER COMPUTATION ===")
    pareto_df = appv2.compute_pareto_frontier(df, "Question", "Answer", steps=5)
    print(pareto_df.to_string())
    assert len(pareto_df) == 5

    print("\n=== 5. TESTING COUNTERFACTUAL ENGINE (PHASE 3) ===")
    sample_q = df["Question"].iloc[0]
    cf_q = appv2.generate_counterfactual(sample_q)
    print(f"Original:       {sample_q}")
    print(f"Counterfactual: {cf_q}")
    assert cf_q != ""
    assert cf_q != sample_q  # "17yo" should swap to "32yo"

    print("\n=== 6. TESTING PDF COMPLIANCE CERTIFICATE GENERATION ===")
    latest_m = appv2.run_in_processing_audit(df, "Question", "Answer", 0.4)
    latest_m["status"] = "Compliant: Low Gender Bias & Balanced Impact"
    pdf_bytes = appv2.create_pdf_report(
        tabular_df=tab_res,
        in_proc_metrics=latest_m,
        qa_stability_rate=95.0,
        total_evaluated=len(df),
        failed_evals=[],
        manual_improvements=["Establish answer parity guidelines"],
        post_proc_checklist_scores={"deployment_protection": 4, "monitoring_auditing": 5, "complaints_redress": 3}
    )
    print(f"PDF certificate generated successfully: {len(pdf_bytes)} bytes")
    assert len(pdf_bytes) > 1000

    print("\n>>> ALL TESTS COMPLETED AND VERIFIED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    run_tests()
