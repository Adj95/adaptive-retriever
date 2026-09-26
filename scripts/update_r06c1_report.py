"""Update reports/r06c1_transformations.json with Part 1 corrections."""

import json
from pathlib import Path

def main():
    report_path = Path("reports/r06c1_transformations.json")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Update multiplicity statement
    data["c2_measurement"]["multiplicity_statement"] = (
        "3 treatment arms x 3 primary-ish metrics (nDCG@10, nDCG@5, MRR) = 9 tests, alpha = 0.00556. "
        "T3 (self_query) relabelled NOT APPLICABLE and removed from multiplicity count."
    )
    
    # 2. Relabel T3 arm summary and paired test
    if "T3 (self_query)" in data["c2_measurement"]["arms_summary"]:
        data["c2_measurement"]["arms_summary"]["T3 (self_query) [NOT APPLICABLE]"] = data["c2_measurement"]["arms_summary"].pop("T3 (self_query)")
        data["c2_measurement"]["arms_summary"]["T3 (self_query) [NOT APPLICABLE]"]["status_note"] = (
            "NOT APPLICABLE - NFCorpus has no structured metadata schema, so filter extraction had nothing to produce and the arm degenerated to control."
        )

    if "T3 (self_query)" in data["c2_measurement"]["paired_bootstrap_tests"]:
        data["c2_measurement"]["paired_bootstrap_tests"]["T3 (self_query) [NOT APPLICABLE]"] = data["c2_measurement"]["paired_bootstrap_tests"].pop("T3 (self_query)")
        for mk in data["c2_measurement"]["paired_bootstrap_tests"]["T3 (self_query) [NOT APPLICABLE]"]:
            data["c2_measurement"]["paired_bootstrap_tests"]["T3 (self_query) [NOT APPLICABLE]"][mk]["verdict"] = "N/A"
            data["c2_measurement"]["paired_bootstrap_tests"]["T3 (self_query) [NOT APPLICABLE]"][mk]["bonferroni_alpha"] = None

    # 3. Restate bonferroni_alpha to 0.00556 for remaining 3 arms
    alpha_new = 0.00556
    primary_metrics = ["ndcg_10", "ndcg_5", "mrr"]
    for arm in ["T1 (hyde)", "T2 (decompose)", "T4 (hyde_decompose)"]:
        if arm in data["c2_measurement"]["paired_bootstrap_tests"]:
            for mk, test_res in data["c2_measurement"]["paired_bootstrap_tests"][arm].items():
                if mk in primary_metrics:
                    test_res["bonferroni_alpha"] = alpha_new
                    is_sig = bool(test_res["p_value"] < alpha_new and test_res["mean_diff"] > 0)
                    test_res["verdict"] = "PASS" if is_sig else "FAIL"

    # 4. Scoped verdict
    data["c4_verdict"] = {
        "verdict_choice": "(b)",
        "one_sentence_summary": (
            "At 0.5B generator scale, no query transformation cleared Bonferroni on either corpus. "
            "HyDE measured -0.0029 on corpus A and -0.0026 on NFCorpus - effectively identical across "
            "a synthetic lexical corpus with heuristic relevance and a medical benchmark with human judgments. "
            "The replication suggests the null is a property of the method at this model scale rather than of the corpus. "
            "It does NOT establish that HyDE fails with a larger generator, which was not tested."
        )
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Successfully updated reports/r06c1_transformations.json with Part 1 corrections!")

if __name__ == "__main__":
    main()
