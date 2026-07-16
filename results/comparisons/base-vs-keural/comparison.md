# Eval comparison

- **Baseline:** `base` (Qwen/Qwen3.6-35B-A3B) — /home/work/keural MoE Pro/results/base/20260710T045822Z
- **Candidate:** `keural` (outputs/merged/Keural-35B-A3B) — /home/work/keural MoE Pro/results/keural/20260716T024252Z
- **Verdict:** REGRESSIONS FOUND — do not ship without investigating

| task | suite | metric | baseline | candidate | Δ | |
|---|---|---|---:|---:|---:|---|
| gsm8k | general | exact_match,strict-match | 0.3215 | 0.8688 | +0.5474 | improved |
| humaneval | general | pass@1,create_test | 0.6220 | 0.1159 | -0.5061 | REGRESSION |
| mmlu | general | acc,none | 0.8383 | 0.8237 | -0.0147 | REGRESSION |
| mmlu_abstract_algebra | general | acc,none | 0.6900 | 0.6300 | -0.0600 | REGRESSION |
| mmlu_anatomy | general | acc,none | 0.8741 | 0.8296 | -0.0444 | REGRESSION |
| mmlu_astronomy | general | acc,none | 0.9145 | 0.9145 | +0.0000 |  |
| mmlu_business_ethics | general | acc,none | 0.8800 | 0.8400 | -0.0400 | REGRESSION |
| mmlu_clinical_knowledge | general | acc,none | 0.8981 | 0.8792 | -0.0189 | REGRESSION |
| mmlu_college_biology | general | acc,none | 0.9444 | 0.9444 | +0.0000 |  |
| mmlu_college_chemistry | general | acc,none | 0.6600 | 0.6200 | -0.0400 | REGRESSION |
| mmlu_college_computer_science | general | acc,none | 0.7500 | 0.7300 | -0.0200 | REGRESSION |
| mmlu_college_mathematics | general | acc,none | 0.6700 | 0.6700 | +0.0000 |  |
| mmlu_college_medicine | general | acc,none | 0.8671 | 0.8382 | -0.0289 | REGRESSION |
| mmlu_college_physics | general | acc,none | 0.6961 | 0.7255 | +0.0294 | improved |
| mmlu_computer_security | general | acc,none | 0.8700 | 0.8700 | +0.0000 |  |
| mmlu_conceptual_physics | general | acc,none | 0.9489 | 0.9319 | -0.0170 | REGRESSION |
| mmlu_econometrics | general | acc,none | 0.8158 | 0.7807 | -0.0351 | REGRESSION |
| mmlu_electrical_engineering | general | acc,none | 0.8552 | 0.8414 | -0.0138 | REGRESSION |
| mmlu_elementary_mathematics | general | acc,none | 0.8333 | 0.7989 | -0.0344 | REGRESSION |
| mmlu_formal_logic | general | acc,none | 0.6984 | 0.6984 | +0.0000 |  |
| mmlu_global_facts | general | acc,none | 0.5400 | 0.5400 | +0.0000 |  |
| mmlu_high_school_biology | general | acc,none | 0.9645 | 0.9484 | -0.0161 | REGRESSION |
| mmlu_high_school_chemistry | general | acc,none | 0.8128 | 0.8177 | +0.0049 |  |
| mmlu_high_school_computer_science | general | acc,none | 0.9100 | 0.9000 | -0.0100 | REGRESSION |
| mmlu_high_school_european_history | general | acc,none | 0.8606 | 0.8667 | +0.0061 |  |
| mmlu_high_school_geography | general | acc,none | 0.9343 | 0.9495 | +0.0152 | improved |
| mmlu_high_school_government_and_politics | general | acc,none | 0.9793 | 0.9741 | -0.0052 |  |
| mmlu_high_school_macroeconomics | general | acc,none | 0.8897 | 0.8949 | +0.0051 |  |
| mmlu_high_school_mathematics | general | acc,none | 0.6222 | 0.5852 | -0.0370 | REGRESSION |
| mmlu_high_school_microeconomics | general | acc,none | 0.9454 | 0.9496 | +0.0042 |  |
| mmlu_high_school_physics | general | acc,none | 0.8278 | 0.8278 | +0.0000 |  |
| mmlu_high_school_psychology | general | acc,none | 0.9596 | 0.9596 | +0.0000 |  |
| mmlu_high_school_statistics | general | acc,none | 0.8148 | 0.8009 | -0.0139 | REGRESSION |
| mmlu_high_school_us_history | general | acc,none | 0.9363 | 0.9314 | -0.0049 |  |
| mmlu_high_school_world_history | general | acc,none | 0.9198 | 0.9156 | -0.0042 |  |
| mmlu_human_aging | general | acc,none | 0.8386 | 0.8072 | -0.0314 | REGRESSION |
| mmlu_human_sexuality | general | acc,none | 0.9084 | 0.8931 | -0.0153 | REGRESSION |
| mmlu_humanities | general | acc,none | 0.7815 | 0.7600 | -0.0215 | REGRESSION |
| mmlu_international_law | general | acc,none | 0.9256 | 0.9091 | -0.0165 | REGRESSION |
| mmlu_jurisprudence | general | acc,none | 0.8889 | 0.8704 | -0.0185 | REGRESSION |
| mmlu_logical_fallacies | general | acc,none | 0.9202 | 0.9202 | +0.0000 |  |
| mmlu_machine_learning | general | acc,none | 0.7768 | 0.7768 | +0.0000 |  |
| mmlu_management | general | acc,none | 0.9029 | 0.9223 | +0.0194 | improved |
| mmlu_marketing | general | acc,none | 0.9444 | 0.9444 | +0.0000 |  |
| mmlu_medical_genetics | general | acc,none | 0.9300 | 0.9600 | +0.0300 | improved |
| mmlu_miscellaneous | general | acc,none | 0.9476 | 0.9336 | -0.0140 | REGRESSION |
| mmlu_moral_disputes | general | acc,none | 0.8584 | 0.8237 | -0.0347 | REGRESSION |
| mmlu_moral_scenarios | general | acc,none | 0.6480 | 0.6223 | -0.0257 | REGRESSION |
| mmlu_nutrition | general | acc,none | 0.8922 | 0.8758 | -0.0163 | REGRESSION |
| mmlu_other | general | acc,none | 0.8693 | 0.8548 | -0.0145 | REGRESSION |
| mmlu_philosophy | general | acc,none | 0.8842 | 0.8714 | -0.0129 | REGRESSION |
| mmlu_prehistory | general | acc,none | 0.9136 | 0.8889 | -0.0247 | REGRESSION |
| mmlu_professional_accounting | general | acc,none | 0.7411 | 0.7199 | -0.0213 | REGRESSION |
| mmlu_professional_law | general | acc,none | 0.7021 | 0.6708 | -0.0313 | REGRESSION |
| mmlu_professional_medicine | general | acc,none | 0.9449 | 0.9228 | -0.0221 | REGRESSION |
| mmlu_professional_psychology | general | acc,none | 0.8873 | 0.8758 | -0.0114 | REGRESSION |
| mmlu_public_relations | general | acc,none | 0.7636 | 0.7818 | +0.0182 | improved |
| mmlu_security_studies | general | acc,none | 0.8367 | 0.8245 | -0.0122 | REGRESSION |
| mmlu_social_sciences | general | acc,none | 0.9077 | 0.9051 | -0.0026 |  |
| mmlu_sociology | general | acc,none | 0.9254 | 0.9353 | +0.0100 |  |
| mmlu_stem | general | acc,none | 0.8249 | 0.8084 | -0.0165 | REGRESSION |
| mmlu_us_foreign_policy | general | acc,none | 0.9400 | 0.9300 | -0.0100 |  |
| mmlu_virology | general | acc,none | 0.5783 | 0.5723 | -0.0060 |  |
| mmlu_world_religions | general | acc,none | 0.9064 | 0.9006 | -0.0058 |  |
| haerae | korean | acc,none | 0.7553 | 0.7452 | -0.0101 |  |
| haerae_general_knowledge | korean | acc,none | 0.5682 | 0.5398 | -0.0284 |  |
| haerae_history | korean | acc,none | 0.8777 | 0.8564 | -0.0213 |  |
| haerae_loan_word | korean | acc,none | 0.6627 | 0.7929 | +0.1302 | improved |
| haerae_rare_word | korean | acc,none | 0.8247 | 0.7679 | -0.0568 |  |
| haerae_standard_nomenclature | korean | acc,none | 0.7386 | 0.7320 | -0.0065 |  |
| kmmlu | korean | acc,none | 0.6446 | 0.6238 | -0.0208 |  |
| kmmlu_accounting | korean | acc,none | 0.7100 | 0.6800 | -0.0300 |  |
| kmmlu_agricultural_sciences | korean | acc,none | 0.5280 | 0.4920 | -0.0360 |  |
| kmmlu_applied_science | korean | acc,none | 0.6227 | 0.5990 | -0.0237 |  |
| kmmlu_aviation_engineering_and_maintenance | korean | acc,none | 0.6730 | 0.6510 | -0.0220 |  |
| kmmlu_biology | korean | acc,none | 0.6130 | 0.5840 | -0.0290 |  |
| kmmlu_chemical_engineering | korean | acc,none | 0.6720 | 0.6710 | -0.0010 |  |
| kmmlu_chemistry | korean | acc,none | 0.7417 | 0.7217 | -0.0200 |  |
| kmmlu_civil_engineering | korean | acc,none | 0.5770 | 0.5440 | -0.0330 |  |
| kmmlu_computer_science | korean | acc,none | 0.8640 | 0.8520 | -0.0120 |  |
| kmmlu_construction | korean | acc,none | 0.5400 | 0.5100 | -0.0300 |  |
| kmmlu_criminal_law | korean | acc,none | 0.5200 | 0.4750 | -0.0450 |  |
| kmmlu_ecology | korean | acc,none | 0.6220 | 0.6060 | -0.0160 |  |
| kmmlu_economics | korean | acc,none | 0.7308 | 0.6692 | -0.0615 |  |
| kmmlu_education | korean | acc,none | 0.8400 | 0.8300 | -0.0100 |  |
| kmmlu_electrical_engineering | korean | acc,none | 0.5000 | 0.4800 | -0.0200 |  |
| kmmlu_electronics_engineering | korean | acc,none | 0.7630 | 0.7380 | -0.0250 |  |
| kmmlu_energy_management | korean | acc,none | 0.5080 | 0.4870 | -0.0210 |  |
| kmmlu_environmental_science | korean | acc,none | 0.5170 | 0.4800 | -0.0370 |  |
| kmmlu_fashion | korean | acc,none | 0.5920 | 0.5820 | -0.0100 |  |
| kmmlu_food_processing | korean | acc,none | 0.6440 | 0.6150 | -0.0290 |  |
| kmmlu_gas_technology_and_engineering | korean | acc,none | 0.5490 | 0.5200 | -0.0290 |  |
| kmmlu_geomatics | korean | acc,none | 0.5650 | 0.5210 | -0.0440 |  |
| kmmlu_health | korean | acc,none | 0.8000 | 0.7600 | -0.0400 |  |
| kmmlu_humss | korean | acc,none | 0.6727 | 0.6612 | -0.0115 |  |
| kmmlu_industrial_engineer | korean | acc,none | 0.6140 | 0.6030 | -0.0110 |  |
| kmmlu_information_technology | korean | acc,none | 0.8580 | 0.8310 | -0.0270 |  |
| kmmlu_interior_architecture_and_design | korean | acc,none | 0.7500 | 0.7320 | -0.0180 |  |
| kmmlu_korean_history | korean | acc,none | 0.4100 | 0.4300 | +0.0200 | improved |
| kmmlu_law | korean | acc,none | 0.6180 | 0.6010 | -0.0170 |  |
| kmmlu_machine_design_and_manufacturing | korean | acc,none | 0.6870 | 0.6390 | -0.0480 |  |
| kmmlu_management | korean | acc,none | 0.7220 | 0.7280 | +0.0060 |  |
| kmmlu_maritime_engineering | korean | acc,none | 0.6433 | 0.6417 | -0.0017 |  |
| kmmlu_marketing | korean | acc,none | 0.8700 | 0.8620 | -0.0080 |  |
| kmmlu_materials_engineering | korean | acc,none | 0.7200 | 0.7140 | -0.0060 |  |
| kmmlu_math | korean | acc,none | 0.3700 | 0.4033 | +0.0333 | improved |
| kmmlu_mechanical_engineering | korean | acc,none | 0.6340 | 0.6040 | -0.0300 |  |
| kmmlu_nondestructive_testing | korean | acc,none | 0.6440 | 0.6340 | -0.0100 |  |
| kmmlu_other | korean | acc,none | 0.6298 | 0.6038 | -0.0260 |  |
| kmmlu_patent | korean | acc,none | 0.5100 | 0.4500 | -0.0600 |  |
| kmmlu_political_science_and_sociology | korean | acc,none | 0.7133 | 0.7367 | +0.0233 | improved |
| kmmlu_psychology | korean | acc,none | 0.6210 | 0.6030 | -0.0180 |  |
| kmmlu_public_safety | korean | acc,none | 0.5550 | 0.5340 | -0.0210 |  |
| kmmlu_railway_and_automotive_engineering | korean | acc,none | 0.5440 | 0.5330 | -0.0110 |  |
| kmmlu_real_estate | korean | acc,none | 0.6100 | 0.5400 | -0.0700 |  |
| kmmlu_refrigerating_machinery | korean | acc,none | 0.5580 | 0.5160 | -0.0420 |  |
| kmmlu_social_welfare | korean | acc,none | 0.7850 | 0.7610 | -0.0240 |  |
| kmmlu_stem | korean | acc,none | 0.6683 | 0.6505 | -0.0178 |  |
| kmmlu_taxation | korean | acc,none | 0.4800 | 0.5100 | +0.0300 | improved |
| kmmlu_telecommunications_and_wireless_technology | korean | acc,none | 0.7730 | 0.7570 | -0.0160 |  |
| kobest | korean | acc,none | 0.6288 | 0.6729 | +0.0441 | improved |
| kobest_boolq | korean | acc,none | 0.6061 | 0.7913 | +0.1852 | improved |
| kobest_copa | korean | acc,none | 0.8090 | 0.7870 | -0.0220 |  |
| kobest_hellaswag | korean | acc,none | 0.4880 | 0.4680 | -0.0200 |  |
| kobest_sentineg | korean | acc,none | 0.8791 | 0.8111 | -0.0680 |  |
| kobest_wic | korean | acc,none | 0.4881 | 0.4881 | +0.0000 |  |

## Domain eval

- baseline: skipped (no files)
- candidate: skipped (no files)
