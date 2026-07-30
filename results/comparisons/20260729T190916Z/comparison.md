# Eval comparison

- **Baseline:** `base` (Qwen/Qwen3.6-35B-A3B) — /home/work/keural MoE Pro/results/base/20260710T045822Z
- **Candidate:** `v12` (outputs/v12/merged/Keural-Nova-v1.2-experimental) — /home/work/keural MoE Pro/results/v12/20260729T183655Z
- **Verdict:** REGRESSIONS FOUND — do not ship without investigating

| task | suite | metric | baseline | candidate | Δ | |
|---|---|---|---:|---:|---:|---|
| gsm8k | general | exact_match,strict-match | 0.3215 | 0.7225 | +0.4011 | improved |
| humaneval | general | pass@1,create_test | 0.6220 | 0.6829 | +0.0610 | improved |
| mmlu | general | acc,none | 0.8383 | 0.8288 | -0.0095 |  |
| mmlu_abstract_algebra | general | acc,none | 0.6900 | 0.6900 | +0.0000 |  |
| mmlu_anatomy | general | acc,none | 0.8741 | 0.8593 | -0.0148 | REGRESSION |
| mmlu_astronomy | general | acc,none | 0.9145 | 0.9211 | +0.0066 |  |
| mmlu_business_ethics | general | acc,none | 0.8800 | 0.8300 | -0.0500 | REGRESSION |
| mmlu_clinical_knowledge | general | acc,none | 0.8981 | 0.9057 | +0.0075 |  |
| mmlu_college_biology | general | acc,none | 0.9444 | 0.9306 | -0.0139 | REGRESSION |
| mmlu_college_chemistry | general | acc,none | 0.6600 | 0.6400 | -0.0200 | REGRESSION |
| mmlu_college_computer_science | general | acc,none | 0.7500 | 0.7600 | +0.0100 | improved |
| mmlu_college_mathematics | general | acc,none | 0.6700 | 0.6400 | -0.0300 | REGRESSION |
| mmlu_college_medicine | general | acc,none | 0.8671 | 0.8208 | -0.0462 | REGRESSION |
| mmlu_college_physics | general | acc,none | 0.6961 | 0.6569 | -0.0392 | REGRESSION |
| mmlu_computer_security | general | acc,none | 0.8700 | 0.8500 | -0.0200 | REGRESSION |
| mmlu_conceptual_physics | general | acc,none | 0.9489 | 0.9489 | +0.0000 |  |
| mmlu_econometrics | general | acc,none | 0.8158 | 0.7719 | -0.0439 | REGRESSION |
| mmlu_electrical_engineering | general | acc,none | 0.8552 | 0.8345 | -0.0207 | REGRESSION |
| mmlu_elementary_mathematics | general | acc,none | 0.8333 | 0.8254 | -0.0079 |  |
| mmlu_formal_logic | general | acc,none | 0.6984 | 0.7143 | +0.0159 | improved |
| mmlu_global_facts | general | acc,none | 0.5400 | 0.6100 | +0.0700 | improved |
| mmlu_high_school_biology | general | acc,none | 0.9645 | 0.9484 | -0.0161 | REGRESSION |
| mmlu_high_school_chemistry | general | acc,none | 0.8128 | 0.8030 | -0.0099 |  |
| mmlu_high_school_computer_science | general | acc,none | 0.9100 | 0.9100 | +0.0000 |  |
| mmlu_high_school_european_history | general | acc,none | 0.8606 | 0.8727 | +0.0121 | improved |
| mmlu_high_school_geography | general | acc,none | 0.9343 | 0.9394 | +0.0051 |  |
| mmlu_high_school_government_and_politics | general | acc,none | 0.9793 | 0.9845 | +0.0052 |  |
| mmlu_high_school_macroeconomics | general | acc,none | 0.8897 | 0.8897 | +0.0000 |  |
| mmlu_high_school_mathematics | general | acc,none | 0.6222 | 0.5704 | -0.0519 | REGRESSION |
| mmlu_high_school_microeconomics | general | acc,none | 0.9454 | 0.9538 | +0.0084 |  |
| mmlu_high_school_physics | general | acc,none | 0.8278 | 0.7682 | -0.0596 | REGRESSION |
| mmlu_high_school_psychology | general | acc,none | 0.9596 | 0.9578 | -0.0018 |  |
| mmlu_high_school_statistics | general | acc,none | 0.8148 | 0.7778 | -0.0370 | REGRESSION |
| mmlu_high_school_us_history | general | acc,none | 0.9363 | 0.9265 | -0.0098 |  |
| mmlu_high_school_world_history | general | acc,none | 0.9198 | 0.9241 | +0.0042 |  |
| mmlu_human_aging | general | acc,none | 0.8386 | 0.8341 | -0.0045 |  |
| mmlu_human_sexuality | general | acc,none | 0.9084 | 0.8855 | -0.0229 | REGRESSION |
| mmlu_humanities | general | acc,none | 0.7815 | 0.7715 | -0.0100 |  |
| mmlu_international_law | general | acc,none | 0.9256 | 0.9008 | -0.0248 | REGRESSION |
| mmlu_jurisprudence | general | acc,none | 0.8889 | 0.9167 | +0.0278 | improved |
| mmlu_logical_fallacies | general | acc,none | 0.9202 | 0.9018 | -0.0184 | REGRESSION |
| mmlu_machine_learning | general | acc,none | 0.7768 | 0.7679 | -0.0089 |  |
| mmlu_management | general | acc,none | 0.9029 | 0.9223 | +0.0194 | improved |
| mmlu_marketing | general | acc,none | 0.9444 | 0.9402 | -0.0043 |  |
| mmlu_medical_genetics | general | acc,none | 0.9300 | 0.9500 | +0.0200 | improved |
| mmlu_miscellaneous | general | acc,none | 0.9476 | 0.9489 | +0.0013 |  |
| mmlu_moral_disputes | general | acc,none | 0.8584 | 0.8526 | -0.0058 |  |
| mmlu_moral_scenarios | general | acc,none | 0.6480 | 0.6503 | +0.0022 |  |
| mmlu_nutrition | general | acc,none | 0.8922 | 0.8922 | +0.0000 |  |
| mmlu_other | general | acc,none | 0.8693 | 0.8674 | -0.0019 |  |
| mmlu_philosophy | general | acc,none | 0.8842 | 0.8617 | -0.0225 | REGRESSION |
| mmlu_prehistory | general | acc,none | 0.9136 | 0.8951 | -0.0185 | REGRESSION |
| mmlu_professional_accounting | general | acc,none | 0.7411 | 0.7234 | -0.0177 | REGRESSION |
| mmlu_professional_law | general | acc,none | 0.7021 | 0.6806 | -0.0215 | REGRESSION |
| mmlu_professional_medicine | general | acc,none | 0.9449 | 0.9449 | +0.0000 |  |
| mmlu_professional_psychology | general | acc,none | 0.8873 | 0.8758 | -0.0114 | REGRESSION |
| mmlu_public_relations | general | acc,none | 0.7636 | 0.7455 | -0.0182 | REGRESSION |
| mmlu_security_studies | general | acc,none | 0.8367 | 0.8122 | -0.0245 | REGRESSION |
| mmlu_social_sciences | general | acc,none | 0.9077 | 0.9002 | -0.0075 |  |
| mmlu_sociology | general | acc,none | 0.9254 | 0.9154 | -0.0100 |  |
| mmlu_stem | general | acc,none | 0.8249 | 0.8065 | -0.0184 | REGRESSION |
| mmlu_us_foreign_policy | general | acc,none | 0.9400 | 0.9300 | -0.0100 |  |
| mmlu_virology | general | acc,none | 0.5783 | 0.5783 | +0.0000 |  |
| mmlu_world_religions | general | acc,none | 0.9064 | 0.9006 | -0.0058 |  |
| haerae | korean | acc,none | 0.7553 | 0.7388 | -0.0165 |  |
| haerae_general_knowledge | korean | acc,none | 0.5682 | 0.5455 | -0.0227 |  |
| haerae_history | korean | acc,none | 0.8777 | 0.8723 | -0.0053 |  |
| haerae_loan_word | korean | acc,none | 0.6627 | 0.6036 | -0.0592 |  |
| haerae_rare_word | korean | acc,none | 0.8247 | 0.8173 | -0.0074 |  |
| haerae_standard_nomenclature | korean | acc,none | 0.7386 | 0.7386 | +0.0000 |  |
| kmmlu | korean | acc,none | 0.6446 | 0.6405 | -0.0041 |  |
| kmmlu_accounting | korean | acc,none | 0.7100 | 0.6700 | -0.0400 |  |
| kmmlu_agricultural_sciences | korean | acc,none | 0.5280 | 0.5120 | -0.0160 |  |
| kmmlu_applied_science | korean | acc,none | 0.6227 | 0.6163 | -0.0064 |  |
| kmmlu_aviation_engineering_and_maintenance | korean | acc,none | 0.6730 | 0.6670 | -0.0060 |  |
| kmmlu_biology | korean | acc,none | 0.6130 | 0.6110 | -0.0020 |  |
| kmmlu_chemical_engineering | korean | acc,none | 0.6720 | 0.6820 | +0.0100 | improved |
| kmmlu_chemistry | korean | acc,none | 0.7417 | 0.7400 | -0.0017 |  |
| kmmlu_civil_engineering | korean | acc,none | 0.5770 | 0.5750 | -0.0020 |  |
| kmmlu_computer_science | korean | acc,none | 0.8640 | 0.8610 | -0.0030 |  |
| kmmlu_construction | korean | acc,none | 0.5400 | 0.5110 | -0.0290 |  |
| kmmlu_criminal_law | korean | acc,none | 0.5200 | 0.4600 | -0.0600 |  |
| kmmlu_ecology | korean | acc,none | 0.6220 | 0.6060 | -0.0160 |  |
| kmmlu_economics | korean | acc,none | 0.7308 | 0.7154 | -0.0154 |  |
| kmmlu_education | korean | acc,none | 0.8400 | 0.8300 | -0.0100 |  |
| kmmlu_electrical_engineering | korean | acc,none | 0.5000 | 0.4660 | -0.0340 |  |
| kmmlu_electronics_engineering | korean | acc,none | 0.7630 | 0.7650 | +0.0020 |  |
| kmmlu_energy_management | korean | acc,none | 0.5080 | 0.5080 | +0.0000 |  |
| kmmlu_environmental_science | korean | acc,none | 0.5170 | 0.4970 | -0.0200 |  |
| kmmlu_fashion | korean | acc,none | 0.5920 | 0.5940 | +0.0020 |  |
| kmmlu_food_processing | korean | acc,none | 0.6440 | 0.6500 | +0.0060 |  |
| kmmlu_gas_technology_and_engineering | korean | acc,none | 0.5490 | 0.5260 | -0.0230 |  |
| kmmlu_geomatics | korean | acc,none | 0.5650 | 0.5480 | -0.0170 |  |
| kmmlu_health | korean | acc,none | 0.8000 | 0.8100 | +0.0100 | improved |
| kmmlu_humss | korean | acc,none | 0.6727 | 0.6760 | +0.0033 |  |
| kmmlu_industrial_engineer | korean | acc,none | 0.6140 | 0.5990 | -0.0150 |  |
| kmmlu_information_technology | korean | acc,none | 0.8580 | 0.8530 | -0.0050 |  |
| kmmlu_interior_architecture_and_design | korean | acc,none | 0.7500 | 0.7400 | -0.0100 |  |
| kmmlu_korean_history | korean | acc,none | 0.4100 | 0.4000 | -0.0100 |  |
| kmmlu_law | korean | acc,none | 0.6180 | 0.6160 | -0.0020 |  |
| kmmlu_machine_design_and_manufacturing | korean | acc,none | 0.6870 | 0.6850 | -0.0020 |  |
| kmmlu_management | korean | acc,none | 0.7220 | 0.7340 | +0.0120 | improved |
| kmmlu_maritime_engineering | korean | acc,none | 0.6433 | 0.6500 | +0.0067 |  |
| kmmlu_marketing | korean | acc,none | 0.8700 | 0.8760 | +0.0060 |  |
| kmmlu_materials_engineering | korean | acc,none | 0.7200 | 0.7260 | +0.0060 |  |
| kmmlu_math | korean | acc,none | 0.3700 | 0.4067 | +0.0367 | improved |
| kmmlu_mechanical_engineering | korean | acc,none | 0.6340 | 0.6490 | +0.0150 | improved |
| kmmlu_nondestructive_testing | korean | acc,none | 0.6440 | 0.6370 | -0.0070 |  |
| kmmlu_other | korean | acc,none | 0.6298 | 0.6221 | -0.0076 |  |
| kmmlu_patent | korean | acc,none | 0.5100 | 0.5000 | -0.0100 |  |
| kmmlu_political_science_and_sociology | korean | acc,none | 0.7133 | 0.7233 | +0.0100 | improved |
| kmmlu_psychology | korean | acc,none | 0.6210 | 0.6310 | +0.0100 | improved |
| kmmlu_public_safety | korean | acc,none | 0.5550 | 0.5440 | -0.0110 |  |
| kmmlu_railway_and_automotive_engineering | korean | acc,none | 0.5440 | 0.5460 | +0.0020 |  |
| kmmlu_real_estate | korean | acc,none | 0.6100 | 0.5700 | -0.0400 |  |
| kmmlu_refrigerating_machinery | korean | acc,none | 0.5580 | 0.5540 | -0.0040 |  |
| kmmlu_social_welfare | korean | acc,none | 0.7850 | 0.7920 | +0.0070 |  |
| kmmlu_stem | korean | acc,none | 0.6683 | 0.6662 | -0.0021 |  |
| kmmlu_taxation | korean | acc,none | 0.4800 | 0.5150 | +0.0350 | improved |
| kmmlu_telecommunications_and_wireless_technology | korean | acc,none | 0.7730 | 0.7810 | +0.0080 |  |
| kobest | korean | acc,none | 0.6288 | 0.7003 | +0.0715 | improved |
| kobest_boolq | korean | acc,none | 0.6061 | 0.8269 | +0.2208 | improved |
| kobest_copa | korean | acc,none | 0.8090 | 0.8220 | +0.0130 | improved |
| kobest_hellaswag | korean | acc,none | 0.4880 | 0.4880 | +0.0000 |  |
| kobest_sentineg | korean | acc,none | 0.8791 | 0.8866 | +0.0076 |  |
| kobest_wic | korean | acc,none | 0.4881 | 0.4881 | +0.0000 |  |

## Domain eval

- baseline: skipped (no files)
- candidate: skipped (no files)
