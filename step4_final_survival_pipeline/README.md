# Survival Analysis Pipeline

This step integrates clinical features and image embeddings to train, evaluate,
and compare multiple survival modeling configurations across different labeling
strategies, horizons, and experiment settings.

## Notes
Due to patient privacy and data use agreements, raw data, intermediate files,
and model outputs are **not included in this repository**. The code is provided
for transparency and to illustrate the methodology and analysis pipeline.

`pipeline_main_v2_7_0.py` is a cleaned Python version of the main notebook (`pipeline_main_v2_7_0.ipynb`), with
exploratory plots and notebook-only outputs removed.

## Contents
- Unified feature preprocessing and labeling for survival analysis
- Model training across clinical-only, image-only, and fusion configurations
- Controlled vs. uncontrolled experiment design
- Evaluation across multiple time horizons
- Result aggregation and comparative analysis

## Contribution
The survival modeling pipeline—including feature preprocessing, experiment
design, model training, evaluation, and result aggregation—was primarily
designed and implemented by Haojie Yin. The analysis and interpretation of
experimental results were conducted by Haojie Yin. Minor components
were contributed by collaborators and integrated into the unified framework.
