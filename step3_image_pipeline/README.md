# Image Pipeline

This step focuses on extracting patient-level representations from CT imaging data.
It consists of two stages: image preprocessing and filtering, followed by image-only
modeling to learn fixed-length patient embeddings for downstream analysis.

## Notes
Due to patient privacy and data use agreements, raw data, intermediate files,
and model outputs are **not included in this repository**. The code is provided
for transparency and to illustrate the methodology and analysis pipeline.


## Structure

### 1. Image Preprocessing and Filtering (`filter/`)
This stage performs quality control and preprocessing on raw CT scans, including
slice-level filtering and metadata handling. Details are documented in
`image_preprocessing.pdf`.

### 2. Image-only Modeling (`model/`)
This stage trains image-only models on the preprocessed scans and aggregates
slice-level representations into patient-level embeddings. Model design and
training details are documented in `image_only_model.pdf`.

## Outputs
- Patient-level image embeddings used as inputs to the final survival analysis
