# Wildfir Damage and Severity Assessment (Jasper, Alberta, 2024)

Remote-sensing based assessment of the 2024 Jasper, Alberta wildfire: burned
area detection from Sentinel-2 imagery, burn severity mapping, and
machine-learning models that classify/predict burn severity from spectral
and derived indices.

## Project structure

```
.
├── burned_area_download.ipynb        # Pulls pre/post-fire Sentinel-2 L2A scenes
│                                      # (Earth Search STAC API) for the Jasper AOI
├── impact_assessment.ipynb           # Main analysis: NBR/NDVI/dNBR, severity
│                                      # mapping, and ML models (LR, RF, MLP, CNN)
├── alberta_burned_area_project/
│   └── data/
│       └── processed/
│           ├── fire_metadata_jasper.csv   # Fire event metadata
│           ├── *.png                      # Generated figures (severity maps,
│           │                              #   feature importance, model comparison,
│           │                              #   confusion matrices, etc.)
│           └── *.joblib                   # Trained model pipelines (LR / RF / MLP)
├── requirements.txt
└── .gitignore
```

Raw satellite GeoTIFFs and large intermediate arrays/datasets
(`data/raw/`, and the `.npy`/`.csv`/`.parquet` files under `data/processed/`)
are **not** committed to this repo — they're multiple gigabytes and fully
reproducible from the notebooks (see below). Only the small metadata file,
generated figures, and trained model files are tracked.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the analysis

1. Run `burned_area_download.ipynb` to fetch pre-fire and post-fire
   Sentinel-2 bands (NIR, Red, SWIR2.2) for the Jasper AOI into
   `alberta_burned_area_project/data/raw/`.
2. Run `impact_assessment.ipynb` to:
   - Compute NDVI/NBR/dNBR and classify burn severity
   - Build the pixel-level modelling dataset
   - Train and evaluate Logistic Regression, Random Forest, MLP, and CNN
     models on spectral and fused feature sets
   - Generate the figures and trained model artifacts saved under
     `alberta_burned_area_project/data/processed/`

## Results

Key outputs (see `alberta_burned_area_project/data/processed/`):

- `03_dnbr_severity_map.png` — spatial burn severity classification
- `04_severity_distribution.png` — severity class breakdown
- `07_correlation_heatmap.png` — feature correlation across spectral indices
- `11_model_comparison.png` — accuracy/F1 comparison across models
- `12_confusion_matrices.png` — per-model confusion matrices

Trained model pipelines (`phase2_*.joblib`, `pipeline_*.joblib`) can be
loaded directly with `joblib.load(...)` for inference without retraining.
