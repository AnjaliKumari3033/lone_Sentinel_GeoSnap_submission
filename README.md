# 🌍 Geo-SNAP
### Geospatial Scene Classification using RGB and Multispectral Satellite Images

## Team - Lone Sentinel
**Member:**
- Anjali Kumari

Geo-SNAP is a deep learning project for land cover classification using the EuroSAT dataset. The project compares RGB and multispectral satellite imagery and studies how spectral information improves classification performance. It also includes explainability techniques and environmental analysis using remote sensing indices.

## 📌 Project Objectives

- Classify satellite images into 10 land cover classes.
- Compare RGB and multispectral image classification.
- Analyze spectral characteristics of different land cover classes.
- Visualize model decisions using Grad-CAM.
- Perform environmental analysis using NDVI, NDWI and NDBI.

## 📂 Dataset

**Dataset:** EuroSAT

The dataset contains Sentinel-2 satellite images belonging to 10 land cover classes.

### Classes

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

### Dataset Structure
```text
EuroSAT_Dataset/
│
├── EuroSAT/
│   ├── train/
│   └── val/
├── EuroSAT_test_flat/
│
├── EuroSATallBands/
│   ├── train/
│   └── val/
├── EuroSATallBands_test_flat/
│
├── train.csv
└── validation.csv
```

## 🧠 Models Used

### RGB Model

- EfficientNet-B0
- Input: RGB images (3 channels)
- Output: 10 land cover classes

### Multispectral Model

Custom CNN architecture with:

- 13 Sentinel-2 spectral bands
- Residual Blocks
- Spectral Squeeze-and-Excitation (SE) Module
- Global Average Pooling
- Fully Connected Classification Layer

## 📁 Project Structure
```text
GEO-SNAP/
│
├── checkpoints/
│   ├── rgb_best.pth
│   └── ms_best.pth
│
├── EuroSAT_Dataset/
│
├── models/
│   ├── rgb_model.py
│   └── ms_model.py
│
├── Notebooks/
│   ├── Task1_eda.ipynb
│   ├── Task1_training_analysis.ipynb
│   ├── Task1A_rgb_efficientnetb0.ipynb
│   ├── Task1B_Multispectral.ipynb
│   ├── Task2_explainability.ipynb
│   └── Task3_environmental_insights.ipynb
│
├── outputs/
│   ├── figures/
│   └── predictions/
│
├── training_logs/
│   ├── training_logs_efficientnet.csv
│   └── training_logs_multispectral.csv
│
├── .gitignore
│
├── ms_band_stats.json
│
├── README.md
│
├── Report.pdf
│
└── requirements.txt
```
# 📊 Exploratory Data Analysis (eda.ipynb)

The notebook includes:

- Dataset overview
- Train and validation sample counts
- Class distribution plots
- Sample RGB images
- RGB mean pixel statistics
- Visualization of all 13 multispectral bands
- Spectral signatures of all classes
- NDVI, NDWI and NDBI previews
- Summary observations

Generated figures:

- class_distribution.png
- sample_rgb_images.png
- rgb_means_per_class.png
- multispectral_bands.png
- spectral_signatures.png

# 📈 Training Analysis (training_analysis.ipynb)

This notebook analyzes the training performance of both models.

It includes:

- Training and validation loss curves for the RGB model
- Training and validation accuracy curves for the RGB model
- Training and validation loss curves for the multispectral model
- Training and validation accuracy curves for the multispectral model
- Best validation accuracy highlighted on the accuracy plots
- Model comparison table
- Combined RGB vs Multispectral validation accuracy plot
- Classification reports for both models

Generated figures:

- rgb_training_curves.png
- ms_training_curves.png
- model_comparison.png

# 🔍 Explainability (explainability.ipynb)

## Grad-CAM

Grad-CAM was applied to the RGB EfficientNet model.

Visualizations include:

- Correctly classified samples
- Incorrectly classified samples
- Original image
- Heatmap
- Overlay

## Band Importance

For the multispectral model:

- Gradient-based importance computed over validation samples
- Importance of all 13 Sentinel-2 bands visualized

## Confusion Matrix

Generated for both models.

### RGB Model

Includes:

- Confusion matrix
- Top confused class pairs
- Error analysis

### Multispectral Model

Includes:

- Confusion matrix
- Top confused class pairs
- Spectral interpretation

Generated figures:

- gradcam_correct.png
- gradcam_incorrect.png
- band_importance.png
- rgb_confusion_matrix.png
- ms_confusion_matrix.png

# 🌱 Environmental Insights (environmental_insights.ipynb)

Environmental analysis performed using multispectral imagery.

Computed indices:

- NDVI
- NDWI
- NDBI

Visualizations:

- NDVI distribution
- NDWI distribution
- NDBI distribution
- t-SNE visualization using spectral features
- Environmental dashboard
- Environmental Suitability Score
- Summary of environmental observations

Generated figures:

- ndvi_distribution.png
- ndwi_distribution.png
- ndbi_distribution.png
- tsne_spectral.png
- index_scatter.png
- environmental_score.png (Bonus)

# 📈 Spectral Indices

### NDVI
Measures vegetation health.

Higher values indicate dense vegetation.

### NDWI
Measures water content.

Higher values indicate water-rich regions.

### NDBI
Measures built-up areas.

Higher values indicate urban or industrial regions.

# 💻 Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- tifffile
- rasterio
- scikit-learn
- tqdm

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/AnjaliKumari3033/lone_Sentinel_GeoSnap_submission
cd GEO-SNAP
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment (Windows)

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

# ▶️ Running the Project

Run notebooks in the following order:

- Task1_eda.ipynb
- Task1A_rgb_efficientnetb0.ipynb
- Task1B_Multispectral.ipynb
- Task1_training_analysis.ipynb
- Task2_explainability.ipynb
- Task3_environmental_insights.ipynb

Ensure that:
- Dataset folders are correctly placed.
- Model checkpoints are available.
- `ms_band_stats.json` is present.
- Output folders exist.

# 📁 Outputs
The project generates:

- Training and validation plots
- Model comparison plots
- Classification figures
- Grad-CAM visualizations
- Confusion matrices
- Spectral band importance plots
- Environmental analysis plots
- Prediction CSV files
- Evaluation reports

All outputs are saved inside:
outputs/

## 📄 Report

A detailed project report describing the methodology, experiments, explainability analysis, environmental insights, and conclusions is included as:
- Report.pdf

# 📌 Key Observations

- Multispectral imagery provides richer spectral information than RGB imagery.
- Vegetation classes show higher NDVI values.
- Water bodies are clearly identified using NDWI.
- Built-up regions exhibit higher NDBI values.
- Grad-CAM highlights the regions influencing RGB model predictions.
- Band importance analysis identifies the most informative Sentinel-2 spectral bands.
- Environmental indices provide additional insights into land cover characteristics beyond classification.
