# 🌍 Geo-SNAP
### Geospatial Scene Classification using RGB and Multispectral Satellite Images

Geo-SNAP is a deep learning project for land cover classification using the EuroSAT dataset. The project compares RGB and multispectral satellite imagery and studies how spectral information improves classification performance. It also includes explainability techniques and environmental analysis using remote sensing indices.

## 📌 Project Objectives

- Classify satellite images into 10 land cover classes.
- Compare RGB and multispectral image classification.
- Analyze spectral characteristics of different land cover types.
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

EuroSAT_Dataset/
│
├── EuroSAT/
│   ├── train/
│   ├── val/
│   └── test/
│
├── EuroSATallBands/
│   ├── train/
│   ├── val/
│   └── test/
│
├── train.csv
└── validation.csv

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

GEO-SNAP/
│
├── checkpoints/
│   ├── rgb_best.pth
│   └── ms_best.pth
│
├── models/
│   ├── rgb_model.py
│   └── ms_model.py
│
├── Notebooks/
│   ├── eda.ipynb
│   ├── rgb_efficientnetb0.ipynb
│   ├── Multispectral.ipynb
│   ├── explainability.ipynb
│   └── environmental_insights.ipynb
│
├── outputs/
│   ├── figures/
│   ├── predictions/
│   └── reports/
│
├── EuroSAT_Dataset/
│
├── ms_band_stats.json
│
├── requirements.txt
│
└── README.md

# 📊 Exploratory Data Analysis (01_eda.ipynb)

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

# 🔍 Explainability (04_explainability.ipynb)

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

# 🌱 Environmental Insights (05_environmental_insights.ipynb)

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
git clone <repository_link>
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

eda.ipynb
rgb_efficientnetb0.ipynb
Multispectral.ipynb
explainability.ipynb
environmental_insights.ipynb

Ensure that:
- Dataset folders are correctly placed.
- Model checkpoints are available.
- `ms_band_stats.json` is present.
- Output folders exist.

# 📁 Outputs
The project generates:

- Classification figures
- Grad-CAM visualizations
- Confusion matrices
- Band importance plots
- Environmental analysis plots
- Feature CSV files

All outputs are saved inside:
outputs/

# 📌 Key Observations

- Multispectral imagery provides richer spectral information than RGB imagery.
- Vegetation classes show higher NDVI values.
- Water bodies are clearly identified using NDWI.
- Built-up regions exhibit higher NDBI values.
- Grad-CAM highlights the regions influencing RGB model predictions.
- Band importance analysis identifies the most informative Sentinel-2 spectral bands.
- Environmental indices provide additional insights into land cover characteristics beyond classification.
