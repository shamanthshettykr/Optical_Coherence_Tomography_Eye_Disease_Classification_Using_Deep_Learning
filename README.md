# OculoAI – OCT Eye Disease Detection

An AI-powered web application for detecting 7 retinal diseases from OCT scans using MobileNetV2 deep learning.

## Model Performance
- **Validation Accuracy: 84.4%**
- Architecture: MobileNetV2 (transfer learning)
- Dataset: OCTDL (2,064 images, 7 classes)
- Inference: 5-pass Test-Time Augmentation (TTA)

## Detectable Conditions
| Code | Disease | Val Accuracy |
|------|---------|-------------|
| AMD  | Age-Related Macular Degeneration | 90.2% |
| DME  | Diabetic Macular Edema | 68.2% |
| ERM  | Epiretinal Membrane | 52.2% |
| NO   | Normal Retina | 91.8% |
| RAO  | Retinal Artery Occlusion | 100% |
| RVO  | Retinal Vein Occlusion | 40.0% |
| VID  | Vitreomacular Interface Disease | 90.9% |

## Pages
- **Home** – Overview and disease cards
- **Information** – Full OCT guide for all 7 conditions
- **Predict** – Upload OCT scan and get AI prediction
- **Analysis** – Confusion matrix, accuracy/loss curves, per-class F1/precision/recall, sample images
- **History** – All past predictions

## Setup

### 1. Install dependencies
`ash
pip install -r requirements.txt
`

### 2. Download OCTDL dataset
Place the dataset in OCTDL/OCTDL/ with subfolders: AMD/ DME/ ERM/ NO/ RAO/ RVO/ VID/

Dataset: https://www.kaggle.com/datasets/

### 3. Train the model
`ash
python train.py
`

### 4. Run the app
`ash
python app.py
`
Open http://127.0.0.1:5000

## Tech Stack
- **Backend**: Flask, SQLAlchemy, TensorFlow 2.16
- **Frontend**: HTML5, CSS3, Chart.js, Font Awesome
- **ML**: MobileNetV2, ImageDataGenerator, TTA

## Project Structure
`
Eye Disease Detection/
+-- app.py              # Flask application
+-- train.py            # Training script
+-- ml_utils.py         # Inference wrapper
+-- database.py         # SQLAlchemy models
+-- requirements.txt
+-- static/
¦   +-- css/style.css
¦   +-- js/script.js
¦   +-- sample_images/  # Sample OCT scans per class
¦   +-- data/metrics.json
+-- templates/
¦   +-- base.html
¦   +-- index.html
¦   +-- information.html
¦   +-- predict.html
¦   +-- result.html
¦   +-- analysis.html
¦   +-- history.html
+-- plots/              # Training plots
`

## Disclaimer
This tool is for informational/educational purposes only. Not a substitute for professional medical advice.
