# Optimized EfficientNet 3Class CNN Model
### Cat vs Dog vs Car Classification with 95%+ MCC Target

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/AthithyaKrishnaa/EfficientNet_3Class_CNN)

> **High-reliability image classifier** combining EfficientNet ensemble, Test-Time Augmentation (TTA), and Conformal Prediction for calibrated uncertainty quantification in safety-critical applications.

---

## 🔍 Overview

This project implements a **production-grade 3-class image classifier** (Cat / Dog / Car) designed for scenarios requiring:
- **High accuracy** (MCC ≥ 0.95)
- **Calibrated uncertainty** quantification
- **Transparent error analysis**
- **Medical-AI-style reliability**

The system uses:
- **Ensemble of 3 EfficientNet models** (2× B0 + 1× B1)
- **Advanced data augmentation** (including Mixup)
- **Test-Time Augmentation** (15× per model = 45 predictions averaged)
- **Conformal Prediction** for coverage guarantees

---

## ✨ Key Features

### 🧠 Model Architecture
- **Transfer Learning**: Pre-trained EfficientNet backbones (ImageNet)
- **Ensemble Diversity**: Multiple architectures + different random seeds
- **Heavy Regularization**: L2 reg + BatchNorm + Dropout (0.3-0.5)

### 📊 Data Pipeline
- **4,500 images**: 1,500 per class (Cat, Dog, Car)
- **High resolution**: 128×128 pixels (4× more than baseline)
- **Quality filtering**: Strict variance and brightness thresholds
- **Image enhancement**: Sharpening kernels applied via OpenCV

### 🎲 Advanced Training
- **Class balancing**: Computed weights for imbalanced datasets
- **Dynamic learning rate**: ReduceLROnPlateau (0.0001 → 1e-7)
- **Early stopping**: Patience-based with best model restoration
- **Augmentation**: Flip, Rotation, Zoom, Translation, Contrast, Brightness + Mixup

### 🔬 Uncertainty Quantification
- **Conformal Prediction**: Mathematically guaranteed coverage (95%, 90%, 80%)
- **Prediction Sets**: Single-class (certain) vs multi-class (uncertain)
- **Hard Negative Mining**: Automatic identification of difficult cases

---

## 🏗️ Architecture

```
Input (128×128×3)
    ↓
Data Augmentation (Keras Sequential)
    ↓
EfficientNet Preprocessing
    ↓
┌─────────────────────────────────────┐
│  Ensemble (3 models)                │
│  • EfficientNetB0 (seed=42)         │
│  • EfficientNetB0 (seed=52)         │
│  • EfficientNetB1 (seed=62)         │
└─────────────────────────────────────┘
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) + BatchNorm + Dropout(0.5)
    ↓
Dense(256) + BatchNorm + Dropout(0.3)
    ↓
Dense(3, softmax)
    ↓
Test-Time Augmentation (15×)
    ↓
Ensemble Averaging (3 models)
    ↓
Conformal Prediction Sets
```

---

## 📈 Results

### Overall Performance

| Metric | Baseline (Single) | Ensemble (TTA) | Conformal (90%) |
|--------|------------------|----------------|-----------------|
| **Accuracy** | 85-90% | 94.4% | 94.4% |
| **MCC** | 0.905 | **0.917** | **0.917** |
| **Coverage** | N/A | N/A | 94.4% (target: 90%) |

### Per-Class Metrics

| Class | Sensitivity | Specificity | Precision | F1-Score | MCC |
|-------|------------|-------------|-----------|----------|-----|
| **Cat** | 0.889 | 0.972 | 0.941 | 0.914 | 0.874 |
| **Dog** | 0.944 | 0.944 | 0.895 | 0.919 | 0.877 |
| **Car** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### Confusion Matrix

```
           Predicted
           Cat  Dog  Car
Actual Cat [160  20   0]   89% recall
       Dog [ 10 170   0]   94% recall
       Car [  0   0 180]  100% recall
```

**Key Insights:**
- ✅ **Car class**: Perfect separation (0 errors)
- ⚠️ **Cat↔Dog confusion**: All 30 errors occur between animal classes
- 🎯 **Total errors**: 30/540 (5.6% error rate)

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
CUDA 11.x (for GPU support)
```

### Setup

#### Option 1: Google Colab (Recommended)
Click the badge at the top to open directly in Colab. All dependencies pre-installed!

#### Option 2: Local Installation
```bash
# Clone repository
git clone https://github.com/yourusername/conformal-3class-ensemble.git
cd conformal-3class-ensemble

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt
```txt
tensorflow>=2.19.0
numpy>=1.23.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.2.0
opencv-python>=4.7.0
Pillow>=9.4.0
requests>=2.28.0
```

---

## 💻 Usage

### Quick Start

```python
# Run the complete pipeline
jupyter notebook EfficientNet_3Class_CNN.ipynb
```

### Step-by-Step Execution

```python
# 1. Data Loading (automatic download)
# Downloads Microsoft Cats&Dogs + CIFAR-10 Car images

# 2. Training Ensemble
# Trains 3 models with different architectures/seeds
# Expected time: 30-60 minutes on GPU

# 3. Evaluation with TTA
# 45 predictions per sample (3 models × 15 augmentations)
# Expected time: 5-7 minutes

# 4. Conformal Prediction
# Computes prediction sets at 95%, 90%, 80% confidence

# 5. Generate Visualizations
# Creates 4 comprehensive analysis figures
```

### Custom Dataset

```python
# Modify data loading function
def load_custom_images(path, label, max_images=1500):
    images, labels = [], []
    for filename in os.listdir(path):
        img = keras.preprocessing.image.load_img(
            os.path.join(path, filename),
            target_size=(128, 128)
        )
        img_array = keras.preprocessing.image.img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(label)
    return images, labels
```

---

## 🔬 Methodology

### 1. Data Preparation
- **Sources**: 
  - Cats/Dogs: [Microsoft PetImages](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
  - Cars: CIFAR-10 "automobile" class
- **Preprocessing**:
  - Resize to 128×128
  - Normalize to [0,1]
  - Quality filtering (std > 0.04, mean > 0.12)
  - Sharpening kernel application

### 2. Model Training
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: Sparse categorical crossentropy
- **Callbacks**: 
  - EarlyStopping (patience=12, monitor=val_loss)
  - ReduceLROnPlateau (factor=0.5, patience=5)
  - ModelCheckpoint (save best on val_accuracy)
- **Epochs**: Up to 60 (typically stops at ~45-50)
- **Batch size**: 32

### 3. Test-Time Augmentation
For each test sample:
```
Original prediction
+ 14 augmented predictions (flip, rotate, zoom, etc.)
= Average of 15 predictions per model
× 3 models in ensemble
= 45 total predictions averaged
```

### 4. Conformal Prediction
**Algorithm**:
1. Compute non-conformity scores on calibration set:
   ```
   score = 1 - P(true_class)
   ```
2. For significance level α (e.g., 0.10 for 90% confidence):
   ```
   threshold = quantile(scores, (n+1)(1-α)/n)
   ```
3. Build prediction set for test sample:
   ```
   Include class k if: 1 - P(k) ≤ threshold
   ```

**Properties**:
- ✅ Coverage guarantee: ≥ (1-α) with high probability
- ✅ Distribution-free: no assumptions about data
- ✅ Post-hoc: works with any trained model

---

## 📊 Visualizations

The notebook generates 4 comprehensive figures:

### 1. Confusion Matrix
<img width="807" height="788" alt="ensemble-confusion-matrix" src="https://github.com/user-attachments/assets/0fe867ba-5313-4655-a329-b7a8772bedeb" />

*Image showing Ensemble Confusion Matrix*

### 2. Performance Metrics
<img width="1583" height="1190" alt="performance-metrics" src="https://github.com/user-attachments/assets/d1e6c630-f34d-469b-ba1b-28341bef4b5c" />
*Four subplots: (a) Per-class metrics, (b) Conformal coverage, (c) Set size distribution, (d) MCC comparison*

### 3. Error Analysis
<img width="1311" height="859" alt="error-analysis" src="https://github.com/user-attachments/assets/9b612924-eb81-4997-afdd-6e1e50c5941f" />
*Three subplots: (a) Errors by Class, (b) Confusion patterns, (c) Confidence: Correct vs Wrong*

### 4. Sample Predictions
<img width="1598" height="1019" alt="sample-predictions" src="https://github.com/user-attachments/assets/2774f64d-652e-489e-9ca8-86b1155b45f6" />
*Visual grid showing: high/low confidence correct, wrong predictions, conformal ambiguous cases*

---

## 🎓 Key Concepts Explained

### Matthews Correlation Coefficient (MCC)
- **Range**: -1 to +1
- **Interpretation**: 
  - +1 = perfect prediction
  - 0 = random guessing
  - -1 = perfect disagreement
- **Advantage**: Balanced metric for multi-class problems, accounts for all TP/TN/FP/FN

### Conformal Prediction
- **Purpose**: Provide calibrated uncertainty with mathematical guarantees
- **Output**: Prediction sets (not just point predictions)
- **Guarantee**: Coverage ≥ (1-α) regardless of data distribution
- **Use case**: Medical AI, autonomous vehicles, high-stakes decisions

### Test-Time Augmentation
- **Idea**: Apply augmentations during inference, not just training
- **Benefit**: Reduces variance, improves robustness
- **Trade-off**: Slower inference (15× slower in our case)

---

## 🚀 Improvements & Future Work

### Short-term (MCC 0.95+ target)
- [ ] Increase ensemble to 5 models
- [ ] Use EfficientNetB2/B3 (larger backbones)
- [ ] Add focal loss for hard examples
- [ ] Implement class-conditional augmentation

### Long-term
- [ ] Add Grad-CAM for interpretability
- [ ] Implement active learning loop
- [ ] Deploy as REST API (FastAPI + Docker)
- [ ] Add A/B testing framework
- [ ] Create web demo (Streamlit/Gradio)

---

## Acknowledgments

- **EfficientNet**: Mingxing Tan and Quoc V. Le ([paper](https://arxiv.org/abs/1905.11946))
- **Conformal Prediction**: Vovk et al. ([book](https://link.springer.com/book/10.1007/978-3-031-06649-8))
- **Microsoft PetImages**: [Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- **CIFAR-10**: Alex Krizhevsky ([dataset](https://www.cs.toronto.edu/~kriz/cifar.html))

---

## 📧 Contact

**Mail** - athithyakrishnaam@email.com

Project Link: [https://github.com/yourusername/EfficientNet_3Class_CNN](https://github.com/AthithyaKrishnaa/EfficientNet_3Class_CNN)

---
