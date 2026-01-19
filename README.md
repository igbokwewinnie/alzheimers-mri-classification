# Alzheimer’s Disease Classification using Transfer Learning (MobileNetV2)

## Project Overview

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder that affects memory and cognitive function. Early and accurate classification of Alzheimer’s stages can support better clinical decision-making and treatment planning.

This project explores **multi-class classification of Alzheimer’s stages from MRI brain scans** using **deep learning and transfer learning**. The work was carried out as part of a structured data project challenge and designed as a **portfolio-ready, end-to-end machine learning project**.

**Final Achievement: 89.90% test accuracy** 

---

## Dataset

* **Source:** Kaggle – [Alzheimer’s Multi-Class Dataset (Equal & Augmented)](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented)
* **Data Type:** MRI brain scan images
* **Number of Images:** ~44,000
* **Classes (4):**

  * NonDemented
  * VeryMildDemented
  * MildDemented
  * ModerateDemented

The dataset is pre-processed and augmented to improve class balance.

---

## Project Structure & Workflow

### 1. Exploratory Data Analysis (EDA)

Performed entirely in **Google Colab**

Key EDA steps included:

* Class distribution analysis
* Visual inspection of MRI samples per class
* Pixel intensity and brightness distribution analysis
* Feature-level statistics (mean intensity, contrast)
* Comparison of distributions across Alzheimer’s stages

### 2. Model Development

Model training and hyperparameter tuning were performed in a **Kaggle Notebook** to leverage free GPU resources.

Key techniques:

* PyTorch framework
* Transfer Learning with **MobileNetV2 (ImageNet weights)**
* Two-phase training strategy: frozen backbone - fine-tuning
* Data augmentation (rotation, flipping, contrast adjustment)
* Mixed precision training for 2-3x speedup
* Model checkpointing to preserve best models

---

## Model Architecture

* **Base Model:** MobileNetV2 (pre-trained on ImageNet)
* **Training Strategy:**

 **Phase 1: Frozen Backbone Training**
- Freeze all MobileNetV2 feature layers
- Train only the custom classifier head
- Purpose: Find optimal hyperparameters quickly
- Result: ~63% validation accuracy

**Phase 2: Fine-Tuning**
- Unfreeze last 3 layers of MobileNetV2
- Use differential learning rates:
- Purpose: Adapt pre-trained features to medical imaging
- Result: 90.54% validation accuracy (+27 percentage points)

---

## Training & Evaluation

### Metrics Used
**Final Model Performance**

Metric              Validation       Test
* Accuracy       90.54%           89.90%
* Precision      0.8407           0.90
* Recall         0.8401           0.90


### Hyperparameter Tuning

The following hyperparameters were systematically explored:

* Learning Rate: 0.0001, 0.001, 0.01
* Batch Size: 16, 32, 64
* Dropout Probability: 0.0, 0.3, 0.5

Experiments were conducted using controlled loops to ensure fair comparison.

### Optimal Hyperparameters
- Learning Rate: 0.001
- Batch Size: 64
- Dropout: 0.0

---

## Tools & Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab (EDA)
* Kaggle Notebooks (GPU training)

---

## Key Findings

* Pixel intensity distributions across classes were similar, indicating no severe brightness bias.
* Dropout of 0.0 provided best balance between underfitting and overfitting
* Transfer learning significantly reduced training time while achieving reasonable validation performance.

---

## Notes

* EDA and modeling were intentionally separated across platforms to optimize workflow.
* This repository focuses on **clarity, reproducibility, and learning outcomes** rather than leaderboard optimization.

---

## Future Improvements

* Unfreeze deeper layers for fine-tuning
* Experiment with EfficientNet architectures
* Add Grad-CAM visual explanations
* Perform cross-validation

---
