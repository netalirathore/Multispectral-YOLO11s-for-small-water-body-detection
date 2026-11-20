# Multispectral YOLO11s for Small Water Body Detection

<p align="center">
  <b>Deep Learning-Based Detection of Small Water Bodies Using Multispectral UAV Imagery</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLO11s-blue" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" />
  <img src="https://img.shields.io/badge/Task-Object%20Detection-green" />
  <img src="https://img.shields.io/badge/Domain-Remote%20Sensing-orange" />
</p>

## Overview

Detecting small water bodies from aerial imagery is challenging due to their limited spatial extent, spectral similarities with surrounding objects, varying illumination conditions, and complex environmental backgrounds.

This project develops a **multispectral object detection framework based on YOLO11s** to improve the detection of small water bodies using information from multiple spectral sources.

Unlike conventional RGB-based object detection, the proposed approach incorporates additional spectral information, including **Near-Infrared (NIR)** and **Normalized Difference Water Index (NDWI)**, to investigate how different spectral-band combinations influence detection performance.

The framework extends the YOLO11s architecture to support different input channel configurations and provides a common experimental pipeline for training and evaluating multispectral detection models.

## Key Features

- **Custom Multispectral YOLO11s:** Extends the YOLO11s architecture to accept multispectral inputs instead of conventional three-channel RGB images.
- **Multiple Spectral Configurations:** Supports RGB, RGB + NIR, RGB + NDWI, and RGB + NIR + NDWI experiments.
- **Custom Dataset Pipeline:** Implements multispectral image loading, preprocessing, synchronized transformations, and YOLO-format bounding-box annotations.
- **Controlled Experiments:** Enables comparison of different spectral combinations using consistent experimental configurations.
- **Data Augmentation:** Supports horizontal flipping, vertical flipping, and 90-degree rotations while maintaining bounding-box alignment.
- **Comprehensive Evaluation:** Supports object-detection metrics, including Precision, Recall, F1-score, mAP@50, and mAP@50-95, with per-class evaluation.

## Methodology

The framework investigates whether additional spectral information can improve small water body detection compared with conventional RGB imagery.

### 1. Multispectral Input Configurations

Four experimental configurations are implemented:

| Experiment | Input Bands | Input Channels |
|---|---|---:|
| E0 | RGB | 3 |
| E1 | RGB + NIR | 4 |
| E2 | RGB + NDWI | 4 |
| E3 | RGB + NIR + NDWI | 5 |

These configurations allow the contribution of NIR and NDWI information to be investigated independently and jointly.

### 2. Model Architecture

The project uses YOLO11s as its base object detection architecture.

A custom multispectral adaptation is implemented to accommodate varying numbers of input channels.

The model supports:
- Loading pretrained YOLO11s weights.
- Adapting the input layer for 3-, 4-, and 5-channel imagery.
- Configuring the detection head for four object classes.
- Training and evaluating models under different spectral configurations.

### 3. Dataset and Object Classes

The dataset consists of annotated UAV imagery collected in Binh Duong Province, Vietnam, using a DJI Phantom 4 Multispectral (P4M) drone.

The detection task considers four classes:

| Class ID | Class Name |
|---|---|
| 0 | Visible Water |
| 1 | Invisible Water |
| 2 | Irrigated Water |
| 3 | Building |

The annotations follow the YOLO bounding-box format:

`class_id x_center y_center width height`

Bounding-box coordinates are normalized relative to image dimensions.

### 4. Training and Evaluation

Each spectral configuration is processed through the corresponding multispectral dataset pipeline.

The framework supports:
- Training, validation, and testing splits.
- Consistent geometric augmentation across corresponding spectral inputs.
- Model checkpoint handling.
- Test-set evaluation.
- Per-class and overall performance analysis.
- Generation of evaluation plots and result summaries.

## Repository Structure

```text
Multispectral-YOLO11s-for-small-water-body-detection/
│
├── Training/
│   └──                 # Training-related files and configurations
│
├── multispectral_experiment.py
│                       # Experimental execution and evaluation
│
├── multispectral_model.py
│                       # Multispectral YOLO11s initialization
│
├── multispectral_moudule.py
│                       # Dataset, preprocessing and augmentation
│
├── multispectral_trainer.py
│                       # Custom training implementation
│
├── ultralytics_dataset_adapter.py
│                       # Dataset integration with Ultralytics
│
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/netalirathore/Multispectral-YOLO11s-for-small-water-body-detection.git

cd Multispectral-YOLO11s-for-small-water-body-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

The implementation uses PyTorch and the Ultralytics YOLO framework, along with supporting scientific Python libraries.

```bash
pip install torch torchvision
pip install ultralytics
pip install numpy pandas pillow matplotlib
```

Install any additional dependencies required by the training scripts.

## Dataset Preparation

The dataset must be prepared in the directory structure expected by the custom dataset loader.

The implementation uses experiment-specific directories and CSV manifests to associate RGB images, NIR images, NDWI images, and their corresponding annotations.

A conceptual structure is shown below:

```text
Prepared_Multispectral/
│
└── experiments/
    │
    ├── E0_RGB/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    │
    ├── E1_RGB_NIR/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    │
    ├── E2_RGB_NDWI/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    │
    └── E3_RGB_NIR_NDWI/
        ├── train.csv
        ├── val.csv
        └── test.csv
```

Each manifest contains the paths required by its corresponding experiment, including:

- `image_id`
- `rgb_path`
- `nir_path` (where applicable)
- `ndwi_path` (where applicable)
- `label_path`

The image paths must point to valid files, and annotations must follow the expected YOLO format.

**Note:** The dataset is not included in this repository. You must prepare the dataset and configure the project paths before running the experiments.

## Experimental Comparison

The primary objective is to study the impact of different spectral combinations on small water body detection.

The experiments are designed to answer the following questions:

1. How well does RGB-only YOLO11s perform on small water body detection?
2. Does incorporating NIR information improve detection performance?
3. Does NDWI provide additional information for distinguishing water bodies from surrounding objects?
4. Does combining RGB, NIR, and NDWI provide better detection performance than individual spectral combinations?

Performance is evaluated using:

| Metric | Description |
|---|---|
| Precision | Proportion of predicted detections that are correct |
| Recall | Proportion of ground-truth objects successfully detected |
| F1-score | Harmonic mean of Precision and Recall |
| mAP@50 | Mean Average Precision at IoU 0.50 |
| mAP@50-95 | Mean Average Precision across IoU thresholds from 0.50 to 0.95 |

## Results

The experiments are designed to compare the detection performance of the four spectral configurations.

| Experiment | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| E0: RGB | TBD | TBD | TBD | TBD |
| E1: RGB + NIR | TBD | TBD | TBD | TBD |
| E2: RGB + NDWI | TBD | TBD | TBD | TBD |
| E3: RGB + NIR + NDWI | TBD | TBD | TBD | TBD |

*Replace TBD with the final test-set results after completing the experiments.*

## Technologies Used

- **Python** – Primary programming language.
- **PyTorch** – Deep learning framework.
- **Ultralytics YOLO11** – Base object detection architecture.
- **NumPy** – Numerical operations and image processing.
- **Pandas** – Dataset manifests and data management.
- **Matplotlib** – Visualization of evaluation results.

## Future Work

Potential directions for further development include:

- Exploring alternative multispectral fusion strategies.
- Improving detection of extremely small and low-visibility water bodies.
- Investigating additional spectral indices and remote-sensing features.
- Evaluating model generalization across different geographical regions and UAV datasets.
- Exploring lightweight deployment for practical UAV-based monitoring applications.

## Acknowledgements

This project builds upon the YOLO11 object detection framework and extends its application to multispectral remote-sensing imagery.

## License

A license has not yet been specified. Please add an appropriate license before distributing or reusing this repository.
