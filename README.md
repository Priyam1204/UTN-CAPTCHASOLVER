<div align="center">
  <img src="https://www.strichpunkt-design.de/storage/app/media/work/technische-universitaet-nuernberg-corporate-design-corporate-identity/technische-universitaet-nuernberg-corporate-design-corporate-identity-6-1920x1080__automuted--poster.jpg" alt="University of Technology Nuremberg" width="400"/>
</div>

# CAPTCHA Solver: Deep Learning for Computer Vision

An end-to-end CAPTCHA recognition system developed as the **final project** for the *Computer Vision (WiSe 2024/25)* course at the **University of Technology Nuremberg (UTN)**.  
The project explores multiple deep learning approaches for decoding alphanumeric CAPTCHAs with degradations, distortions, and distractors.

## 🌟 Features

- **Hybrid Architectures**: Implemented both a **ResNet18 + YOLO-based detection head** (found in main branch) and a **CTC-sequence recognition model** (found in ctc-heads branch)

- **Custom Data Pipeline**: Preprocessing, augmentation, and dataset handling for 100k+ CAPTCHA images
- **Evaluation Metrics**: Supports **Levenshtein Error Rate (LER)** and **mAP@0.5:0.95** (for bounding box tasks)
- **Scalable Training Loop**: Configurable optimizers (Adam, AdamW, SGD), cosine learning rate scheduling, and warmup
- **Experiment Tracking**: Outputs results in JSON for reproducibility and analysis
- **Bonus Tasks**: Extended experiments with degradations and oriented bounding boxes

## 📊 Dataset

- **Size**: 100k images (60k train / 20k val / 20k test)  
- **Resolution**: 640×160  
- **Alphabet**: `0–9`, `A–Z`  
- **Annotations**: Ground truth strings & bounding boxes (train/val)  
- **Variations**: Rotation, shear, noise, distractors, complex backgrounds  

## 🏗️ Project Structure

├── Evaluate_CTC.py ├── Evaluation Results │   ├── ctc_part2_test_evaluation_results.json │   ├── ctc_part3_test_evaluation_results.json │   └── ctc_part4_test_evalaution_results.json ├── README.md ├── requirements.txt ├── Src │   ├── Data │   │   ├── Collate_CTC.py │   │   ├── Collate.py │   │   ├── DataLoader.py │   │   ├── DataSet.py │   │   ├── NoisePolicy.py │   │   ├── __pycache__ │   │   │   ├── Collate.cpython-310.pyc │   │   │   ├── Collate_CTC.cpython-310.pyc │   │   │   ├── DataLoader.cpython-310.pyc │   │   │   ├── DataSet.cpython-310.pyc │   │   │   └── Transform.cpython-310.pyc │   │   └── Transform.py │   ├── Inference │   │   ├── __pycache__ │   │   │   ├── VisualEvaluator.cpython-310.pyc │   │   │   ├── VisualEvaluator.cpython-311.pyc │   │   │   └── VisualEvaluator_CTC.cpython-310.pyc │   │   ├── Validation_Loss_Checkpoints.py │   │   └── VisualEvaluator_CTC.py │   ├── __init__.py │   ├── Model │   │   ├── Backbone.py │   │   ├── CTC_Head.py │   │   ├── CTC_Loss.py │   │   ├── CTC_Model.py │   │   └── __pycache__ │   │   ├── Backbone.cpython-310.pyc │   │   ├── CTC_Head.cpython-310.pyc │   │   ├── CTC_Loss.cpython-310.pyc │   │   ├── CTC_Model.cpython-310.pyc │   │   ├── Head.cpython-310.pyc │   │   └── Model.cpython-310.pyc │   ├── __pycache__ │   │   ├── __init__.cpython-310.pyc │   │   └── __init__.cpython-311.pyc │   ├── Training │   │   ├── __pycache__ │   │   │   └── Trainer_CTC.cpython-310.pyc │   │   └── Trainer_CTC.py │   └── Utils │   ├── BoundingBoxVisualization.py │   ├── Decoder.py │   ├── Evaluations.py │   ├── __init__.py │   ├── IoU.py │   ├── NMS.py │   ├── __pycache__ │   │   ├── Decoder.cpython-310.pyc │   │   ├── __init__.cpython-310.pyc │   │   ├── IoU.cpython-310.pyc │   │   ├── NMS.cpython-310.pyc │   │   ├── TargetPreparer.cpython-310.pyc │   │   └── WeightInitializer.cpython-310.pyc │   ├── TargetPreparer.py │   └── WeightInitializer.py └── Trainer_CTC.py

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- PyTorch (GPU recommended, but CPU fallback supported)
- Dependencies from `requirements.txt`

### Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/your-username/UTN-CAPTCHASOLVER.git
   cd UTN-CAPTCHASOLVER
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running Training

Run the CTC trainer on the provided CAPTCHA dataset:
```bash
python Src/Training/Trainer_CTC.py \
    --data_dir /path/to/UTN-CV25-Captcha-Dataset/part2/train \
    --val_dir /path/to/UTN-CV25-Captcha-Dataset/part2/val \
    --epochs 20 \
    --batch_size 16 \
    --base_lr 3e-4
```

### Model Evaluation

#### Evaluate a single checkpoint
```bash
python Evaluate_CTC.py \
    --model_ckpt ./checkpoints/checkpoint_epoch_10.pth \
    --data_dir /path/to/UTN-CV25-Captcha-Dataset/part2/val \
    --json_path results/part2_epoch10_eval.json
```

Outputs:

Console summary (Accuracy, LER, Levenshtein/20000)

JSON file in dataset submission format

#### Evaluate all checkpoints in a folder
```bash
python Src/Inference/Validation_Loss_Checkpoints.py \
    --checkpoint_dir ./checkpoints \
    --data_dir /path/to/UTN-CV25-Captcha-Dataset/part2/val \
    --out_json results/ctc_checkpoint_sweep.json \
    --out_csv results/ctc_checkpoint_sweep.csv
```

## 📊 Evaluation Metrics

The evaluation results on the provided dataset can be found in the Evaluation Results folder. The checkpoints used for Evaluation are found in the releases part of the repo.

## 🙏 Acknowledgments

- Prof. Dr. Eddy Ilg from University of Technology, Nuremberg for Captcha Dataset


## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [noas.shaalan@gmail.com] or [noas.shaalan@utn.de] 

---

**Keywords**: CAPTCHA Solver, Optical Character Recognition (OCR), Connectionist Temporal Classification (CTC), Deep Learning, Computer Vision, PyTorch, ResNet-18, Synthetic CAPTCHA Dataset
