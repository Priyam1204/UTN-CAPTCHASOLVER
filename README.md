<div align="center">
  <img src="https://www.strichpunkt-design.de/storage/app/media/work/technische-universitaet-nuernberg-corporate-design-corporate-identity/technische-universitaet-nuernberg-corporate-design-corporate-identity-6-1920x1080__automuted--poster.jpg" alt="University of Technology Nuremberg" width="400"/>
</div>

# CAPTCHA Solver: Deep Learning for Computer Vision

An end-to-end CAPTCHA recognition system developed as the **final project** for the *Computer Vision (WiSe 2024/25)* course at the **[University of Technology Nuremberg (UTN)](https://www.utn.de/)**.  
The project explores learning approaches with a ResNet-18 style backbone combined with **two separate classification head structures** (YOLOv8-style detection head and CTC-style sequence recognition head) built from scratch for decoding alphanumeric CAPTCHAs.

## 🌟 Features

- **Hybrid Architectures**: Implemented both a **ResNet18 + YOLO-based detection head** (found in main branch) and a **CTC-sequence recognition model** (found in [ctc-heads branch](https://github.com/Priyam1204/UTN-CAPTCHASOLVER/tree/ctc-branch))

- **Custom Data Pipeline**: Preprocessing, augmentation, and dataset handling for 100k+ CAPTCHA images
- **Evaluation Metrics**: We evaluated the project **Average Levenshtein Distance** on Validation Set. 
- **Scalable Training Loop**: Configurable optimizers (Adam, AdamW, SGD)
- **Experiment Tracking**: Outputs results in JSON for reproducibility and analysis
- **Bonus Tasks**: Extended experiments with degradations and oriented bounding boxes

## 📊 Dataset Used for Project

- **Size**: 100k images (60k train / 20k val / 20k test)  
- **Resolution**: 640×160  
- **Alphabet**: `0–9`, `A–Z`  
- **Annotations**: Ground truth strings & bounding boxes in labels.json (train/val)  
  

## 🏗️ Project Structure


## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- PyTorch (GPU recommended, but CPU fallback supported)
- Dependencies from `requirements.txt`

### Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/Priyam1204/UTN-CAPTCHASOLVER.git
   cd UTN-CAPTCHASOLVER
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training the Model

Train the CAPTCHA solver model using the provided dataset:

```bash
python TrainModel.py \
    --data_dir /path/to/UTN-CV25-Captcha-Dataset/part2/train \
    --epochs 20 \
    --optimizer adamw \
    --lr 0.00003 \
    --weight_decay 1e-4 \
    --device cuda \
    --save_dir ./checkpoints \
    --save_every 5
```

#### Resume Training from Checkpoint

Continue training from a previously saved checkpoint:

```bash
python TrainModel.py \
    --data_dir /path/to/UTN-CV25-Captcha-Dataset/part2/train \
    --resume ./checkpoints/best_model.pth \
    --epochs 30 \
    --save_every 5
```

#### Available Training Arguments

- `--data_dir`: Path to training data directory (required)
- `--epochs`: Number of epochs to train (default: 5)
- `--optimizer`: Optimizer type [adam, adamw, sgd] (default: adam)
- `--lr`: Learning rate (default: 0.00003)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--device`: Device to use [cuda, cpu] (default: cuda)
- `--save_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--resume`: Path to checkpoint to resume from (optional)
- `--save_every`: Save checkpoint every N epochs (default: 2)

### Model Inference

#### Run Inference on Single Image or Folder

```bash
python ModelInference.py \
    --data_dir /path/to/images \
    --num_images 10
```

#### Inference on All Images in Directory

```bash
python ModelInference.py \
    --data_dir /path/to/images
```

#### Available Inference Arguments

- `--data_dir`: Path to directory containing images (required)
- `--num_images`: Number of images to process (default: all images)

#### Inference Output

The inference script generates:
- **JSON Results**: `predictions_conf_0.40.json` with predicted CAPTCHA strings and bounding boxes
- **Visualizations**: Annotated images in `inference_results/` folder showing detected characters with bounding boxes
- **Console Output**: Progress information and predicted strings

Example output format:
```json
[
    {
        "height": 160,
        "width": 640,
        "image_id": "000001",
        "captcha_string": "ABC123",
        "annotations": [
            {
                "bbox": [45.2, 32.1, 78.9, 89.4],
                "category_id": 10
            }
        ]
    }
]
```

### Model Configuration

The model uses the following default configuration:
- **Input Size**: 640×160 pixels
- **Grid Size**: 10×40 (height×width)
- **Classes**: 36 (0-9, A-Z)
- **Confidence Threshold**: 0.4
- **IoU Threshold**: 0.3

## 📊 Project Submission

The results for the ResNet18 and YOLOv8 style head Part2 test set can be found in [Part2 Result](https://github.com/Priyam1204/UTN-CAPTCHASOLVER/tree/main/Part2%20Result) calculated on the provided dataset. The checkpoints used for evaluation are found in [ModelWeights](https://github.com/Priyam1204/UTN-CAPTCHASOLVER/tree/main/ModelWeights).


## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [priyammishra1204@gmail.com] or [priyam.mishra@utn.de]
- Contact: [noas.shaalan@gmail.com] or [noas.shaalan@utn.de] 
- Contact: [Adam.Jen.Khai.Lo@utn.de]