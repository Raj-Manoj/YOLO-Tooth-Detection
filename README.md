# 🦷 YOLO-Based Tooth Numbering Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Raj-Manoj/YOLO-Tooth-Detection/blob/main/YOLO_Tooth_Detection.ipynb)

This repository contains the implementation of a **YOLOv8-based object detection model** trained to detect and classify human teeth using the dental numbering system. The model is trained on approximately 500 annotated dental X-ray images and achieves high accuracy in tooth detection and classification.

## 🎯 Overview

Automated tooth detection and numbering is crucial for dental diagnosis and treatment planning. This project implements a state-of-the-art YOLO-based solution that can:

- **Detect individual teeth** in dental X-ray images
- **Classify teeth types** (Incisors, Canines, Premolars, Molars)
- **Apply dental numbering system** for clinical use
- **Provide high accuracy** with mAP@0.5 of ~94.7%

## 📂 Project Structure

```
YOLO-Tooth-Detection/
├── YOLO_Tooth_Detection.ipynb          # Main training notebook (Colab)
├── YOLO_Tooth_Detection_Report.docx    # Detailed project report
├── data.yaml                           # YOLO dataset configuration
├── splits/                             # Dataset splits
│   ├── train/
│   ├── val/
│   └── test/
├── images/                             # Raw dataset images
├── labels/                             # YOLO format annotations
├── runs/                               # Training outputs & results
│   └── detect/
│       └── train/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── confusion_matrix.png
│           ├── PR_curve.png
│           └── results.png
├── predictions/                        # Model predictions
├── requirements.txt                    # Python dependencies
└── README.md                          # Project documentation
```

## 📊 Dataset

### Dataset Overview
- **Size**: ~500 annotated dental X-ray images
- **Format**: YOLO format (class + normalized bounding box coordinates)
- **Classes**: 32 tooth categories based on dental numbering system
- **Split**: 70% train, 20% validation, 10% test

### Tooth Classifications
The model classifies teeth into the following categories:
- **Incisors**: Central Incisors (1,8,9,16,17,24,25,32), Lateral Incisors (2,7,10,15,18,23,26,31)
- **Canines**: (3,6,11,14,19,22,27,30)
- **Premolars**: First Premolars (4,5,12,13,20,21,28,29)
- **Molars**: First, Second, and Third Molars (remaining positions)

### Data Preprocessing
- Images resized to 640x640 pixels
- Normalized bounding box coordinates
- Data augmentation applied during training
- Quality control and annotation verification

## ⚙️ Model Configuration

### Training Setup
```yaml
# Training hyperparameters
model: yolov8s.pt              # Base model (pre-trained on COCO)
image_size: 640                # Input image size
batch_size: 16                 # Batch size for training
epochs: 120                    # Training epochs
optimizer: Auto                # Automatic optimizer selection
lr_scheduler: cosine           # Cosine learning rate scheduler
device: cuda                   # GPU acceleration (Tesla T4)
workers: 2                     # Data loading workers
```

### Model Architecture
- **Backbone**: YOLOv8s (Small variant)
- **Input Resolution**: 640×640 pixels
- **Output Classes**: 32 tooth categories
- **Anchor-free**: Uses anchor-free detection head
- **Loss Function**: Binary cross-entropy + CIoU loss

## 🏆 Results

### Validation Performance
| Metric | Value |
|--------|-------|
| **Precision (P)** | 0.916 |
| **Recall (R)** | 0.922 |
| **mAP@0.5** | 0.947 |
| **mAP@0.5:0.95** | 0.688 |
| **F1-Score** | 0.919 |

### Test Performance
| Metric | Value |
|--------|-------|
| **Precision (P)** | 0.892 |
| **Recall (R)** | 0.920 |
| **mAP@0.5** | 0.938 |
| **mAP@0.5:0.95** | 0.661 |
| **F1-Score** | 0.906 |

### Training Metrics
- **Training Time**: ~2.5 hours on Tesla T4
- **Best Epoch**: 98/120
- **Final Loss**: 0.0234
- **Inference Speed**: ~15ms per image

## 📈 Visualizations

### Training Curves
The model training progress showing loss reduction and mAP improvement over epochs.

### Confusion Matrix
Detailed confusion matrix showing per-class performance and common misclassifications.

### Precision-Recall Curves
PR curves for each tooth class demonstrating model confidence and accuracy trade-offs.

### Sample Predictions
Example detections on test images with bounding boxes and confidence scores.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Raj-Manoj/YOLO-Tooth-Detection.git
cd YOLO-Tooth-Detection

# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install ultralytics>=8.0.0
pip install opencv-python
pip install matplotlib
pip install seaborn
pip install pandas
```

### 2. Dataset Setup

```bash
# Ensure your dataset follows this structure:
# data.yaml should point to your dataset paths
python -c "
import yaml
config = {
    'train': './splits/train',
    'val': './splits/val', 
    'test': './splits/test',
    'nc': 32,  # number of classes
    'names': ['tooth_1', 'tooth_2', ..., 'tooth_32']
}
with open('data.yaml', 'w') as f:
    yaml.dump(config, f)
"
```

### 3. Training

```bash
# Train the model
yolo detect train \
    data=data.yaml \
    model=yolov8s.pt \
    imgsz=640 \
    batch=16 \
    epochs=120 \
    device=0 \
    project=runs/detect \
    name=tooth_detection
```

### 4. Inference

```bash
# Run inference on new images
yolo predict \
    model=runs/detect/tooth_detection/weights/best.pt \
    source=path/to/your/images \
    save=True \
    conf=0.5 \
    device=0
```

### 5. Validation

```bash
# Validate model performance
yolo val \
    model=runs/detect/tooth_detection/weights/best.pt \
    data=data.yaml \
    device=0
```

## 🐍 Python API Usage

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load trained model
model = YOLO('runs/detect/tooth_detection/weights/best.pt')

# Run inference
results = model('path/to/dental_xray.jpg')

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()
        
        print(f"Tooth {class_id + 1}: {confidence:.3f} confidence")
        print(f"Coordinates: {coordinates}")

# Visualize results
annotated_img = results[0].plot()
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

## 📱 Google Colab Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Raj-Manoj/YOLO-Tooth-Detection/blob/main/YOLO_Tooth_Detection.ipynb)

1. Click the Colab badge above
2. Run all cells in sequence
3. Upload your dataset or use the provided sample
4. Train and evaluate the model
5. Download results and trained weights

## 📋 Requirements

```
ultralytics>=8.0.0
opencv-python>=4.5.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
Pillow>=8.3.0
PyYAML>=5.4.0
torch>=1.11.0
torchvision>=0.12.0
```

## 🔧 Hyperparameter Tuning

For optimal results, consider tuning these hyperparameters:

```yaml
# Advanced training configuration
lr0: 0.01                    # initial learning rate
lrf: 0.01                    # final learning rate
momentum: 0.937              # SGD momentum
weight_decay: 0.0005         # optimizer weight decay
warmup_epochs: 3.0           # warmup epochs
warmup_momentum: 0.8         # warmup initial momentum
box: 0.05                    # box loss gain
cls: 0.5                     # cls loss gain
cls_pw: 1.0                  # cls BCELoss positive_weight
obj: 1.0                     # obj loss gain
obj_pw: 1.0                  # obj BCELoss positive_weight
iou_t: 0.20                  # IoU training threshold
anchor_t: 4.0                # anchor-multiple threshold
```

## 📊 Model Performance Analysis

### Per-Class Performance
The model shows excellent performance across all tooth categories:
- **Best performing**: Molars (mAP: 0.95+)
- **Most challenging**: Incisors in crowded areas (mAP: 0.89+)
- **Overall balance**: Good precision-recall trade-off

### Common Issues
- **Overlapping teeth**: Challenging in crowded dental areas
- **Image quality**: Lower performance on poor quality X-rays  
- **Edge cases**: Wisdom teeth and dental work may affect accuracy

### Improvement Suggestions
- Increase dataset size for rare tooth positions
- Add data augmentation for better generalization
- Consider ensemble methods for production use

## 📄 Documentation

### Detailed Report
Complete methodology, results, and analysis are available in:
📑 **[YOLO_Tooth_Detection_Report.docx](./YOLO_Tooth_Detection_Report.docx)**

### Research Paper
This work is based on modern YOLO architectures and dental imaging research:
- YOLOv8: Ultralytics implementation
- Dental numbering systems: ISO 3950, ADA standards
- Medical imaging: DICOM compatibility considerations

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup
```bash
# For development
pip install -e .
pip install pre-commit
pre-commit install
```

## 📌 Future Work

### Short Term
- [ ] Improve dataset size and class balance
- [ ] Experiment with YOLOv8m/l for higher accuracy
- [ ] Add support for additional dental numbering systems
- [ ] Implement model quantization for faster inference

### Long Term  
- [ ] **Web Application**: Deploy as a web service for dentists
- [ ] **Mobile App**: iOS/Android application for chairside use
- [ ] **DICOM Integration**: Direct integration with dental imaging systems
- [ ] **3D Extension**: Extend to 3D dental imaging (CBCT)
- [ ] **Multi-modal**: Combine with clinical notes and patient history

## 🏥 Clinical Applications

This model can be integrated into:
- **Dental Practice Management Systems**
- **Radiology Workflow Software** 
- **Dental Education Platforms**
- **Teledentistry Solutions**
- **Insurance Claim Processing**

## ⚠️ Medical Disclaimer

This software is intended for research and educational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions regarding medical conditions.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Raj Manoj** - Full Stack Developer & AI/ML Enthusiast

🔭 Currently building next-gen web applications  
🌱 Learning Advanced React, Web3 & Cloud Architecture  
👯 Open to collaborate on innovative projects  
💬 Ask me about React, JavaScript, Python, AI/ML  
📫 Reach me at **rajmanojb817@gmail.com**  
⚡ Fun fact: I debug with console.log and I'm proud of it!  

**Connect with me:**
- 📧 Email: rajmanojb817@gmail.com
- 🐙 GitHub: [@Raj-Manoj](https://github.com/Raj-Manoj)
- 💼 LinkedIn: [Connect on LinkedIn](https://https://www.linkedin.com/in/raj-manoj-bytari-948944258/)
- 🌐 Portfolio: [View Projects](https://raj-manoj.github.io)

*"Clean code always looks like it was written by someone who cares"*

## 🙏 Acknowledgments

- **Ultralytics** for the excellent YOLOv8 implementation
- **Dental Dataset Contributors** for providing annotated data
- **Google Colab** for providing free GPU resources
- **Open Source Community** for tools and libraries used

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{yolo_tooth_detection_2024,
  title={YOLO-Based Tooth Numbering Detection},
  author={Raj Manoj},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Raj-Manoj/YOLO-Tooth-Detection}
}
```

---

<div align="center">

### ⭐ If this project helped you, please give it a star! ⭐

**Made with ❤️ for the dental community**

</div>
