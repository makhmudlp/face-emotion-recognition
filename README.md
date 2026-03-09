# Face Emotion Recognition

Real-time facial emotion detection using fine-tuned ResNet18 and MediaPipe, deployed on Hugging Face Spaces.

🚀 **Live Demo:** [huggingface.co/spaces/makhmudlp/face-emotion-recognition](https://huggingface.co/spaces/makhmudlp/face-emotion-recognition)

---

## Demo

Real-time webcam inference with 468-point MediaPipe face mesh and emotion prediction.

---

## Model

### Architecture: Fine-tuned ResNet18
- **Base model:** ResNet18 pretrained on ImageNet
- **Input modification:** `conv1` changed from 3→1 channel (grayscale), weights averaged from RGB channels
- **Classifier head:** `Linear(512 → 256) → ReLU → Dropout(0.4) → Linear(256 → 7)`
- **Parameters:** ~11M total

### Dataset: FER2013
- 35,887 grayscale 48×48 images across 7 emotion classes
- **Train:** 28,709 | **Test:** 7,178
- Classes: `angry, disgust, fear, happy, neutral, sad, surprise`
- Class imbalance handled with weighted CrossEntropyLoss

### Training Strategy (3-phase fine-tuning)
| Phase | Epochs | LR | Unfrozen Layers | Val Acc |
|-------|--------|----|-----------------|---------|
| 1 | 5 | 0.001 | Head only | ~48% |
| 2 | 10 | 0.0001 | layer3 + layer4 | ~62% |
| 3 | 10 | 0.00001 | All layers | **68.35%** |

### Training Details
- **Device:** Apple MPS (M-series Mac)
- **Optimizer:** Adam
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Augmentation:** RandomHorizontalFlip, RandomRotation(5°)

---

## Project Structure
```
├── models/
│   ├── resnet.py         ← EmotionResNet architecture
│   ├── best_resnet.pth   ← trained weights (68.35% val acc)
├── inference.py          ← local real-time webcam demo
├── all.ipynb             ← training notebook
├── requirements.txt
```

---

## Local Setup
```bash
git clone https://github.com/makhmudlp/face-emotion-recognition
cd face-emotion-recognition

pip install -r requirements.txt

# Download FER2013 dataset
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/

# Run live inference
python inference.py
```

**Requirements:** Python 3.11+, PyTorch, MediaPipe 0.10.32, OpenCV

---

## Results

| Metric | Value |
|--------|-------|
| Val Accuracy | **68.35%** |
| Dataset | FER2013 |
| Model | ResNet18 fine-tuned |

---

## Tech Stack
`PyTorch` `ResNet18` `MediaPipe` `OpenCV` `Gradio` `Hugging Face Spaces`
