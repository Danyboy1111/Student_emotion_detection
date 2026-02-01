# Student Emotion Detection

## Project overview

Student Emotion Detection is a research and demonstration project that detects students' emotions from images or webcam video frames using deep learning. It contains code for dataset preparation, model training, evaluation, and inference (real-time webcam). The project is designed to be reproducible and easy to run for development and experimentation.

## Features

- Face detection and alignment pre-processing
- Emotion classification (e.g., happy, sad, angry, surprised, neutral)
- Training scripts with configurable hyperparameters
- Evaluation and metrics reporting
- Real-time webcam inference demo
- Clear project structure and example notebooks

## Table of contents

- Installation
- Requirements
- Project structure
- Dataset
- Usage
  - Training
  - Evaluation
  - Inference (webcam)
- Model architecture
- Configuration
- Tips for improving performance
- Troubleshooting
- Contributing
- License
- Contact

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Danyboy1111/Student_emotion_detection.git
   cd Student_emotion_detection
   ```

2. Create a Python virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate    # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

If there is no requirements.txt, install common dependencies used in emotion detection projects:

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn tensorflow keras torch torchvision tqdm seaborn
```

## Requirements

- Python 3.8+
- GPU recommended for training (NVIDIA + CUDA)
- At least 8GB RAM

## Project structure

A suggested project layout (actual layout in the repo may differ):

````markdown name=PROJECT_STRUCTURE.md
```text
Student_emotion_detection/
├── data/                   # datasets, raw and processed
├── notebooks/              # exploratory notebooks and demos
├── src/                    # source code (preprocessing, models, training, inference)
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
├── models/                 # saved model checkpoints
├── requirements.txt
├── README.md
└── LICENSE
```
````

## Dataset

This project does not include copyrighted datasets. You can use common public emotion datasets such as FER2013, RAF-DB, or CK+.

- FER2013: available on Kaggle and contains grayscale 48x48 face images labelled with emotion categories.
- RAF-DB: variety of real-world facial expressions with bounding boxes and labels.

Place your dataset in the `data/` folder and update paths in configuration files or scripts.

## Usage

### Training

1. Prepare dataset and run preprocessing to extract faces and save aligned crops.

   ```bash
   python src/preprocessing.py --input_dir data/raw --output_dir data/processed --detector mtcnn
   ```

2. Train the model:

   ```bash
   python src/train.py --data_dir data/processed --epochs 50 --batch_size 64 --lr 1e-3
   ```

Check `src/train.py` for available CLI arguments and configuration file support.

### Evaluation

```bash
python src/evaluate.py --data_dir data/processed --model checkpoints/best_model.pth
```

Evaluation prints class-wise accuracy, confusion matrix and saves metrics to `results/`.

### Inference (webcam)

Run a demo that opens your webcam and displays predicted emotion labels in real-time:

```bash
python src/inference.py --model checkpoints/best_model.pth --camera 0
```

## Model architecture

Common architectures used:

- Lightweight CNNs (small conv + dense layers)
- Transfer learning from ImageNet backbones (ResNet, MobileNet, EfficientNet)

Check `src/model.py` to see which architecture this repo implements and how to switch backbones.

## Configuration

Configuration options are usually available either as CLI arguments or in a YAML/JSON config file. Typical settings:

- data paths
- model backbone and output classes
- training hyperparameters (epochs, batch size, learning rate)
- augmentation settings

## Tips for improving performance

- Use data augmentation (random crops, horizontal flip, brightness/contrast jitter)
- Balance classes with oversampling or weighted loss
- Use pretrained backbones and fine-tune with lower learning rates
- Enforce face alignment and normalization
- Experiment with focal loss or label smoothing

## Troubleshooting

- If OpenCV cannot access your camera on Linux, try running with sudo or check permissions.
- If CUDA is not detected, ensure GPU drivers and CUDA toolkit match your PyTorch/TensorFlow build.
- Check shapes in your data pipeline when you see size mismatch errors.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes and open a pull request

Please follow PEP8 and include tests for new features where possible.

## License

Specify an open-source license for your project. If you need a suggestion, consider MIT or Apache-2.0.

## Contact

Created by Danyboy1111. For questions or help, open an issue or contact via GitHub profile: https://github.com/Danyboy1111
