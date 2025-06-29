# 📚 E-Waste Classification System - Complete Usage Guide

Welcome to the comprehensive usage guide for the E-Waste Classification System! This guide will walk you through every aspect of using the system, from setup to advanced features.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation & Setup](#installation--setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Training Your Model](#training-your-model)
6. [Using the Web Application](#using-the-web-application)
7. [Command Line Interface](#command-line-interface)
8. [Python API Usage](#python-api-usage)
9. [Interactive Notebooks](#interactive-notebooks)
10. [Advanced Configuration](#advanced-configuration)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)

---

## 🚀 Quick Start

### Option 1: Web Application (Recommended for Beginners)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch web app
streamlit run app.py
```

### Option 2: Complete Training Pipeline
```bash
# Run the complete training pipeline
python train_model.py
```

### Option 3: Interactive Learning
```bash
# Launch Jupyter notebook
jupyter notebook notebooks/E-Waste_Classification_Demo.ipynb
```

---

## 💻 System Requirements

### Minimum Requirements:
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended for training)
- **Storage**: 5GB free space
- **GPU**: Optional but recommended for training (CUDA-compatible)

### Recommended Requirements:
- **Python**: 3.9+
- **RAM**: 16GB+
- **Storage**: 10GB+ free space
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **OS**: Windows 10+, macOS 10.15+, or Linux

---

## 🛠️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd E-Waste-Classification
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv ewaste_env
ewaste_env\Scripts\activate

# macOS/Linux
python -m venv ewaste_env
source ewaste_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Setup Script (Optional)
```bash
python setup.py --all
```

### Step 5: Verify Installation
```bash
python -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)"
```

---

## 📊 Dataset Preparation

### Required Dataset Structure:
```
modified-dataset/
├── train/
│   ├── Battery/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Keyboard/
│   ├── Microwave/
│   ├── Mobile/
│   ├── Mouse/
│   ├── PCB/
│   ├── Player/
│   ├── Printer/
│   ├── Television/
│   └── Washing Machine/
├── val/
│   └── (same structure as train)
└── test/
    └── (same structure as train)
```

### Supported E-Waste Categories:
1. **Battery** - Various types of batteries
2. **Keyboard** - Computer keyboards
3. **Microwave** - Microwave ovens
4. **Mobile** - Mobile phones and smartphones
5. **Mouse** - Computer mice
6. **PCB** - Printed Circuit Boards
7. **Player** - Media players and audio devices
8. **Printer** - Printers and scanners
9. **Television** - TVs and monitors
10. **Washing Machine** - Washing machines and dryers

### Dataset Guidelines:
- **Image Format**: JPG, JPEG, PNG
- **Image Size**: Any size (will be resized to 224×224)
- **Quality**: Clear, well-lit images
- **Quantity**: Minimum 100 images per category (more is better)
- **Diversity**: Various angles, lighting conditions, backgrounds

---

## 🏋️ Training Your Model

### Method 1: Complete Training Script
```bash
# Basic training
python train_model.py

# With custom configuration
python train_model.py --config config/custom_config.yaml

# Skip preprocessing (if already done)
python train_model.py --skip-preprocessing
```

### Method 2: Python API
```python
from src.model_training import EWasteModelTrainer

# Initialize trainer
trainer = EWasteModelTrainer('config/config.yaml')

# Train the model
model, history = trainer.train_complete_model()

# Plot training history
trainer.plot_training_history()
```

### Training Process:
1. **Data Preprocessing**: Analyze and prepare dataset
2. **Phase 1 Training**: Train classification head (frozen base model)
3. **Phase 2 Training**: Fine-tune entire model
4. **Evaluation**: Comprehensive performance analysis
5. **Model Saving**: Save best model and checkpoints

### Expected Training Time:
- **CPU Only**: 2-4 hours
- **GPU (GTX 1060+)**: 30-60 minutes
- **GPU (RTX 3070+)**: 15-30 minutes

---

## 🌐 Using the Web Application

### Launch the App:
```bash
streamlit run app.py
```

### Features:

#### 🏠 Home Page
- System overview and features
- Supported e-waste categories
- Quick statistics

#### 🔮 Predict Page
- **Upload Image**: Drag and drop or browse files
- **Take Photo**: Use camera for real-time classification
- **Sample Images**: Try pre-loaded examples
- **Results**: Detailed predictions with confidence scores
- **Recycling Info**: Environmental impact and recycling tips

#### 📊 Dataset Info Page
- Dataset statistics and distribution
- Class balance visualization
- Data augmentation details

#### 📈 Model Performance Page
- Training history plots
- Performance metrics
- Class-wise accuracy analysis

#### ℹ️ About Page
- Technical details
- Environmental impact
- Technology stack information

### Web App Tips:
- **Best Image Quality**: Use clear, well-lit images
- **Multiple Angles**: Try different perspectives for better accuracy
- **Batch Processing**: Upload multiple images for comparison
- **Confidence Scores**: Pay attention to prediction confidence

---

## 💻 Command Line Interface

### Training Commands:
```bash
# Complete training pipeline
python train_model.py

# Evaluation only
python train_model.py --evaluate-only --model-path models/saved_models/model.h5

# Custom configuration
python train_model.py --config config/custom_config.yaml
```

### Data Preprocessing:
```bash
# Run preprocessing only
python src/data_preprocessing.py
```

### Model Evaluation:
```bash
# Evaluate existing model
python src/model_evaluation.py
```

### Prediction Examples:
```bash
# Single image prediction
python -c "
from src.predictor import EWastePredictor
predictor = EWastePredictor(model_path='models/saved_models/model.h5')
result = predictor.predict_single_image('path/to/image.jpg')
print(f'Predicted: {result[\"predicted_class\"]} ({result[\"confidence\"]*100:.2f}%)')
"
```

---

## 🐍 Python API Usage

### Data Preprocessing:
```python
from src.data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor('config/config.yaml')

# Analyze dataset
data_info = preprocessor.load_and_analyze_data()

# Check image quality
corrupted, dimensions = preprocessor.check_image_quality()

# Create visualizations
preprocessor.visualize_class_distribution()
preprocessor.visualize_sample_images()

# Create data generators
train_gen, val_gen, test_gen = preprocessor.create_data_generators()
```

### Model Training:
```python
from src.model_training import EWasteModelTrainer

# Initialize trainer
trainer = EWasteModelTrainer('config/config.yaml')

# Build model
model = trainer.build_model()

# Train complete pipeline
model, history = trainer.train_complete_model()

# Plot training history
trainer.plot_training_history()
```

### Model Evaluation:
```python
from src.model_evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator('config/config.yaml')

# Load model
evaluator.load_model('models/saved_models/model.h5')

# Comprehensive evaluation
results = evaluator.comprehensive_evaluation()

# Individual evaluation components
test_results = evaluator.evaluate_on_test_set(test_generator)
predictions, pred_classes, true_classes = evaluator.generate_predictions(test_generator)
cm = evaluator.plot_confusion_matrix(true_classes, pred_classes)
```

### Predictions:
```python
from src.predictor import EWastePredictor

# Initialize predictor
predictor = EWastePredictor(
    config_path='config/config.yaml',
    model_path='models/saved_models/model.h5'
)

# Single image prediction
result = predictor.predict_single_image('path/to/image.jpg')
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']*100:.2f}%")

# Batch prediction
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
batch_results = predictor.predict_batch(image_paths)

# Camera prediction
camera_results = predictor.predict_from_camera(num_predictions=5)

# Confidence analysis
analysis = predictor.analyze_prediction_confidence('path/to/image.jpg')
```

---

## 📓 Interactive Notebooks

### Main Demo Notebook:
```bash
jupyter notebook notebooks/E-Waste_Classification_Demo.ipynb
```

### Notebook Sections:
1. **Setup and Configuration**: Environment setup and imports
2. **Data Exploration**: Dataset analysis and visualization
3. **Model Training**: Interactive training demonstration
4. **Model Evaluation**: Performance analysis and metrics
5. **Prediction Examples**: Real-world usage examples
6. **Results Analysis**: Comprehensive results interpretation

### Notebook Tips:
- **Run cells sequentially** for best results
- **Modify parameters** to experiment with different settings
- **Save your work** regularly
- **Use GPU runtime** if available (Google Colab, etc.)

---

## ⚙️ Advanced Configuration

### Configuration File (`config/config.yaml`):

```yaml
model:
  name: "EfficientNetV2B0"
  input_shape: [224, 224, 3]
  num_classes: 10
  weights: "imagenet"

training:
  batch_size: 32
  epochs_phase1: 5
  epochs_phase2: 10
  learning_rate_phase1: 0.001
  learning_rate_phase2: 0.0001
  early_stopping_patience: 5
  reduce_lr_patience: 3
  reduce_lr_factor: 0.5

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  target_size: [224, 224]
  augmentation:
    rotation_range: 20
    width_shift_range: 0.2
    height_shift_range: 0.2
    shear_range: 0.2
    zoom_range: 0.2
    horizontal_flip: true
    fill_mode: "nearest"

paths:
  data_dir: "modified-dataset"
  processed_dir: "data/processed"
  augmented_dir: "data/augmented"
  models_dir: "models/saved_models"
  checkpoints_dir: "models/checkpoints"
  logs_dir: "models/logs"

classes:
  - "Battery"
  - "Keyboard"
  - "Microwave"
  - "Mobile"
  - "Mouse"
  - "PCB"
  - "Player"
  - "Printer"
  - "Television"
  - "Washing Machine"
```

### Customization Options:

#### Model Architecture:
- Change `model.name` to use different architectures
- Adjust `input_shape` for different image sizes
- Modify `num_classes` for different number of categories

#### Training Parameters:
- Adjust `batch_size` based on available memory
- Modify `epochs_phase1` and `epochs_phase2` for training duration
- Change learning rates for different convergence behavior

#### Data Augmentation:
- Enable/disable specific augmentation techniques
- Adjust augmentation intensity parameters
- Add custom augmentation methods

---

## 🔧 Troubleshooting

### Common Issues and Solutions:

#### 1. Installation Issues
**Problem**: Package installation fails
```bash
# Solution: Upgrade pip and try again
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Memory Issues
**Problem**: Out of memory during training
```yaml
# Solution: Reduce batch size in config.yaml
training:
  batch_size: 16  # or 8 for very limited memory
```

#### 3. Dataset Not Found
**Problem**: Dataset directory not found
```bash
# Solution: Check dataset structure
ls modified-dataset/
ls modified-dataset/train/
```

#### 4. Model Loading Issues
**Problem**: Cannot load saved model
```python
# Solution: Check model path and file existence
import os
model_path = "models/saved_models/your_model.h5"
print(f"Model exists: {os.path.exists(model_path)}")
```

#### 5. GPU Not Detected
**Problem**: TensorFlow not using GPU
```python
# Solution: Check GPU availability
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

#### 6. Streamlit Issues
**Problem**: Web app not starting
```bash
# Solution: Check Streamlit installation and port
pip install --upgrade streamlit
streamlit run app.py --server.port 8502
```

### Performance Optimization:

#### For Training:
- Use GPU if available
- Increase batch size (if memory allows)
- Use mixed precision training
- Enable XLA compilation

#### For Inference:
- Use TensorFlow Lite for mobile deployment
- Implement model quantization
- Use batch prediction for multiple images
- Cache model in memory for repeated use

---

## 🎯 Best Practices

### Dataset Best Practices:
1. **Quality over Quantity**: Better to have fewer high-quality images
2. **Balanced Classes**: Ensure similar number of images per category
3. **Diverse Data**: Include various angles, lighting, backgrounds
4. **Clean Labels**: Verify correct classification of training images
5. **Regular Updates**: Continuously improve dataset with new images

### Training Best Practices:
1. **Start Small**: Begin with a subset of data for quick experiments
2. **Monitor Training**: Watch for overfitting and adjust accordingly
3. **Save Checkpoints**: Regular model saving during training
4. **Experiment**: Try different hyperparameters and architectures
5. **Validate Results**: Always test on unseen data

### Deployment Best Practices:
1. **Model Versioning**: Keep track of different model versions
2. **Performance Monitoring**: Monitor accuracy in production
3. **Fallback Mechanisms**: Handle edge cases and low-confidence predictions
4. **User Feedback**: Collect feedback to improve the system
5. **Regular Updates**: Retrain with new data periodically

### Code Best Practices:
1. **Version Control**: Use Git for code management
2. **Documentation**: Keep code well-documented
3. **Testing**: Write tests for critical functions
4. **Modular Design**: Keep code modular and reusable
5. **Error Handling**: Implement robust error handling

---

## 📞 Support and Resources

### Getting Help:
- **Documentation**: This guide and README.md
- **Code Comments**: Detailed comments in source code
- **Sample Scripts**: Example usage in notebooks and scripts
- **Configuration**: Well-documented config files

### Learning Resources:
- **TensorFlow Documentation**: https://tensorflow.org/
- **Keras Guides**: https://keras.io/guides/
- **Computer Vision**: Online courses and tutorials
- **Transfer Learning**: Research papers and articles

### Community:
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Join community discussions
- **Contributions**: Contribute improvements and fixes

---

## 🎉 Conclusion

Congratulations! You now have a comprehensive understanding of the E-Waste Classification System. Whether you're a beginner looking to classify e-waste images or an advanced user wanting to customize the system, this guide provides all the information you need.

Remember:
- Start with the web application for quick results
- Use the interactive notebooks for learning
- Customize the configuration for your specific needs
- Follow best practices for optimal performance

**Happy classifying! 🌍♻️**

---

*For additional support or questions, please refer to the project documentation or contact the development team.*