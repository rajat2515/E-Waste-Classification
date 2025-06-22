import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the data preprocessor with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data_dir = self.config['paths']['data_dir']
        self.target_size = tuple(self.config['data']['target_size'])
        self.num_classes = self.config['model']['num_classes']
        self.classes = self.config['classes']
        
    def load_and_analyze_data(self):
        """Load and analyze the dataset structure."""
        print("🔍 Analyzing dataset structure...")
        
        data_info = []
        total_images = 0
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, 'train', class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                data_info.append({
                    'class': class_name,
                    'train_count': len(images),
                    'train_path': class_path
                })
                total_images += len(images)
                
                # Check validation set
                val_path = os.path.join(self.data_dir, 'val', class_name)
                if os.path.exists(val_path):
                    val_images = [f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    data_info[-1]['val_count'] = len(val_images)
                    total_images += len(val_images)
                
                # Check test set
                test_path = os.path.join(self.data_dir, 'test', class_name)
                if os.path.exists(test_path):
                    test_images = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    data_info[-1]['test_count'] = len(test_images)
                    total_images += len(test_images)
        
        self.data_info = pd.DataFrame(data_info)
        print(f"📊 Total images found: {total_images}")
        print("\n📋 Dataset Summary:")
        print(self.data_info)
        
        return self.data_info
    
    def check_image_quality(self, sample_size=50):
        """Check for corrupted images and analyze image dimensions."""
        print("🔍 Checking image quality...")
        
        corrupted_images = []
        dimensions = []
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, 'train', class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Sample images for quality check
                sample_images = np.random.choice(images, min(sample_size, len(images)), replace=False)
                
                for img_name in tqdm(sample_images, desc=f"Checking {class_name}"):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        with Image.open(img_path) as img:
                            dimensions.append(img.size)
                    except Exception as e:
                        corrupted_images.append(img_path)
                        print(f"❌ Corrupted image: {img_path} - {e}")
        
        if corrupted_images:
            print(f"⚠️  Found {len(corrupted_images)} corrupted images")
        else:
            print("✅ No corrupted images found")
        
        # Analyze dimensions
        if dimensions:
            widths, heights = zip(*dimensions)
            print(f"📏 Image dimensions - Width: {min(widths)}-{max(widths)}, Height: {min(heights)}-{max(heights)}")
        
        return corrupted_images, dimensions
    
    def create_data_generators(self):
        """Create data generators for training, validation, and testing."""
        print("🔄 Creating data generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=self.config['data']['augmentation']['rotation_range'],
            width_shift_range=self.config['data']['augmentation']['width_shift_range'],
            height_shift_range=self.config['data']['augmentation']['height_shift_range'],
            shear_range=self.config['data']['augmentation']['shear_range'],
            zoom_range=self.config['data']['augmentation']['zoom_range'],
            horizontal_flip=self.config['data']['augmentation']['horizontal_flip'],
            fill_mode=self.config['data']['augmentation']['fill_mode']
        )
        
        # Only rescaling for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.target_size,
            batch_size=self.config['training']['batch_size'],
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'val'),
            target_size=self.target_size,
            batch_size=self.config['training']['batch_size'],
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=self.target_size,
            batch_size=self.config['training']['batch_size'],
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {val_generator.samples}")
        print(f"✅ Test samples: {test_generator.samples}")
        
        return train_generator, val_generator, test_generator
    
    def visualize_class_distribution(self):
        """Create visualizations for class distribution."""
        print("📊 Creating class distribution visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training set distribution
        train_counts = self.data_info['train_count'].values
        axes[0, 0].bar(self.classes, train_counts, color='skyblue')
        axes[0, 0].set_title('Training Set Distribution')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Validation set distribution
        if 'val_count' in self.data_info.columns:
            val_counts = self.data_info['val_count'].values
            axes[0, 1].bar(self.classes, val_counts, color='lightgreen')
            axes[0, 1].set_title('Validation Set Distribution')
            axes[0, 1].set_ylabel('Number of Images')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Test set distribution
        if 'test_count' in self.data_info.columns:
            test_counts = self.data_info['test_count'].values
            axes[1, 0].bar(self.classes, test_counts, color='lightcoral')
            axes[1, 0].set_title('Test Set Distribution')
            axes[1, 0].set_ylabel('Number of Images')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall distribution
        total_counts = []
        for i, class_name in enumerate(self.classes):
            total = train_counts[i]
            if 'val_count' in self.data_info.columns:
                total += val_counts[i]
            if 'test_count' in self.data_info.columns:
                total += test_counts[i]
            total_counts.append(total)
        
        axes[1, 1].pie(total_counts, labels=self.classes, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Overall Class Distribution')
        
        plt.tight_layout()
        plt.savefig('models/logs/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def visualize_sample_images(self, samples_per_class=3):
        """Visualize sample images from each class."""
        print("🖼️  Creating sample image visualizations...")
        
        fig, axes = plt.subplots(samples_per_class, len(self.classes), figsize=(20, 3*samples_per_class))
        
        for i, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_dir, 'train', class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                sample_images = np.random.choice(images, min(samples_per_class, len(images)), replace=False)
                
                for j, img_name in enumerate(sample_images):
                    img_path = os.path.join(class_path, img_name)
                    img = Image.open(img_path)
                    img = img.resize(self.target_size)
                    
                    if samples_per_class == 1:
                        axes[i].imshow(img)
                        axes[i].set_title(class_name)
                        axes[i].axis('off')
                    else:
                        axes[j, i].imshow(img)
                        if j == 0:
                            axes[j, i].set_title(class_name, fontsize=12)
                        axes[j, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('models/logs/sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def preprocess_single_image(self, image_path):
        """Preprocess a single image for prediction."""
        try:
            img = Image.open(image_path)
            img = img.resize(self.target_size)
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except Exception as e:
            print(f"❌ Error preprocessing image {image_path}: {e}")
            return None

if __name__ == "__main__":
    # Test the data preprocessor
    preprocessor = DataPreprocessor()
    
    # Analyze data
    data_info = preprocessor.load_and_analyze_data()
    
    # Check image quality
    corrupted, dimensions = preprocessor.check_image_quality()
    
    # Create visualizations
    preprocessor.visualize_class_distribution()
    preprocessor.visualize_sample_images()
    
    # Create data generators
    train_gen, val_gen, test_gen = preprocessor.create_data_generators()
    
    print("✅ Data preprocessing completed successfully!") 