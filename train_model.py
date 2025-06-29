#!/usr/bin/env python3
"""
E-Waste Classification Model Training Script

This script provides a complete training pipeline for the e-waste classification model.
It includes data preprocessing, model training, evaluation, and result visualization.

Usage:
    python train_model.py [--config config/config.yaml] [--skip-preprocessing] [--evaluate-only]

Author: E-Waste Classification Team
Date: 2024
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime

# Add src directory to path
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from model_training import EWasteModelTrainer
from model_evaluation import ModelEvaluator

def setup_logging():
    """Set up logging configuration."""
    log_dir = "models/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def check_dataset_exists(data_dir):
    """Check if the dataset exists and has the required structure."""
    if not os.path.exists(data_dir):
        return False, f"Dataset directory '{data_dir}' not found"
    
    required_splits = ['train', 'val', 'test']
    missing_splits = []
    
    for split in required_splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            missing_splits.append(split)
    
    if missing_splits:
        return False, f"Missing dataset splits: {missing_splits}"
    
    return True, "Dataset structure is valid"

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='E-Waste Classification Model Training')
    parser.add_argument('--config', default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip data preprocessing step')
    parser.add_argument('--evaluate-only', action='store_true', help='Only run evaluation on existing model')
    parser.add_argument('--model-path', help='Path to model for evaluation-only mode')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 Starting E-Waste Classification Training Pipeline")
    logger.info(f"Configuration file: {args.config}")
    
    # Load configuration
    try:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("✅ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Check dataset
    data_dir = config['paths']['data_dir']
    dataset_valid, message = check_dataset_exists(data_dir)
    
    if not dataset_valid:
        logger.error(f"❌ Dataset validation failed: {message}")
        logger.info("Please ensure your dataset is organized as follows:")
        logger.info("modified-dataset/")
        logger.info("├── train/")
        logger.info("│   ├── Battery/")
        logger.info("│   ├── Keyboard/")
        logger.info("│   └── ... (other classes)")
        logger.info("├── val/")
        logger.info("│   └── ... (same structure)")
        logger.info("└── test/")
        logger.info("    └── ... (same structure)")
        return 1
    
    logger.info(f"✅ {message}")
    
    try:
        # Evaluation-only mode
        if args.evaluate_only:
            logger.info("🔍 Running evaluation-only mode")
            
            evaluator = ModelEvaluator(args.config)
            
            if args.model_path:
                model_path = args.model_path
            else:
                # Find latest model
                models_dir = config['paths']['models_dir']
                if not os.path.exists(models_dir):
                    logger.error("❌ No models directory found")
                    return 1
                
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
                if not model_files:
                    logger.error("❌ No trained models found")
                    return 1
                
                latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
                model_path = os.path.join(models_dir, latest_model)
                logger.info(f"📁 Using latest model: {latest_model}")
            
            # Run comprehensive evaluation
            results = evaluator.comprehensive_evaluation(model_path)
            logger.info("✅ Evaluation completed successfully")
            return 0
        
        # Data preprocessing
        if not args.skip_preprocessing:
            logger.info("📊 Starting data preprocessing...")
            
            preprocessor = DataPreprocessor(args.config)
            
            # Analyze dataset
            logger.info("🔍 Analyzing dataset structure...")
            data_info = preprocessor.load_and_analyze_data()
            
            # Check image quality
            logger.info("🔍 Checking image quality...")
            corrupted, dimensions = preprocessor.check_image_quality()
            
            if corrupted:
                logger.warning(f"⚠️  Found {len(corrupted)} corrupted images")
                for img_path in corrupted[:5]:  # Show first 5
                    logger.warning(f"   - {img_path}")
                if len(corrupted) > 5:
                    logger.warning(f"   ... and {len(corrupted) - 5} more")
            
            # Create visualizations
            logger.info("📊 Creating data visualizations...")
            preprocessor.visualize_class_distribution()
            preprocessor.visualize_sample_images()
            
            logger.info("✅ Data preprocessing completed")
        else:
            logger.info("⏭️  Skipping data preprocessing")
        
        # Model training
        logger.info("🏗️  Starting model training...")
        
        trainer = EWasteModelTrainer(args.config)
        
        # Train the complete model
        logger.info("🚀 Training model with two-phase approach...")
        model, history = trainer.train_complete_model()
        
        # Plot training history
        logger.info("📊 Creating training visualizations...")
        trainer.plot_training_history()
        
        # Show model summary
        trainer.get_model_summary()
        
        logger.info("✅ Model training completed successfully")
        
        # Model evaluation
        logger.info("🔍 Starting model evaluation...")
        
        evaluator = ModelEvaluator(args.config)
        evaluator.model = model  # Use the trained model
        
        # Get data generators for evaluation
        preprocessor = DataPreprocessor(args.config)
        _, _, test_generator = preprocessor.create_data_generators()
        
        # Run comprehensive evaluation
        results = evaluator.comprehensive_evaluation(test_generator=test_generator)
        
        logger.info("✅ Model evaluation completed successfully")
        
        # Summary
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info("📊 Final Results Summary:")
        logger.info(f"   Test Accuracy: {results['test_metrics']['test_accuracy']:.4f}")
        logger.info(f"   Test Top-3 Accuracy: {results['test_metrics']['test_top3_accuracy']:.4f}")
        logger.info(f"   Mean Confidence: {results['confidence_stats']['mean_confidence']:.4f}")
        
        # Best performing classes
        class_acc = results['class_wise_accuracy']
        best_class = max(class_acc.items(), key=lambda x: x[1])
        worst_class = min(class_acc.items(), key=lambda x: x[1])
        
        logger.info(f"   Best performing class: {best_class[0]} ({best_class[1]:.3f})")
        logger.info(f"   Needs improvement: {worst_class[0]} ({worst_class[1]:.3f})")
        
        logger.info("📁 Output files saved to:")
        logger.info(f"   Models: {config['paths']['models_dir']}")
        logger.info(f"   Logs: {config['paths']['logs_dir']}")
        logger.info(f"   Checkpoints: {config['paths']['checkpoints_dir']}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.exception("Full error traceback:")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)