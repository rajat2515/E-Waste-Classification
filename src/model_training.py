import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from datetime import datetime
import json
from data_preprocessing import DataPreprocessor

class EWasteModelTrainer:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the model trainer with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model = None
        self.history = None
        self.preprocessor = DataPreprocessor(config_path)
        
        # Create directories if they don't exist
        os.makedirs(self.config['paths']['models_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['checkpoints_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
        
    def build_model(self):
        """Build the EfficientNetV2B0 model for e-waste classification."""
        print("🏗️  Building EfficientNetV2B0 model...")
        
        # Input layer
        inputs = keras.Input(shape=self.config['model']['input_shape'])
        
        # Base model (EfficientNetV2B0)
        base_model = EfficientNetV2B0(
            weights=self.config['model']['weights'],
            include_top=False,
            input_tensor=inputs
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.config['model']['num_classes'], activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs, outputs)
        
        print("✅ Model built successfully!")
        print(f"📊 Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def compile_model(self, learning_rate=None):
        """Compile the model with optimizer, loss, and metrics."""
        if learning_rate is None:
            learning_rate = self.config['training']['learning_rate_phase1']
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print(f"✅ Model compiled with learning rate: {learning_rate}")
    
    def get_callbacks(self, phase="phase1"):
        """Get training callbacks."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config['training']['reduce_lr_factor'],
                patience=self.config['training']['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config['paths']['checkpoints_dir'],
                    f'ewaste_model_{phase}_{timestamp}.h5'
                ),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_phase1(self, train_generator, val_generator):
        """Phase 1: Train with frozen base model."""
        print("🚀 Starting Phase 1 Training (Frozen Base Model)...")
        
        # Compile model for phase 1
        self.compile_model(self.config['training']['learning_rate_phase1'])
        
        # Get callbacks
        callbacks = self.get_callbacks("phase1")
        
        # Train the model
        history_phase1 = self.model.fit(
            train_generator,
            epochs=self.config['training']['epochs_phase1'],
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Phase 1 training completed!")
        return history_phase1
    
    def train_phase2(self, train_generator, val_generator):
        """Phase 2: Fine-tune with unfrozen base model."""
        print("🚀 Starting Phase 2 Training (Fine-tuning)...")
        
        # Unfreeze the base model
        base_model = self.model.layers[1]  # EfficientNetV2B0 layer
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) // 2
        
        # Freeze the earlier layers
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Compile with lower learning rate
        self.compile_model(self.config['training']['learning_rate_phase2'])
        
        # Get callbacks
        callbacks = self.get_callbacks("phase2")
        
        # Continue training
        history_phase2 = self.model.fit(
            train_generator,
            epochs=self.config['training']['epochs_phase2'],
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Phase 2 training completed!")
        return history_phase2
    
    def train_complete_model(self):
        """Complete training pipeline."""
        print("🎯 Starting complete model training pipeline...")
        
        # Build model
        self.build_model()
        
        # Get data generators
        train_gen, val_gen, test_gen = self.preprocessor.create_data_generators()
        
        # Phase 1 training
        history_phase1 = self.train_phase1(train_gen, val_gen)
        
        # Phase 2 training (fine-tuning)
        history_phase2 = self.train_phase2(train_gen, val_gen)
        
        # Combine histories
        self.history = {
            'phase1': history_phase1.history,
            'phase2': history_phase2.history
        }
        
        # Save the final model
        model_path = os.path.join(
            self.config['paths']['models_dir'],
            f'ewaste_classifier_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        )
        self.model.save(model_path)
        print(f"💾 Final model saved to: {model_path}")
        
        # Save training history
        history_path = os.path.join(
            self.config['paths']['logs_dir'],
            f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.model, self.history
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("❌ No training history available. Train the model first.")
            return
        
        print("📊 Creating training history plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combine phase histories
        all_loss = self.history['phase1']['loss'] + self.history['phase2']['loss']
        all_val_loss = self.history['phase1']['val_loss'] + self.history['phase2']['val_loss']
        all_accuracy = self.history['phase1']['accuracy'] + self.history['phase2']['accuracy']
        all_val_accuracy = self.history['phase1']['val_accuracy'] + self.history['phase2']['val_accuracy']
        
        epochs = range(1, len(all_loss) + 1)
        phase1_epochs = len(self.history['phase1']['loss'])
        
        # Plot training & validation loss
        axes[0, 0].plot(epochs, all_loss, 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, all_val_loss, 'r-', label='Validation Loss')
        axes[0, 0].axvline(x=phase1_epochs, color='g', linestyle='--', alpha=0.7, label='Phase 2 Start')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot training & validation accuracy
        axes[0, 1].plot(epochs, all_accuracy, 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, all_val_accuracy, 'r-', label='Validation Accuracy')
        axes[0, 1].axvline(x=phase1_epochs, color='g', linestyle='--', alpha=0.7, label='Phase 2 Start')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot learning rate (if available)
        if 'lr' in self.history['phase1']:
            all_lr = self.history['phase1']['lr'] + self.history['phase2']['lr']
            axes[1, 0].plot(epochs, all_lr, 'g-', label='Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot top-3 accuracy if available
        if 'top_3_accuracy' in self.history['phase1']:
            all_top3 = self.history['phase1']['top_3_accuracy'] + self.history['phase2']['top_3_accuracy']
            all_val_top3 = self.history['phase1']['val_top_3_accuracy'] + self.history['phase2']['val_top_3_accuracy']
            axes[1, 1].plot(epochs, all_top3, 'b-', label='Training Top-3 Accuracy')
            axes[1, 1].plot(epochs, all_val_top3, 'r-', label='Validation Top-3 Accuracy')
            axes[1, 1].axvline(x=phase1_epochs, color='g', linestyle='--', alpha=0.7, label='Phase 2 Start')
            axes[1, 1].set_title('Top-3 Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Top-3 Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            self.config['paths']['logs_dir'],
            f'training_history_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Training history plot saved to: {plot_path}")
        
        return fig
    
    def get_model_summary(self):
        """Get and display model summary."""
        if self.model is None:
            print("❌ No model available. Build the model first.")
            return
        
        print("📋 Model Summary:")
        self.model.summary()
        
        # Save model architecture plot
        try:
            plot_path = os.path.join(
                self.config['paths']['logs_dir'],
                'model_architecture.png'
            )
            keras.utils.plot_model(
                self.model,
                to_file=plot_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=False,
                dpi=150
            )
            print(f"🏗️  Model architecture diagram saved to: {plot_path}")
        except Exception as e:
            print(f"⚠️  Could not save model architecture diagram: {e}")

if __name__ == "__main__":
    # Initialize trainer
    trainer = EWasteModelTrainer()
    
    # Train the complete model
    model, history = trainer.train_complete_model()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Show model summary
    trainer.get_model_summary()
    
    print("🎉 Model training completed successfully!")