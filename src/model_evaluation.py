import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
from datetime import datetime
import json
from data_preprocessing import DataPreprocessor
from itertools import cycle

class ModelEvaluator:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the model evaluator with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model = None
        self.preprocessor = DataPreprocessor(config_path)
        self.class_names = self.config['classes']
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
    
    def load_model(self, model_path):
        """Load a trained model."""
        print(f"📥 Loading model from: {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
            return self.model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def evaluate_on_test_set(self, test_generator):
        """Evaluate model on test set."""
        if self.model is None:
            print("❌ No model loaded. Load a model first.")
            return None
        
        print("🔍 Evaluating model on test set...")
        
        # Get test loss and accuracy
        test_loss, test_accuracy, test_top3_accuracy = self.model.evaluate(
            test_generator, 
            verbose=1
        )
        
        print(f"📊 Test Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   Test Top-3 Accuracy: {test_top3_accuracy:.4f} ({test_top3_accuracy*100:.2f}%)")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top3_accuracy': test_top3_accuracy
        }
    
    def generate_predictions(self, test_generator):
        """Generate predictions for test set."""
        if self.model is None:
            print("❌ No model loaded. Load a model first.")
            return None, None
        
        print("🔮 Generating predictions...")
        
        # Reset generator
        test_generator.reset()
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = test_generator.classes
        
        return predictions, predicted_classes, true_classes
    
    def plot_confusion_matrix(self, true_classes, predicted_classes, normalize=True):
        """Plot confusion matrix."""
        print("📊 Creating confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        # Create plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save plot
        plot_path = os.path.join(
            self.config['paths']['logs_dir'],
            f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Confusion matrix saved to: {plot_path}")
        
        return cm
    
    def generate_classification_report(self, true_classes, predicted_classes):
        """Generate detailed classification report."""
        print("📋 Generating classification report...")
        
        # Generate report
        report = classification_report(
            true_classes,
            predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convert to DataFrame for better visualization
        report_df = pd.DataFrame(report).transpose()
        
        print("\n📊 Classification Report:")
        print("=" * 80)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("=" * 80)
        
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                      f"{metrics['f1-score']:<10.3f} {int(metrics['support']):<10}")
        
        print("=" * 80)
        print(f"{'Accuracy':<15} {'':<10} {'':<10} {report['accuracy']:<10.3f} {int(report['macro avg']['support']):<10}")
        print(f"{'Macro Avg':<15} {report['macro avg']['precision']:<10.3f} {report['macro avg']['recall']:<10.3f} "
              f"{report['macro avg']['f1-score']:<10.3f} {int(report['macro avg']['support']):<10}")
        print(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<10.3f} {report['weighted avg']['recall']:<10.3f} "
              f"{report['weighted avg']['f1-score']:<10.3f} {int(report['weighted avg']['support']):<10}")
        
        # Save report
        report_path = os.path.join(
            self.config['paths']['logs_dir'],
            f'classification_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Classification report saved to: {report_path}")
        
        return report, report_df
    
    def plot_class_wise_accuracy(self, true_classes, predicted_classes):
        """Plot class-wise accuracy."""
        print("📊 Creating class-wise accuracy plot...")
        
        # Calculate class-wise accuracy
        class_accuracies = []
        class_counts = []
        
        for i, class_name in enumerate(self.class_names):
            class_mask = (true_classes == i)
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
                class_accuracies.append(class_accuracy)
                class_counts.append(np.sum(class_mask))
            else:
                class_accuracies.append(0)
                class_counts.append(0)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Accuracy plot
        bars1 = ax1.bar(self.class_names, class_accuracies, color='skyblue', alpha=0.8)
        ax1.set_title('Class-wise Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, class_accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Sample count plot
        bars2 = ax2.bar(self.class_names, class_counts, color='lightcoral', alpha=0.8)
        ax2.set_title('Test Samples per Class', fontsize=14, fontweight='bold')
        ax2.set_xlabel('E-Waste Classes', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars2, class_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            self.config['paths']['logs_dir'],
            f'class_wise_accuracy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Class-wise accuracy plot saved to: {plot_path}")
        
        return class_accuracies, class_counts
    
    def plot_prediction_confidence(self, predictions, true_classes, predicted_classes):
        """Plot prediction confidence distribution."""
        print("📊 Creating prediction confidence plots...")
        
        # Get confidence scores (max probability for each prediction)
        confidence_scores = np.max(predictions, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = (predicted_classes == true_classes)
        correct_confidences = confidence_scores[correct_mask]
        incorrect_confidences = confidence_scores[~correct_mask]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall confidence distribution
        axes[0, 0].hist(confidence_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Overall Prediction Confidence Distribution')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(confidence_scores):.3f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Correct vs Incorrect predictions
        axes[0, 1].hist(correct_confidences, bins=20, alpha=0.7, color='green', 
                       label=f'Correct ({len(correct_confidences)})', edgecolor='black')
        axes[0, 1].hist(incorrect_confidences, bins=20, alpha=0.7, color='red', 
                       label=f'Incorrect ({len(incorrect_confidences)})', edgecolor='black')
        axes[0, 1].set_title('Confidence: Correct vs Incorrect Predictions')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot of confidence by correctness
        confidence_data = [correct_confidences, incorrect_confidences]
        axes[1, 0].boxplot(confidence_data, labels=['Correct', 'Incorrect'])
        axes[1, 0].set_title('Confidence Distribution by Prediction Correctness')
        axes[1, 0].set_ylabel('Confidence Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence vs Accuracy scatter
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_centers = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(correct_mask[bin_mask])
                bin_accuracies.append(bin_accuracy)
                bin_centers.append((confidence_bins[i] + confidence_bins[i + 1]) / 2)
        
        axes[1, 1].plot(bin_centers, bin_accuracies, 'bo-', linewidth=2, markersize=8)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Calibration')
        axes[1, 1].set_title('Confidence vs Accuracy (Reliability Diagram)')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            self.config['paths']['logs_dir'],
            f'prediction_confidence_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Prediction confidence plots saved to: {plot_path}")
        
        return confidence_scores
    
    def analyze_misclassifications(self, predictions, true_classes, predicted_classes, test_generator, top_n=5):
        """Analyze most common misclassifications."""
        print("🔍 Analyzing misclassifications...")
        
        # Find misclassified samples
        misclassified_mask = (predicted_classes != true_classes)
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            print("🎉 No misclassifications found!")
            return
        
        # Count misclassification patterns
        misclass_patterns = {}
        for idx in misclassified_indices:
            true_class = self.class_names[true_classes[idx]]
            pred_class = self.class_names[predicted_classes[idx]]
            pattern = f"{true_class} → {pred_class}"
            
            if pattern not in misclass_patterns:
                misclass_patterns[pattern] = []
            misclass_patterns[pattern].append(idx)
        
        # Sort by frequency
        sorted_patterns = sorted(misclass_patterns.items(), key=lambda x: len(x[1]), reverse=True)
        
        print(f"\n📊 Top {min(top_n, len(sorted_patterns))} Misclassification Patterns:")
        print("=" * 60)
        for i, (pattern, indices) in enumerate(sorted_patterns[:top_n]):
            print(f"{i+1}. {pattern}: {len(indices)} cases")
        
        # Create misclassification matrix
        misclass_matrix = np.zeros((len(self.class_names), len(self.class_names)))
        for idx in misclassified_indices:
            true_idx = true_classes[idx]
            pred_idx = predicted_classes[idx]
            misclass_matrix[true_idx, pred_idx] += 1
        
        # Plot misclassification matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            misclass_matrix,
            annot=True,
            fmt='g',
            cmap='Reds',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Number of Misclassifications'}
        )
        
        plt.title('Misclassification Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save plot
        plot_path = os.path.join(
            self.config['paths']['logs_dir'],
            f'misclassification_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Misclassification matrix saved to: {plot_path}")
        
        return sorted_patterns, misclass_matrix
    
    def comprehensive_evaluation(self, model_path=None, test_generator=None):
        """Run comprehensive model evaluation."""
        print("🚀 Starting comprehensive model evaluation...")
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        
        # Create test generator if not provided
        if test_generator is None:
            _, _, test_generator = self.preprocessor.create_data_generators()
        
        # Basic evaluation
        test_results = self.evaluate_on_test_set(test_generator)
        
        # Generate predictions
        predictions, predicted_classes, true_classes = self.generate_predictions(test_generator)
        
        # Create visualizations and reports
        cm = self.plot_confusion_matrix(true_classes, predicted_classes)
        report, report_df = self.generate_classification_report(true_classes, predicted_classes)
        class_acc, class_counts = self.plot_class_wise_accuracy(true_classes, predicted_classes)
        confidence_scores = self.plot_prediction_confidence(predictions, true_classes, predicted_classes)
        misclass_patterns, misclass_matrix = self.analyze_misclassifications(
            predictions, true_classes, predicted_classes, test_generator
        )
        
        # Compile comprehensive results
        evaluation_results = {
            'test_metrics': test_results,
            'classification_report': report,
            'class_wise_accuracy': dict(zip(self.class_names, class_acc)),
            'class_sample_counts': dict(zip(self.class_names, class_counts)),
            'confidence_stats': {
                'mean_confidence': float(np.mean(confidence_scores)),
                'std_confidence': float(np.std(confidence_scores)),
                'min_confidence': float(np.min(confidence_scores)),
                'max_confidence': float(np.max(confidence_scores))
            },
            'misclassification_patterns': [(pattern, len(indices)) for pattern, indices in misclass_patterns[:10]]
        }
        
        # Save comprehensive results
        results_path = os.path.join(
            self.config['paths']['logs_dir'],
            f'comprehensive_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"💾 Comprehensive evaluation results saved to: {results_path}")
        print("✅ Comprehensive evaluation completed!")
        
        return evaluation_results

if __name__ == "__main__":
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Find the latest model
    models_dir = evaluator.config['paths']['models_dir']
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
            model_path = os.path.join(models_dir, latest_model)
            
            print(f"🔍 Found latest model: {latest_model}")
            
            # Run comprehensive evaluation
            results = evaluator.comprehensive_evaluation(model_path)
            
            print("🎉 Model evaluation completed successfully!")
        else:
            print("❌ No trained models found. Train a model first.")
    else:
        print("❌ Models directory not found. Train a model first.")