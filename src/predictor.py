import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import json

class EWastePredictor:
    def __init__(self, config_path='config/config.yaml', model_path=None):
        """Initialize the E-waste predictor."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model = None
        self.class_names = self.config['classes']
        self.target_size = tuple(self.config['data']['target_size'])
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained model."""
        print(f"📥 Loading model from: {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_input):
        """Preprocess image for prediction."""
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = image_input
            elif isinstance(image_input, np.ndarray):
                # NumPy array
                if image_input.shape[-1] == 3:  # RGB
                    image = Image.fromarray(image_input.astype('uint8'))
                else:
                    raise ValueError("Unsupported image array format")
            else:
                raise ValueError("Unsupported image input type")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to target size
            image = image.resize(self.target_size)
            
            # Convert to array and normalize
            img_array = np.array(image)
            img_array = img_array.astype('float32') / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, image
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None, None
    
    def predict_single_image(self, image_input, show_confidence=True, top_k=3):
        """Predict e-waste class for a single image."""
        if self.model is None:
            print("❌ No model loaded. Load a model first.")
            return None
        
        # Preprocess image
        img_array, original_image = self.preprocess_image(image_input)
        if img_array is None:
            return None
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        prediction_probs = predictions[0]
        
        # Get top-k predictions
        top_indices = np.argsort(prediction_probs)[::-1][:top_k]
        top_classes = [self.class_names[i] for i in top_indices]
        top_probs = [prediction_probs[i] for i in top_indices]
        
        # Prepare results
        result = {
            'predicted_class': top_classes[0],
            'confidence': float(top_probs[0]),
            'top_predictions': [
                {
                    'class': class_name,
                    'confidence': float(prob),
                    'percentage': f"{prob*100:.2f}%"
                }
                for class_name, prob in zip(top_classes, top_probs)
            ],
            'all_probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, prediction_probs)
            }
        }
        
        if show_confidence:
            self._display_prediction_results(original_image, result)
        
        return result
    
    def predict_batch(self, image_paths, batch_size=32):
        """Predict e-waste classes for multiple images."""
        if self.model is None:
            print("❌ No model loaded. Load a model first.")
            return None
        
        print(f"🔮 Predicting {len(image_paths)} images...")
        
        results = []
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_paths = []
            
            # Preprocess batch
            for path in batch_paths:
                img_array, _ = self.preprocess_image(path)
                if img_array is not None:
                    batch_images.append(img_array[0])  # Remove batch dimension
                    valid_paths.append(path)
            
            if not batch_images:
                continue
            
            # Convert to batch array
            batch_array = np.array(batch_images)
            
            # Make predictions
            batch_predictions = self.model.predict(batch_array, verbose=0)
            
            # Process results
            for j, (path, predictions) in enumerate(zip(valid_paths, batch_predictions)):
                predicted_class_idx = np.argmax(predictions)
                predicted_class = self.class_names[predicted_class_idx]
                confidence = float(predictions[predicted_class_idx])
                
                results.append({
                    'image_path': path,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'percentage': f"{confidence*100:.2f}%"
                })
        
        print(f"✅ Batch prediction completed for {len(results)} images")
        return results
    
    def _display_prediction_results(self, image, result):
        """Display prediction results with image."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display image
        ax1.imshow(image)
        ax1.set_title(f"Predicted: {result['predicted_class']}\n"
                     f"Confidence: {result['confidence']*100:.2f}%", 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Display top predictions bar chart
        top_preds = result['top_predictions']
        classes = [pred['class'] for pred in top_preds]
        confidences = [pred['confidence'] for pred in top_preds]
        
        bars = ax2.barh(classes, confidences, color=['green' if i == 0 else 'skyblue' for i in range(len(classes))])
        ax2.set_xlabel('Confidence Score')
        ax2.set_title('Top Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        
        # Add percentage labels
        for bar, conf in zip(bars, confidences):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{conf*100:.1f}%', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def predict_from_camera(self, camera_index=0, num_predictions=5):
        """Predict e-waste from camera feed."""
        if self.model is None:
            print("❌ No model loaded. Load a model first.")
            return None
        
        print("📷 Starting camera prediction...")
        print("Press 'c' to capture and predict, 'q' to quit")
        
        cap = cv2.VideoCapture(camera_index)
        predictions_made = 0
        results = []
        
        try:
            while predictions_made < num_predictions:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                # Display frame
                cv2.imshow('E-Waste Classifier - Press "c" to capture, "q" to quit', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Make prediction
                    result = self.predict_single_image(rgb_frame, show_confidence=False)
                    
                    if result:
                        predictions_made += 1
                        result['capture_time'] = datetime.now().isoformat()
                        results.append(result)
                        
                        print(f"\n🔮 Prediction {predictions_made}:")
                        print(f"   Class: {result['predicted_class']}")
                        print(f"   Confidence: {result['confidence']*100:.2f}%")
                        
                        # Save captured image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = f"captured_ewaste_{timestamp}.jpg"
                        cv2.imwrite(save_path, frame)
                        print(f"   Image saved: {save_path}")
                
                elif key == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"📷 Camera prediction completed. Made {len(results)} predictions.")
        return results
    
    def analyze_prediction_confidence(self, image_input):
        """Analyze prediction confidence and provide insights."""
        result = self.predict_single_image(image_input, show_confidence=False)
        
        if result is None:
            return None
        
        confidence = result['confidence']
        predicted_class = result['predicted_class']
        
        # Confidence analysis
        if confidence >= 0.9:
            confidence_level = "Very High"
            reliability = "Excellent"
            color = "green"
        elif confidence >= 0.7:
            confidence_level = "High"
            reliability = "Good"
            color = "lightgreen"
        elif confidence >= 0.5:
            confidence_level = "Moderate"
            reliability = "Fair"
            color = "orange"
        else:
            confidence_level = "Low"
            reliability = "Poor"
            color = "red"
        
        # Check for ambiguous predictions
        top_2_diff = result['top_predictions'][0]['confidence'] - result['top_predictions'][1]['confidence']
        is_ambiguous = top_2_diff < 0.2
        
        analysis = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'reliability': reliability,
            'is_ambiguous': is_ambiguous,
            'top_2_difference': top_2_diff,
            'recommendations': []
        }
        
        # Add recommendations
        if confidence < 0.7:
            analysis['recommendations'].append("Consider taking a clearer image")
            analysis['recommendations'].append("Ensure good lighting conditions")
            analysis['recommendations'].append("Make sure the e-waste item is clearly visible")
        
        if is_ambiguous:
            analysis['recommendations'].append(f"Prediction is ambiguous between {result['top_predictions'][0]['class']} and {result['top_predictions'][1]['class']}")
            analysis['recommendations'].append("Consider multiple angles or better image quality")
        
        # Display analysis
        print(f"\n🔍 Prediction Analysis:")
        print(f"   Predicted Class: {predicted_class}")
        print(f"   Confidence: {confidence*100:.2f}%")
        print(f"   Confidence Level: {confidence_level}")
        print(f"   Reliability: {reliability}")
        print(f"   Ambiguous: {'Yes' if is_ambiguous else 'No'}")
        
        if analysis['recommendations']:
            print(f"   Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   - {rec}")
        
        return analysis
    
    def save_prediction_results(self, results, filename=None):
        """Save prediction results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_results_{timestamp}.json"
        
        # Ensure logs directory exists
        os.makedirs(self.config['paths']['logs_dir'], exist_ok=True)
        filepath = os.path.join(self.config['paths']['logs_dir'], filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Prediction results saved to: {filepath}")
        return filepath

def find_latest_model(models_dir):
    """Find the latest trained model."""
    if not os.path.exists(models_dir):
        return None
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    if not model_files:
        return None
    
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    return os.path.join(models_dir, latest_model)

if __name__ == "__main__":
    # Initialize predictor
    predictor = EWastePredictor()
    
    # Find and load the latest model
    models_dir = predictor.config['paths']['models_dir']
    latest_model_path = find_latest_model(models_dir)
    
    if latest_model_path:
        print(f"🔍 Found latest model: {os.path.basename(latest_model_path)}")
        if predictor.load_model(latest_model_path):
            print("🎉 Predictor ready for use!")
            
            # Example usage
            print("\n📝 Example usage:")
            print("predictor.predict_single_image('path/to/image.jpg')")
            print("predictor.predict_from_camera()")
            print("predictor.analyze_prediction_confidence('path/to/image.jpg')")
        else:
            print("❌ Failed to load model")
    else:
        print("❌ No trained models found. Train a model first.")