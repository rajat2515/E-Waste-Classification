import streamlit as st
import os
import sys
import yaml
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Add src directory to path
sys.path.append('src')

from predictor import EWastePredictor, find_latest_model
from data_preprocessing import DataPreprocessor

# Page configuration
st.set_page_config(
    page_title="E-Waste Classification System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #2E8B57;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #2E8B57;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF8C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #DC143C;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_config():
    """Load configuration file."""
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

@st.cache_resource
def initialize_predictor():
    """Initialize the predictor with the latest model."""
    config = load_config()
    models_dir = config['paths']['models_dir']
    latest_model_path = find_latest_model(models_dir)
    
    if latest_model_path:
        predictor = EWastePredictor(model_path=latest_model_path)
        return predictor, os.path.basename(latest_model_path)
    return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">♻️ E-Waste Classification System</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Electronic Waste Classification for Sustainable Recycling**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🔮 Predict", "📊 Dataset Info", "📈 Model Performance", "ℹ️ About"]
    )
    
    # Initialize predictor
    predictor, model_name = initialize_predictor()
    
    if predictor is None:
        st.error("❌ No trained model found. Please train a model first.")
        st.stop()
    
    # Display model info in sidebar
    st.sidebar.success(f"✅ Model Loaded: {model_name}")
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(predictor)
    elif page == "🔮 Predict":
        show_prediction_page(predictor)
    elif page == "📊 Dataset Info":
        show_dataset_info()
    elif page == "📈 Model Performance":
        show_model_performance()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(predictor):
    """Display the home page."""
    st.markdown('<h2 class="sub-header">Welcome to E-Waste Classification System</h2>', unsafe_allow_html=True)
    
    # Key features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accurate Classification</h3>
            <p>AI-powered classification of 10 different e-waste categories with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Fast Processing</h3>
            <p>Real-time image processing and classification in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🌱 Eco-Friendly</h3>
            <p>Supporting sustainable e-waste management and recycling</p>
        </div>
        """, unsafe_allow_html=True)
    
    # E-waste categories
    st.markdown('<h3 class="sub-header">📱 Supported E-Waste Categories</h3>', unsafe_allow_html=True)
    
    config = load_config()
    categories = config['classes']
    
    # Create a grid of categories
    cols = st.columns(5)
    category_icons = {
        'Battery': '🔋',
        'Keyboard': '⌨️',
        'Microwave': '📡',
        'Mobile': '📱',
        'Mouse': '🖱️',
        'PCB': '🔌',
        'Player': '📻',
        'Printer': '🖨️',
        'Television': '📺',
        'Washing Machine': '🧺'
    }
    
    for i, category in enumerate(categories):
        with cols[i % 5]:
            icon = category_icons.get(category, '📦')
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; margin: 0.5rem 0;">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-weight: bold; margin-top: 0.5rem;">{category}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown('<h3 class="sub-header">📊 System Statistics</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Categories", "10", "E-waste types")
    
    with col2:
        st.metric("Model Type", "EfficientNetV2B0", "Deep Learning")
    
    with col3:
        st.metric("Input Size", "224x224", "pixels")
    
    with col4:
        st.metric("Status", "Ready", "✅")

def show_prediction_page(predictor):
    """Display the prediction page."""
    st.markdown('<h2 class="sub-header">🔮 E-Waste Classification</h2>', unsafe_allow_html=True)
    
    # Upload method selection
    upload_method = st.radio(
        "Choose upload method:",
        ["📁 Upload Image File", "📷 Take Photo", "🖼️ Use Sample Images"]
    )
    
    uploaded_image = None
    
    if upload_method == "📁 Upload Image File":
        uploaded_file = st.file_uploader(
            "Choose an e-waste image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of electronic waste for classification"
        )
        
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
    
    elif upload_method == "📷 Take Photo":
        camera_image = st.camera_input("Take a photo of e-waste")
        
        if camera_image is not None:
            uploaded_image = Image.open(camera_image)
    
    elif upload_method == "🖼️ Use Sample Images":
        # Sample images (you would need to add sample images to a samples folder)
        st.info("Sample images feature - Add sample images to demonstrate the system")
        sample_options = ["Battery Sample", "Mobile Sample", "Keyboard Sample"]
        selected_sample = st.selectbox("Choose a sample:", sample_options)
        
        if st.button("Load Sample"):
            st.info("Sample image loading feature to be implemented")
    
    # Prediction
    if uploaded_image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if st.button("🔍 Classify E-Waste", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    result = predictor.predict_single_image(uploaded_image, show_confidence=False)
                    
                    if result:
                        # Display results
                        confidence = result['confidence']
                        predicted_class = result['predicted_class']
                        
                        # Confidence styling
                        if confidence >= 0.8:
                            conf_class = "confidence-high"
                            conf_emoji = "🟢"
                        elif confidence >= 0.6:
                            conf_class = "confidence-medium"
                            conf_emoji = "🟡"
                        else:
                            conf_class = "confidence-low"
                            conf_emoji = "🔴"
                        
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h3>🎯 Classification Result</h3>
                            <h2>{predicted_class}</h2>
                            <p class="{conf_class}">
                                {conf_emoji} Confidence: {confidence*100:.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Top predictions chart
                        st.markdown("### 📊 Top Predictions")
                        
                        top_preds = result['top_predictions'][:5]
                        pred_df = pd.DataFrame(top_preds)
                        
                        fig = px.bar(
                            pred_df,
                            x='confidence',
                            y='class',
                            orientation='h',
                            title="Prediction Confidence Scores",
                            color='confidence',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed probabilities
                        with st.expander("📋 Detailed Probabilities"):
                            all_probs = result['all_probabilities']
                            prob_df = pd.DataFrame(
                                list(all_probs.items()),
                                columns=['Class', 'Probability']
                            )
                            prob_df['Percentage'] = prob_df['Probability'] * 100
                            prob_df = prob_df.sort_values('Probability', ascending=False)
                            st.dataframe(prob_df, use_container_width=True)
                        
                        # Recycling information
                        show_recycling_info(predicted_class)
                    
                    else:
                        st.error("❌ Failed to classify the image. Please try again.")

def show_recycling_info(predicted_class):
    """Show recycling information for the predicted class."""
    recycling_info = {
        'Battery': {
            'icon': '🔋',
            'description': 'Batteries contain toxic materials and should be recycled at specialized facilities.',
            'recycling_tips': [
                'Never throw batteries in regular trash',
                'Take to battery recycling centers',
                'Many electronics stores accept old batteries',
                'Remove batteries from devices before disposal'
            ],
            'environmental_impact': 'Prevents heavy metals from contaminating soil and water'
        },
        'Mobile': {
            'icon': '📱',
            'description': 'Mobile phones contain valuable metals and should be properly recycled.',
            'recycling_tips': [
                'Wipe personal data before disposal',
                'Remove SIM card and memory cards',
                'Take to certified e-waste recyclers',
                'Consider donating if still functional'
            ],
            'environmental_impact': 'Recovers precious metals and reduces mining needs'
        },
        'Keyboard': {
            'icon': '⌨️',
            'description': 'Computer keyboards contain plastics and metals that can be recycled.',
            'recycling_tips': [
                'Remove batteries if wireless',
                'Take to electronics recycling centers',
                'Some parts can be reused for repairs',
                'Clean before recycling'
            ],
            'environmental_impact': 'Reduces plastic waste and recovers materials'
        },
        'Television': {
            'icon': '📺',
            'description': 'TVs contain hazardous materials and valuable components.',
            'recycling_tips': [
                'Never put in regular trash',
                'Take to certified e-waste facilities',
                'Check for manufacturer take-back programs',
                'Consider donation if working'
            ],
            'environmental_impact': 'Prevents lead and mercury contamination'
        },
        'Printer': {
            'icon': '🖨️',
            'description': 'Printers contain metals, plastics, and sometimes toner cartridges.',
            'recycling_tips': [
                'Remove ink/toner cartridges separately',
                'Take to electronics recycling centers',
                'Check manufacturer recycling programs',
                'Remove paper before recycling'
            ],
            'environmental_impact': 'Recovers metals and prevents plastic waste'
        }
    }
    
    # Default info for classes not specifically defined
    default_info = {
        'icon': '♻️',
        'description': 'This electronic waste should be properly recycled at certified facilities.',
        'recycling_tips': [
            'Do not throw in regular trash',
            'Take to certified e-waste recycling centers',
            'Check for manufacturer take-back programs',
            'Remove personal data if applicable'
        ],
        'environmental_impact': 'Proper recycling prevents environmental contamination'
    }
    
    info = recycling_info.get(predicted_class, default_info)
    
    st.markdown("### ♻️ Recycling Information")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; font-size: 4rem; margin: 2rem 0;">
            {info['icon']}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**{predicted_class} Recycling Guide**")
        st.write(info['description'])
        
        st.markdown("**Recycling Tips:**")
        for tip in info['recycling_tips']:
            st.write(f"• {tip}")
        
        st.info(f"🌍 **Environmental Impact:** {info['environmental_impact']}")

def show_dataset_info():
    """Display dataset information."""
    st.markdown('<h2 class="sub-header">📊 Dataset Information</h2>', unsafe_allow_html=True)
    
    config = load_config()
    
    # Dataset overview
    st.markdown("### 📋 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Classes", len(config['classes']))
    
    with col2:
        st.metric("Image Size", f"{config['data']['target_size'][0]}x{config['data']['target_size'][1]}")
    
    with col3:
        st.metric("Data Splits", "Train/Val/Test")
    
    # Class distribution (mock data - replace with actual data)
    st.markdown("### 📊 Class Distribution")
    
    # Mock data for demonstration
    class_counts = {
        'Battery': 850,
        'Keyboard': 920,
        'Microwave': 780,
        'Mobile': 1200,
        'Mouse': 890,
        'PCB': 750,
        'Player': 680,
        'Printer': 820,
        'Television': 950,
        'Washing Machine': 710
    }
    
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    
    fig = px.bar(
        df,
        x='Class',
        y='Count',
        title="Number of Images per E-Waste Category",
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data splits
    st.markdown("### 🔄 Data Splits")
    
    split_data = {
        'Split': ['Training', 'Validation', 'Testing'],
        'Percentage': [70, 15, 15],
        'Approximate Count': [5600, 1200, 1200]
    }
    
    split_df = pd.DataFrame(split_data)
    
    fig = px.pie(
        split_df,
        values='Percentage',
        names='Split',
        title="Dataset Split Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Data augmentation
    st.markdown("### 🔄 Data Augmentation")
    
    aug_config = config['data']['augmentation']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Augmentation Techniques:**")
        st.write(f"• Rotation: ±{aug_config['rotation_range']}°")
        st.write(f"• Width Shift: ±{aug_config['width_shift_range']*100}%")
        st.write(f"• Height Shift: ±{aug_config['height_shift_range']*100}%")
        st.write(f"• Shear: ±{aug_config['shear_range']*100}%")
    
    with col2:
        st.write(f"• Zoom: ±{aug_config['zoom_range']*100}%")
        st.write(f"• Horizontal Flip: {'Yes' if aug_config['horizontal_flip'] else 'No'}")
        st.write(f"• Fill Mode: {aug_config['fill_mode']}")

def show_model_performance():
    """Display model performance metrics."""
    st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', unsafe_allow_html=True)
    
    # Mock performance data (replace with actual data from evaluation)
    st.markdown("### 🎯 Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Accuracy", "87.5%", "2.3%")
    
    with col2:
        st.metric("Top-3 Accuracy", "95.2%", "1.8%")
    
    with col3:
        st.metric("Precision", "86.8%", "1.5%")
    
    with col4:
        st.metric("Recall", "87.1%", "2.1%")
    
    # Training history (mock data)
    st.markdown("### 📊 Training History")
    
    epochs = list(range(1, 16))
    train_acc = [0.45, 0.62, 0.71, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.91, 0.92, 0.92, 0.92]
    val_acc = [0.42, 0.58, 0.68, 0.74, 0.79, 0.82, 0.84, 0.85, 0.86, 0.87, 0.87, 0.88, 0.87, 0.88, 0.875]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Accuracy'))
    
    fig.update_layout(
        title="Model Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Class-wise performance (mock data)
    st.markdown("### 📊 Class-wise Performance")
    
    class_performance = {
        'Class': config['classes'],
        'Precision': [0.89, 0.85, 0.82, 0.91, 0.88, 0.84, 0.79, 0.87, 0.90, 0.83],
        'Recall': [0.87, 0.88, 0.85, 0.89, 0.86, 0.82, 0.81, 0.89, 0.88, 0.85],
        'F1-Score': [0.88, 0.86, 0.83, 0.90, 0.87, 0.83, 0.80, 0.88, 0.89, 0.84]
    }
    
    perf_df = pd.DataFrame(class_performance)
    
    fig = px.bar(
        perf_df,
        x='Class',
        y=['Precision', 'Recall', 'F1-Score'],
        title="Class-wise Performance Metrics",
        barmode='group'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    """Display about page."""
    st.markdown('<h2 class="sub-header">ℹ️ About E-Waste Classification System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This E-Waste Classification System is an AI-powered solution designed to automatically classify electronic waste 
    into different categories to support sustainable recycling and proper disposal practices.
    
    ### 🔬 Technical Details
    
    **Model Architecture:**
    - Base Model: EfficientNetV2B0
    - Transfer Learning with ImageNet weights
    - Custom classification head with dropout layers
    - Two-phase training (frozen + fine-tuning)
    
    **Training Strategy:**
    - Phase 1: Train classification head with frozen base model
    - Phase 2: Fine-tune entire model with lower learning rate
    - Data augmentation for improved generalization
    - Early stopping and learning rate scheduling
    
    **Performance:**
    - 10 e-waste categories classification
    - High accuracy on test dataset
    - Real-time inference capability
    - Confidence scoring for predictions
    
    ### 🌱 Environmental Impact
    
    Proper e-waste classification is crucial for:
    - **Resource Recovery**: Extracting valuable materials like gold, silver, and rare earth elements
    - **Pollution Prevention**: Preventing toxic substances from contaminating the environment
    - **Energy Conservation**: Reducing the need for mining new materials
    - **Circular Economy**: Supporting sustainable electronics lifecycle management
    
    ### 🛠️ Technology Stack
    
    - **Deep Learning**: TensorFlow/Keras
    - **Computer Vision**: OpenCV, PIL
    - **Web Interface**: Streamlit
    - **Data Processing**: NumPy, Pandas
    - **Visualization**: Matplotlib, Plotly
    - **Model**: EfficientNetV2B0
    
    ### 📊 Dataset
    
    The model is trained on a comprehensive dataset containing:
    - 10 different e-waste categories
    - Thousands of high-quality images
    - Balanced class distribution
    - Augmented training data
    
    ### 🎓 Educational Purpose
    
    This project serves as:
    - A demonstration of AI applications in environmental sustainability
    - An educational tool for understanding e-waste management
    - A practical example of computer vision in real-world problems
    - A contribution to sustainable technology practices
    
    ### 🤝 Contributing
    
    This project welcomes contributions in:
    - Model improvements and optimization
    - Additional e-waste categories
    - User interface enhancements
    - Documentation and tutorials
    
    ### 📞 Contact
    
    For questions, suggestions, or collaborations, please reach out to the development team.
    
    ---
    
    **Built with ❤️ for a sustainable future**
    """)

if __name__ == "__main__":
    main()