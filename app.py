import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os
import pickle

layers = tf.keras.layers
models = tf.keras.models
ResNet50 = tf.keras.applications.ResNet50


# %%
# PAGE CONFIG
st.set_page_config(
    page_title="Skin Cancer Classification - HAM10000",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# %%
# CUSTOM CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# %%
# HEADER
st.markdown('<p class="main-header">🔬 Skin Cancer Classification with ResNet50</p>', unsafe_allow_html=True)
st.markdown("### HAM10000 Dataset Analysis & Model Performance")

# %%
# SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    
    "🎯 Model Training",
    "⚖️ Model Comparison", 
    "📊 Dataset Overview", # 👈 new
    "📈 Performance Metrics",
    "🔍 Predictions Explorer"
    
])


# %%
# LOAD DATA FUNCTIONS
@st.cache_data
def load_metadata(path):
    """Load metadata from CSV"""
    meta = pd.read_csv(os.path.join(path, "HAM10000_metadata.csv"))
    meta['label'] = meta['dx'].astype('category').cat.codes
    return meta

@st.cache_resource
def load_model():
    """Load trained model by rebuilding architecture"""
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import layers, models
    
    weights_path = 'model_weights.h5'
    
    if os.path.exists(weights_path):
        try:
            # Rebuild exact architecture (no BatchNorm)
            base_model = ResNet50(
                include_top=False, 
                weights='imagenet', 
                input_shape=(224, 224, 3)
            )
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(7, activation='softmax')
            ])
            
            # Load weights
            model.load_weights(weights_path)
            return model
        except Exception as e:
            st.error(f"Error loading weights: {e}")
            return None
    else:
        st.warning(f"Weights file not found: {weights_path}")
        return None


@st.cache_data
def load_history():
    """Load training history if available"""
    if os.path.exists('history.pkl'):
        with open('history.pkl', 'rb') as f:
            data = pickle.load(f)
            return data.get('history'), data.get('history_finetune')
    return None, None

# %%
# PAGE 1: DATASET OVERVIEW
if page == "📊 Dataset Overview":
    st.markdown('<p class="section-header">Dataset Exploration</p>', unsafe_allow_html=True)
    
    path = st.text_input("Enter dataset path:", value="", help="Enter the path to your HAM10000 dataset folder")
    
    if path and os.path.exists(path):
        meta = load_metadata(path)
        meta_clean = meta.dropna()
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(meta))
        with col2:
            st.metric("Number of Classes", meta['dx'].nunique())
        with col3:
            st.metric("Missing Values", meta.isna().sum().sum())
        with col4:
            st.metric("Unique Patients", meta['lesion_id'].nunique())
        
        st.markdown("---")
        
        # EDA Visualizations
        dx_counts = meta_clean["dx"].value_counts().reset_index()
        dx_counts.columns = ["diagnosis", "count"]
        
        loc_counts = meta_clean["localization"].value_counts().reset_index()
        loc_counts.columns = ["localization", "count"]
        
        # Create comprehensive dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Age Distribution", 
                "Diagnosis Counts", 
                "Sex Distribution", 
                "Lesion Localization",
                "Age by Diagnosis & Sex", ""
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "domain"}, {"type": "xy"}],
                [{"type": "xy", "colspan": 2}, None]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1) Age histogram
        fig.add_trace(
            go.Histogram(
                x=meta_clean["age"], 
                nbinsx=30,
                marker=dict(color="#1f77b4", line=dict(color='white', width=1)),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2) Diagnosis bar
        fig.add_trace(
            go.Bar(
                x=dx_counts["diagnosis"], 
                y=dx_counts["count"],
                marker=dict(color=px.colors.qualitative.Set2),
                text=dx_counts["count"],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3) Sex pie
        sex_counts = meta_clean["sex"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sex_counts.index, 
                values=sex_counts.values,
                hole=0.4,
                marker=dict(colors=['#ff7f0e', '#2ca02c']),
                showlegend=True
            ),
            row=2, col=1
        )
        
        # 4) Localization bar
        fig.add_trace(
            go.Bar(
                x=loc_counts["localization"], 
                y=loc_counts["count"],
                marker=dict(color=px.colors.qualitative.Pastel),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 5) Age vs Diagnosis box plot
        color_map = {"male": "#1f77b4", "female": "#ff7f0e"}
        for sex_val in meta_clean["sex"].dropna().unique():
            subset = meta_clean[meta_clean["sex"] == sex_val]
            fig.add_trace(
                go.Box(
                    x=subset["dx"], 
                    y=subset["age"], 
                    name=str(sex_val),
                    marker=dict(color=color_map.get(sex_val, "gray")),
                    showlegend=True
                ),
                row=3, col=1
            )
        
        # Layout
        fig.update_layout(
            title_text="HAM10000 Dataset Overview",
            height=1000,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Age", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        fig.update_xaxes(title_text="Diagnosis", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_xaxes(title_text="Localization", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        fig.update_xaxes(title_text="Diagnosis", row=3, col=1)
        fig.update_yaxes(title_text="Age", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.markdown("### Sample Data")
        st.dataframe(meta_clean.head(20), use_container_width=True)
        
        # Class distribution details
        st.markdown("### Class Distribution Details")
        class_dist = meta_clean['dx'].value_counts().reset_index()
        class_dist.columns = ['Diagnosis', 'Count']
        class_dist['Percentage'] = (class_dist['Count'] / class_dist['Count'].sum() * 100).round(2)
        st.dataframe(class_dist, use_container_width=True)
        
        # Diagnosis type descriptions
        st.markdown("### Diagnosis Type Descriptions")
        diagnosis_info = {
            'akiec': 'Actinic keratoses and intraepithelial carcinoma',
            'bcc': 'Basal cell carcinoma',
            'bkl': 'Benign keratosis-like lesions',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic nevi',
            'vasc': 'Vascular lesions'
        }
        
        diag_df = pd.DataFrame(list(diagnosis_info.items()), columns=['Code', 'Description'])
        st.dataframe(diag_df, use_container_width=True)
        
    else:
        st.warning("Please enter a valid dataset path to view the data exploration.")
        st.info("💡 Tip: The path should be the folder containing 'HAM10000_metadata.csv'")

# %%
# PAGE 5: MODEL COMPARISON
elif page == "⚖️ Model Comparison":
    st.markdown('<p class="section-header">Compare CV vs Multimodal Models</p>', unsafe_allow_html=True)

    # Load histories
    if os.path.exists("histories.pkl"):
        with open("histories.pkl", "rb") as f:
            data = pickle.load(f)
        history_cv = data.get("history_cv")
        history_mm = data.get("history_mm")
    else:
        st.error("❌ histories.pkl not found – please save training histories.")
        st.stop()

    # Load predictions
    if os.path.exists("y_pred_cv.npy") and os.path.exists("y_pred_mm.npy") and os.path.exists("y_test.npy"):
        y_pred_cv = np.load("y_pred_cv.npy")
        y_pred_mm = np.load("y_pred_mm.npy")
        y_test = np.load("y_test.npy")
    else:
        st.error("❌ Prediction files not found – please save y_pred_cv.npy, y_pred_mm.npy, and y_test.npy")
        st.stop()

    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    num_classes = len(class_names)

    # --------------------
    # 1. Training History Comparison
    # --------------------
    st.markdown("### 1. Training History Comparison")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("CV Accuracy", "CV Loss", "MM Accuracy", "MM Loss")
    )

    # CV Model
    fig.add_trace(go.Scatter(y=history_cv['accuracy'], name="CV Train Acc", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(y=history_cv['val_accuracy'], name="CV Val Acc", line=dict(color="#ff7f0e")), row=1, col=1)
    fig.add_trace(go.Scatter(y=history_cv['loss'], name="CV Train Loss", line=dict(color="#1f77b4")), row=1, col=2)
    fig.add_trace(go.Scatter(y=history_cv['val_loss'], name="CV Val Loss", line=dict(color="#ff7f0e")), row=1, col=2)

    # Multimodal
    fig.add_trace(go.Scatter(y=history_mm['accuracy'], name="MM Train Acc", line=dict(color="#2ca02c")), row=2, col=1)
    fig.add_trace(go.Scatter(y=history_mm['val_accuracy'], name="MM Val Acc", line=dict(color="#d62728")), row=2, col=1)
    fig.add_trace(go.Scatter(y=history_mm['loss'], name="MM Train Loss", line=dict(color="#2ca02c")), row=2, col=2)
    fig.add_trace(go.Scatter(y=history_mm['val_loss'], name="MM Val Loss", line=dict(color="#d62728")), row=2, col=2)

    fig.update_layout(height=800, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # --------------------
    # 2. Confusion Matrices
    # --------------------
    st.markdown("### 2. Confusion Matrices")
    cm_cv = confusion_matrix(y_test, y_pred_cv)
    cm_mm = confusion_matrix(y_test, y_pred_mm)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### CV Model")
        st.plotly_chart(go.Figure(data=go.Heatmap(
            z=cm_cv, x=class_names, y=class_names,
            colorscale="Blues", text=cm_cv, texttemplate="%{text}"
        )).update_layout(title="CV Model Confusion Matrix"), use_container_width=True)

    with col2:
        st.markdown("#### Multimodal Model")
        st.plotly_chart(go.Figure(data=go.Heatmap(
            z=cm_mm, x=class_names, y=class_names,
            colorscale="Greens", text=cm_mm, texttemplate="%{text}"
        )).update_layout(title="Multimodal Confusion Matrix"), use_container_width=True)

    # --------------------
    # 3. Per-Class F1 Comparison
    # --------------------
    st.markdown("### 3. Per-Class F1-Score Comparison")
    _, _, f1_cv, _ = precision_recall_fscore_support(y_test, y_pred_cv, labels=range(num_classes))
    _, _, f1_mm, _ = precision_recall_fscore_support(y_test, y_pred_mm, labels=range(num_classes))

    fig_f1 = go.Figure()
    fig_f1.add_trace(go.Bar(x=class_names, y=f1_cv, name="CV Model", marker_color="#1f77b4"))
    fig_f1.add_trace(go.Bar(x=class_names, y=f1_mm, name="Multimodal", marker_color="#2ca02c"))

    fig_f1.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1]),
        title="Per-Class F1-Score Comparison"
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    # --------------------
    # 4. Summary
    # --------------------
    st.markdown("### 4. Summary")
    acc_cv = accuracy_score(y_test, y_pred_cv)
    acc_mm = accuracy_score(y_test, y_pred_mm)
    improvement = (acc_mm - acc_cv) * 100

    st.write(f"**CV Model Accuracy:** {acc_cv:.4f} ({acc_cv*100:.2f}%)")
    st.write(f"**Multimodal Accuracy:** {acc_mm:.4f} ({acc_mm*100:.2f}%)")
    st.success(f"Improvement: {improvement:+.2f}%")
    if improvement > 0:
        st.success("The Multimodal model outperforms the CV model!")
    elif improvement < 0:
        st.error("The CV model outperforms the Multimodal model.")
    else:
        st.info("Both models perform equally.")


# PAGE 2: MODEL TRAINING
elif page == "🎯 Model Training":
    st.markdown('<p class="section-header">Training History (CV Model) </p>', unsafe_allow_html=True)
    
    # Load history
    history, history_finetune = load_history()
    
    if history is not None:
        # Create training plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Initial Training - Accuracy',
                'Initial Training - Loss',
                'Fine-tuning - Accuracy',
                'Fine-tuning - Loss'
            )
        )
        
        # Initial training accuracy
        epochs_initial = list(range(len(history['accuracy'])))
        fig.add_trace(
            go.Scatter(
                x=epochs_initial, 
                y=history['accuracy'],
                mode='lines',
                line=dict(color='#1f77b4', width=2),
                showlegend=True,
                name='Train Acc'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=epochs_initial, 
                y=history['val_accuracy'],
                mode='lines',
                line=dict(color='#ff7f0e', width=2),
                showlegend=True,
                name='Val Acc'
            ),
            row=1, col=1
        )
        
        # Initial training loss
        fig.add_trace(
            go.Scatter(
                x=epochs_initial, 
                y=history['loss'],
                mode='lines',
                line=dict(color='#1f77b4', width=2),
                showlegend=True,
                name='Train Loss'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=epochs_initial, 
                y=history['val_loss'],
                mode='lines',
                line=dict(color='#ff7f0e', width=2),
                showlegend=True,
                name='Val Loss'
            ),
            row=1, col=2
        )
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Initial Training Epochs", len(history['accuracy']))
        with col2:
            st.metric("Best Val Accuracy (Initial)", f"{max(history['val_accuracy']):.4f}")
        with col3:
            st.metric("Final Train Accuracy (Initial)", f"{history['accuracy'][-1]:.4f}")
        
        if history_finetune:
            # Fine-tuning accuracy
            epochs_finetune = list(range(len(history_finetune['accuracy'])))
            fig.add_trace(
                go.Scatter(
                    x=epochs_finetune, 
                    y=history_finetune['accuracy'],
                    mode='lines',
                    line=dict(color='#2ca02c', width=2),
                    showlegend=True,
                    name='Train Acc (FT)'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs_finetune, 
                    y=history_finetune['val_accuracy'],
                    mode='lines',
                    line=dict(color='#d62728', width=2),
                    showlegend=True,
                    name='Val Acc (FT)'
                ),
                row=2, col=1
            )
            
            # Fine-tuning loss
            fig.add_trace(
                go.Scatter(
                    x=epochs_finetune, 
                    y=history_finetune['loss'],
                    mode='lines',
                    line=dict(color='#2ca02c', width=2),
                    showlegend=True,
                    name='Train Loss (FT)'
                ),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs_finetune, 
                    y=history_finetune['val_loss'],
                    mode='lines',
                    line=dict(color='#d62728', width=2),
                    showlegend=True,
                    name='Val Loss (FT)'
                ),
                row=2, col=2
            )
            
            # Fine-tuning metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fine-tuning Epochs", len(history_finetune['accuracy']))
            with col2:
                st.metric("Best Val Accuracy (Fine-tuned)", f"{max(history_finetune['val_accuracy']):.4f}")
            with col3:
                st.metric("Final Train Accuracy (Fine-tuned)", f"{history_finetune['accuracy'][-1]:.4f}")
        
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=2, col=2)
        
        fig.update_layout(height=800, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Training insights
        st.markdown("### Training Insights")
        
        if history:
            initial_improvement = history['val_accuracy'][-1] - history['val_accuracy'][0]
            st.write(f"**Initial Training:** Validation accuracy improved by {initial_improvement:.4f} ({initial_improvement*100:.2f}%)")
        
        if history_finetune:
            finetune_improvement = history_finetune['val_accuracy'][-1] - history_finetune['val_accuracy'][0]
            st.write(f"**Fine-tuning:** Validation accuracy improved by {finetune_improvement:.4f} ({finetune_improvement*100:.2f}%)")
            
            total_improvement = history_finetune['val_accuracy'][-1] - history['val_accuracy'][0]
            st.write(f"**Overall:** Total improvement of {total_improvement:.4f} ({total_improvement*100:.2f}%)")
        
    else:
        st.info("Training history not available. Please save history during training.")
        st.markdown("### How to Save Training History")
        st.code("""
# After training, add this to your notebook:
import pickle

# Save training history
with open('history.pkl', 'wb') as f:
    pickle.dump({
        'history': history.history,
        'history_finetune': history_finetune.history
    }, f)

print("✓ Training history saved!")
        """, language='python')

# %%
# PAGE 3: PERFORMANCE METRICS
elif page == "📈 Performance Metrics":
    st.markdown('<p class="section-header">Model Performance Analysis</p>', unsafe_allow_html=True)
    
    # File uploaders for predictions
    col1, col2 = st.columns(2)
    with col1:
        uploaded_preds = st.file_uploader("Upload predictions file (numpy array)", type=['npy'], key='preds')
    with col2:
        uploaded_labels = st.file_uploader("Upload true labels file (numpy array)", type=['npy'], key='labels')
    
    if uploaded_preds and uploaded_labels:
        y_pred = np.load(uploaded_preds)
        y_test = np.load(uploaded_labels)
        
        # Get class names
        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        num_classes = len(class_names)
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        st.markdown("### Overall Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
        with col2:
            st.metric("Total Test Samples", len(y_test))
        with col3:
            st.metric("Number of Classes", num_classes)
        
        st.markdown("---")
        
        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Absolute Counts")
            cm = confusion_matrix(y_test, y_pred)
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 12},
                colorbar=dict(title="Count")
            ))
            
            fig_cm.update_layout(
                title="Confusion Matrix - Counts",
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                height=500
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.markdown("#### Normalized (Row %)")
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            fig_cm_norm = go.Figure(data=go.Heatmap(
                z=cm_normalized,
                x=class_names,
                y=class_names,
                colorscale='RdYlGn',
                text=np.round(cm_normalized * 100, 1),
                texttemplate='%{text}%',
                textfont={"size": 12},
                colorbar=dict(title="Percentage"),
                zmin=0,
                zmax=1
            ))
            
            fig_cm_norm.update_layout(
                title="Confusion Matrix - Normalized",
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                height=500
            )
            st.plotly_chart(fig_cm_norm, use_container_width=True)
        
        st.markdown("---")
        
        # Per-class metrics
        st.markdown("### Per-Class Performance")
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=range(num_classes)
        )
        
        # Bar chart
        fig_perf = go.Figure()
        
        fig_perf.add_trace(go.Bar(
            name='Precision',
            x=class_names,
            y=precision,
            text=np.round(precision, 3),
            textposition='outside',
            marker_color='#1f77b4'
        ))
        
        fig_perf.add_trace(go.Bar(
            name='Recall',
            x=class_names,
            y=recall,
            text=np.round(recall, 3),
            textposition='outside',
            marker_color='#ff7f0e'
        ))
        
        fig_perf.add_trace(go.Bar(
            name='F1-Score',
            x=class_names,
            y=f1,
            text=np.round(f1, 3),
            textposition='outside',
            marker_color='#2ca02c'
        ))
        
        fig_perf.update_layout(
            title="Per-Class Performance Metrics",
            xaxis_title="Diagnosis Type",
            yaxis_title="Score",
            barmode='group',
            height=500,
            yaxis=dict(range=[0, 1.1])
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Detailed table
        st.markdown("### Detailed Classification Report")
        
        report_df = pd.DataFrame({
            'Class': class_names,
            'Precision': np.round(precision, 4),
            'Recall': np.round(recall, 4),
            'F1-Score': np.round(f1, 4),
            'Support': support
        })
        
        # Add weighted averages
        weighted_avg = pd.DataFrame({
            'Class': ['Weighted Avg'],
            'Precision': [np.round(np.average(precision, weights=support), 4)],
            'Recall': [np.round(np.average(recall, weights=support), 4)],
            'F1-Score': [np.round(np.average(f1, weights=support), 4)],
            'Support': [support.sum()]
        })
        
        report_df = pd.concat([report_df, weighted_avg], ignore_index=True)
        
        # Style the dataframe
        st.dataframe(
            report_df.style.background_gradient(subset=['Precision', 'Recall', 'F1-Score'], cmap='RdYlGn', vmin=0, vmax=1),
            use_container_width=True
        )
        
        # Class distribution colored by F1-score
        st.markdown("### Test Set Class Distribution")
        
        test_class_counts = pd.Series(y_test).value_counts().sort_index()
        
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Bar(
            x=class_names,
            y=test_class_counts.values,
            marker=dict(
                color=f1,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="F1-Score"),
                cmin=0,
                cmax=1
            ),
            text=test_class_counts.values,
            textposition='outside'
        ))
        
        fig_dist.update_layout(
            title="Test Set Distribution (Colored by F1-Score)",
            xaxis_title="Class Label",
            yaxis_title="Number of Samples",
            height=500
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Model strengths and weaknesses
        st.markdown("### Model Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Best Performing Classes")
            best_classes = report_df.nlargest(3, 'F1-Score')[['Class', 'F1-Score']]
            for idx, row in best_classes.iterrows():
                if row['Class'] != 'Weighted Avg':
                    st.success(f"**{row['Class']}**: F1-Score = {row['F1-Score']:.4f}")
        
        with col2:
            st.markdown("#### ⚠️ Challenging Classes")
            worst_classes = report_df.nsmallest(3, 'F1-Score')[['Class', 'F1-Score']]
            for idx, row in worst_classes.iterrows():
                if row['Class'] != 'Weighted Avg':
                    st.warning(f"**{row['Class']}**: F1-Score = {row['F1-Score']:.4f}")
        
    else:
        st.info("Please upload prediction and label files to view performance metrics.")
        st.markdown("### How to Generate These Files")
        st.code("""
# After training, add this to your notebook:
import numpy as np

# Get predictions
y_pred_probs = HAM_ResNet.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Save for Streamlit
np.save('y_pred.npy', y_pred)
np.save('y_test.npy', y_test)

print("✓ Prediction files saved!")
        """, language='python')

# %%
# PAGE 4: PREDICTIONS EXPLORER
elif page == "🔍 Predictions Explorer":
    st.markdown('<p class="section-header">Interactive Predictions</p>', unsafe_allow_html=True)
    
    model = load_model()
    
    if model:
        st.success("✓ Model loaded successfully!")
        
        # Upload image for prediction
        uploaded_file = st.file_uploader("Upload a skin lesion image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                # Preprocess and predict
                img_array = np.array(image.resize((224, 224))) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                with st.spinner("Analyzing image..."):
                    predictions = model.predict(img_array, verbose=0)
                    pred_class = np.argmax(predictions[0])
                    confidence = predictions[0][pred_class] * 100
                
                class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
                class_full_names = {
                    'akiec': 'Actinic keratoses',
                    'bcc': 'Basal cell carcinoma',
                    'bkl': 'Benign keratosis',
                    'df': 'Dermatofibroma',
                    'mel': 'Melanoma',
                    'nv': 'Melanocytic nevi',
                    'vasc': 'Vascular lesions'
                }
                
                st.markdown("### Prediction Results")
                st.metric("Predicted Class", class_full_names[class_names[pred_class]])
                st.metric("Confidence", f"{confidence:.2f}%")
                
                # Risk assessment
                if class_names[pred_class] in ['mel', 'bcc']:
                    st.error("⚠️ **HIGH RISK**: This prediction indicates a potentially malignant lesion. Please consult a dermatologist immediately.")
                elif class_names[pred_class] == 'akiec':
                    st.warning("⚠️ **MODERATE RISK**: This lesion may require medical attention. Please consult a dermatologist.")
                else:
                    st.info("ℹ️ **LOW RISK**: This appears to be a benign lesion, but always consult a professional for proper diagnosis.")
            
            st.markdown("---")
            
            # Probability distribution
            st.markdown("### Confidence Distribution Across All Classes")
            
            prob_df = pd.DataFrame({
                'Class': [class_full_names[c] for c in class_names],
                'Probability': predictions[0] * 100
            }).sort_values('Probability', ascending=False)
            
            fig_prob = go.Figure(go.Bar(
                x=prob_df['Probability'],
                y=prob_df['Class'],
                orientation='h',
                marker=dict(
                    color=prob_df['Probability'],
                    colorscale='Blues',
                    showscale=False
                ),
                text=np.round(prob_df['Probability'], 2),
                texttemplate='%{text}%',
                textposition='outside'
            ))
            
            fig_prob.update_layout(
                title="Class Probabilities",
                xaxis_title="Probability (%)",
                yaxis_title="Class",
                height=400,
                xaxis=dict(range=[0, 105])
            )
            
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Top 3 predictions
            st.markdown("### Top 3 Predictions")
            top3 = prob_df.head(3)
            
            for idx, row in top3.iterrows():
                with st.expander(f"#{idx+1}: {row['Class']} ({row['Probability']:.2f}%)"):
                    st.write(f"**Confidence:** {row['Probability']:.2f}%")
                    
        else:
            st.info("👆 Please upload an image to get started")
            
            # Show example
            st.markdown("### Sample Images from Dataset")
            st.write("Here's what the model expects:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("**Format:** JPG, JPEG, PNG")
            with col2:
                st.info("**Recommended:** Close-up of lesion")
            with col3:
                st.info("**Quality:** Clear, well-lit")
    else:
        st.error("❌ Model not found!")
        st.markdown("### How to Save Your Model")
        st.code("""
# After training, add this to your notebook:
HAM_ResNet.save('HAM_ResNet_final.keras')
print("✓ Model saved!")
        """, language='python')
        st.info("Please train and save the model as 'HAM_ResNet_final.keras', then place it in the same directory as this Streamlit app.")

# %% 


# %%
# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p style='font-size: 14px;'><strong>HAM10000 Skin Cancer Classification</strong></p>
    <p style='font-size: 12px;'>Built with Streamlit, TensorFlow & ResNet50</p>
    <p style='font-size: 12px; color: #e74c3c;'>
        ⚠️ <strong>DISCLAIMER:</strong> This tool is for educational and research purposes only.<br>
        It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.<br>
        Always consult a qualified healthcare provider for medical concerns.
    </p>
</div>
""", unsafe_allow_html=True)