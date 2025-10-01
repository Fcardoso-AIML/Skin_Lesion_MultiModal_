🔬 Skin Cancer Classification - HAM10000
This project implements automated skin lesion classification using the HAM10000 dataset.
It compares two deep learning approaches:

- **CV-Only Model**: ResNet50 trained solely on dermatoscopic images
- **Multimodal Model**: ResNet50 for images + tabular features (age, sex, localization)

Both models use transfer learning and are evaluated through a Streamlit dashboard with interactive visualizations.


## ⚙️ Installation

### 1. Clone the repository

git clone https://github.com/Fcardoso-AIML/Skin_Lesion_MultiModal_.git
cd Skin_Lesion_MultiModal_

2. Create environment
bashconda create -n skin-lesion python=3.11 -y
conda activate skin-lesion
3. Install requirements
bashpip install -r requirements.txt
requirements.txt:
tensorflow==2.15
streamlit
scikit-learn
plotly
pillow
matplotlib
seaborn
pandas
numpy
joblib
kagglehub

📊 Dataset
The HAM10000 dataset contains 10,015 dermatoscopic images across 7 classes:

nv: Melanocytic nevi (benign) - 67%
mel: Melanoma (malignant)
bkl: Benign keratosis
bcc: Basal cell carcinoma (malignant)
akiec: Actinic keratoses (pre-cancerous)
df: Dermatofibroma (benign)
vasc: Vascular lesions (benign)

The dataset is automatically downloaded via kagglehub:
pythonimport kagglehub
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
Data Split:

Training: 70% (7,010 samples)
Validation: 15% (1,503 samples)
Test: 15% (1,503 samples)

🏗️ Training
Run the Jupyter notebook to train both models:
bashjupyter notebook Skin_Lesion_Analysis_System.ipynb
Training process:

Loads and preprocesses HAM10000 dataset
Trains CV-only model (2 stages: frozen → fine-tuned)
Trains multimodal model (2 stages: frozen → fine-tuned)
Generates comparison visualizations
Saves models, predictions, and training histories

Training Strategy:

Stage 1: Frozen ResNet50 base, train classification head (lr=0.0001, up to 100 epochs)
Stage 2: Unfreeze entire network, fine-tune (lr=0.00001, 20 epochs)
Loss: Sparse categorical crossentropy with balanced class weights
Callbacks: EarlyStopping, ModelCheckpoint

🌐 Streamlit Dashboard
Launch the interactive dashboard:
bashstreamlit run app.py
Navigate to http://localhost:8501
Features
📊 Dataset Overview

Interactive exploratory data analysis with Plotly
Age distribution, diagnosis counts, sex breakdown
Lesion localization patterns
Class imbalance visualization

🎯 Model Training

Training/validation accuracy and loss curves
Comparison of frozen vs fine-tuned phases
Convergence analysis for both models

📈 Performance Metrics

Confusion matrices (absolute counts + normalized)
Per-class precision, recall, F1-score
Best/worst performing classes
Model strengths and weaknesses analysis

🔍 Predictions Explorer

Upload custom lesion images
Real-time classification with confidence scores
Risk assessment (high/moderate/low)
Probability distribution across all classes

⚖️ Model Comparison

Side-by-side accuracy comparison
Per-class F1-score comparison
Confusion matrix comparison
Analysis of multimodal performance

📈 Results
ModelTest AccuracyWeighted F1-ScoreCV-Only85.23%84.96%Multimodal80.31%81.22%Difference-4.92%-3.74%
Key Findings
CV-Only Model Strengths:

Excellent melanocytic nevi detection (92.53% F1)
Strong precision-recall balance
Robust generalization across classes

Multimodal Model Performance:

Underperformed despite additional clinical metadata
4.92% accuracy decrease compared to CV-only
Lower precision across all classes

Conclusion:
Clinical metadata (age, sex, localization) did not improve classification performance. Visual morphology alone captures sufficient diagnostic information for this task, suggesting that dermatoscopic features dominate demographic factors in lesion classification.

🔧 Technical Details

Model Architectures

CV-Only Model:

ResNet50 (ImageNet) → GlobalAvgPool → Dropout(0.5) → 

Dense(128, ReLU) → Dropout(0.3) → Dense(7, Softmax)

Multimodal Model:

Image Branch: ResNet50 → GlobalAvgPool → Dropout(0.5) → Dense(128)

Tabular Branch: Input(3) → Dense(64) → BatchNorm → Dense(32)

Fusion: Concat → Dense(64) → Dropout(0.3) → Dense(7, Softmax)


Preprocessing

Images resized to 224×224, normalized to [0,1]
Age: median imputation + StandardScaler
Sex/Location: label encoding + StandardScaler

⚠️ Disclaimer
This project is for educational and research purposes only.

NOT validated for clinical use
NOT a substitute for professional medical diagnosis
Always consult a qualified healthcare provider for medical concerns

Real-world medical AI deployment requires:

FDA/CE regulatory approval
Prospective clinical trials
Interpretability and uncertainty quantification
Bias assessment across diverse populations

👨‍💻 Author
Developed by Francisco Cardoso
MSc Mathematics & Economics | University of Copenhagen
