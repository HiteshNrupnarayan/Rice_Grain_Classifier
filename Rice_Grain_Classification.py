# -------------------------------------------------------------------------
# Rice Grain Classification - Streamlit Web App (with Google Drive Download)
# -------------------------------------------------------------------------
# This script creates a web app that downloads its dataset from Google Drive,
# then trains, evaluates, and uses models to classify rice grains.
#
# To Run This App:
# 1. Make sure you have Python installed.
# 2. Install the required libraries:
#    pip install streamlit numpy opencv-python-headless Pillow scikit-learn gdown
# 3. Save this code as a Python file (e.g., `app.py`).
# 4. In Google Drive, zip your dataset folder (containing 'Train' and 'Test'
#    subfolders) and get a shareable link.
# 5. Get the FILE_ID from the link (e.g., in .../d/FILE_ID/view, copy the FILE_ID part).
# 6. Replace "YOUR_GOOGLE_DRIVE_FILE_ID" in the script below with your ID.
# 7. Open your terminal and run: streamlit run app.py
# -------------------------------------------------------------------------

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import joblib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import plotly.express as px
import gdown
import zipfile

# --- Page Configuration ---
st.set_page_config(
    page_title="Rice Grain Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Styling ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    h1, h2, h3 { color: #1e293b; }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        max-width: 300px !important;
    }
    .stButton>button {
        background-color: #334155;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    .stButton>button:hover { background-color: #1e293b; }
    .card {
        background-color: #F7F3E9;
        border: 1px solid #E7E0D4;
        border-radius: 0.75rem;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f1f5f9;
        border-left: 5px solid #64748b;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- NEW: Google Drive Download Configuration ---
# ⚠️ ACTION REQUIRED: Replace with your Google Drive file ID.
#    The link should be 'Anyone with the link can view'.
GOOGLE_DRIVE_FILE_ID = "1ujP3FVAiGbpp7WLlzY0vNkCJSRzqEyvD"
DATASET_ZIP_PATH = "dataset.zip"
DATA_DIR = "data"

# --- GLOBAL VARIABLES & PATHS ---
TRAIN_PATH = os.path.join(DATA_DIR, 'Train')
TEST_PATH = os.path.join(DATA_DIR, 'Test')
MODEL_SAVE_PATH = "best_rice_classifier.pkl"

# --- NEW: Function to Download and Unzip Data ---
def download_and_unzip_dataset(file_id, zip_path, dest_dir):
    """Downloads and unzips the dataset from Google Drive if not already present."""
    if os.path.exists(dest_dir):
        st.info("✅ Dataset already exists locally.")
        return

    st.info("📥 Dataset not found locally. Starting download from Google Drive...")
    try:
        # Download the file from Google Drive
        gdown.download(id=file_id, output=zip_path, quiet=False)
        st.success("Download complete!")

        # Unzip the file
        st.info("unzipping dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        st.success("Dataset successfully unzipped!")

        # Clean up the zip file
        os.remove(zip_path)

    except Exception as e:
        st.error(f"An error occurred during download/unzip: {e}")
        st.warning("Please ensure the Google Drive File ID is correct and the link is public.")

# --- CORE ML FUNCTIONS ---

@st.cache_data
def load_and_preprocess_images(directory, classes, class_to_label, image_size=(50, 50)):
    """Loads images, preprocesses them, and assigns labels."""
    X, Y = [], []
    total_files = sum([len(os.listdir(os.path.join(directory, cls))) for cls in classes if os.path.isdir(os.path.join(directory, cls))])
    progress_bar = st.progress(0, text="Loading images...")
    files_processed = 0

    for cls_name in classes:
        class_path = os.path.join(directory, cls_name)
        if not os.path.isdir(class_path):
            continue
        
        image_files = os.listdir(class_path)
        for img_name in image_files:
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path, 0)
                if img is None: continue
                img_resized = cv2.resize(img, image_size)
                X.append(img_resized.flatten())
                Y.append(class_to_label[cls_name])
            except Exception:
                continue
            files_processed += 1
            progress_bar.progress(files_processed / total_files, text=f"Loading {cls_name} images...")

    progress_bar.empty()
    return np.array(X), np.array(Y)

def train_and_evaluate_models(X_train, Y_train, X_test, Y_test, classes):
    """Trains multiple models and returns their performance metrics."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }
    
    results = {}
    best_model_info = {'name': None, 'model': None, 'accuracy': 0}

    for i, (name, model) in enumerate(models.items()):
        st.write(f"--- Training {name} ---")
        progress_bar = st.progress(0, text=f"Training {name}...")
        model.fit(X_train, Y_train)
        progress_bar.progress(100, text=f"{name} Trained!")
        
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        report_dict = classification_report(Y_test, Y_pred, target_names=classes, output_dict=True)
        cm = confusion_matrix(Y_test, Y_pred)
        
        results[name] = {'accuracy': accuracy, 'report': report_dict, 'cm': cm, 'model': model}

        if accuracy > best_model_info['accuracy']:
            best_model_info = {'name': name, 'model': model, 'accuracy': accuracy}
            
        progress_bar.empty()

    return results, best_model_info

@st.cache_resource
def load_model(path):
    """Loads a saved model from disk."""
    if os.path.exists(path):
        return joblib.load(path)
    return None

def get_prediction(image_bytes):
    """Preprocesses an image and gets a prediction from the saved model."""
    model = load_model(MODEL_SAVE_PATH)
    if model is None:
        st.error("Model not found. Please train a model on the 'Training Dashboard' page first.")
        return None, None

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        st.error("Could not process image.")
        return None, None
    
    img_resized = cv2.resize(img_cv, (50, 50))
    img_flattened = img_resized.flatten()
    img_normalized = (img_flattened / 255.0).reshape(1, -1)

    predicted_index = model.predict(img_normalized)[0]
    confidence = model.predict_proba(img_normalized).max()
    
    classes = sorted(os.listdir(TRAIN_PATH))
    predicted_class = classes[predicted_index]
    
    return predicted_class, confidence

# --- PAGE RENDERING FUNCTIONS ---

def home_page():
    st.title("🌾 AI-Powered Rice Grain Classification")
    st.subheader("Automating Quality and Purity Analysis with Machine Learning")
    
    st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:black;" /> """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Project Overview")
    st.write("""
        This application uses a machine learning model to classify rice grains. The dataset includes 
        **Arborio, Basmati, Ipsala, Jasmine and Karacadag**, five different varieties of rice often 
        grown in Turkey. The model is trained on a total of 5,000 images.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Get Started")
    st.write("Navigate to the **Training Dashboard** to download the dataset and train the models, then use the **Real-Time Classifier** to test your own images.")
    st.markdown('</div>', unsafe_allow_html=True)

def training_dashboard_page():
    st.title("⚙️ Training Dashboard")
    
    # MODIFIED: Trigger dataset download and unzip at the top of the page
    download_and_unzip_dataset(GOOGLE_DRIVE_FILE_ID, DATASET_ZIP_PATH, DATA_DIR)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Train and Evaluate Models")
    st.write("Click the button below to load the dataset, train classification models, and evaluate their performance. The best model will be saved automatically.")

    if st.button("Start Training Process"):
        with st.status("Step 1: Loading Data...", expanded=True) as status:
            if not os.path.isdir(TRAIN_PATH) or not os.path.isdir(TEST_PATH):
                st.error(f"Dataset not found! Please ensure the Google Drive download was successful and the '{DATA_DIR}' folder exists.")
                status.update(label="Data Loading Failed!", state="error", expanded=False)
                return

            classes = sorted(os.listdir(TRAIN_PATH))
            class_to_label = {cls_name: i for i, cls_name in enumerate(classes)}
            
            st.write("Loading Training Data...")
            X_train_raw, Y_train = load_and_preprocess_images(TRAIN_PATH, classes, class_to_label)
            st.write("Loading Testing Data...")
            X_test_raw, Y_test = load_and_preprocess_images(TEST_PATH, classes, class_to_label)
            
            X_train = X_train_raw / 255.0
            X_test = X_test_raw / 255.0
            
            st.session_state['data_loaded'] = True
            st.session_state['class_info'] = (classes, class_to_label)
            st.session_state['dataset'] = (X_train, Y_train, X_test, Y_test)
            
            st.success(f"Loaded {len(X_train)} training and {len(X_test)} testing images.")
            status.update(label="Data Loaded Successfully!", state="complete", expanded=False)

        with st.status("Step 2: Training Models...", expanded=True) as status:
            X_train, Y_train, X_test, Y_test = st.session_state['dataset']
            classes, _ = st.session_state['class_info']
            results, best_model_info = train_and_evaluate_models(X_train, Y_train, X_test, Y_test, classes)
            st.session_state['training_results'] = results
            status.update(label="Model Training Complete!", state="complete", expanded=False)

        with st.status("Step 3: Saving Best Model...", expanded=True) as status:
            joblib.dump(best_model_info['model'], MODEL_SAVE_PATH)
            st.success(f"Best model ({best_model_info['name']}) saved to `{MODEL_SAVE_PATH}` with an accuracy of {best_model_info['accuracy']:.2%}.")
            status.update(label="Model Saved!", state="complete", expanded=False)

    st.markdown('</div>', unsafe_allow_html=True)
    
    if 'training_results' in st.session_state:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("Training Results")
        results = st.session_state['training_results']
        classes, _ = st.session_state['class_info']
        
        for name, data in results.items():
            st.subheader(f"Results for: {name}")
            st.metric("Accuracy", f"{data['accuracy']:.2%}")
            
            with st.expander("View Detailed Classification Report"):
                report_df = pd.DataFrame(data['report']).transpose()
                st.dataframe(report_df)
                
            with st.expander("View Confusion Matrix"):
                fig = px.imshow(data['cm'], text_auto=True, labels=dict(x="Predicted", y="Actual"),
                                x=classes, y=classes, title=f"Confusion Matrix for {name}")
                st.plotly_chart(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)

def classifier_page():
    st.title("📷 Real-Time Classifier")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Upload an Image to Classify")
    
    if not os.path.exists(MODEL_SAVE_PATH):
        st.warning("No trained model found. Please go to the 'Training Dashboard' to train and save a model first.")
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(Image.open(uploaded_file), caption='Uploaded Image', width=300)
            
            if st.button("Classify Grain"):
                with st.spinner('Analyzing image with the trained model...'):
                    image_bytes = uploaded_file.getvalue()
                    predicted_class, confidence = get_prediction(image_bytes)
                    
                    if predicted_class:
                        st.success("Classification Complete!")
                        st.metric("Predicted Rice Variety", value=predicted_class)
                        st.write("Confidence:")
                        st.progress(confidence)
                        st.write(f"**{confidence:.2%}** confident")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main App Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Training Dashboard", "Real-Time Classifier"])
st.sidebar.markdown("---")
st.sidebar.info("An end-to-end ML app for rice classification. Train your models on the dashboard, then use the classifier page for predictions.")

# Display the selected page
if page == "Home":
    home_page()
elif page == "Training Dashboard":
    training_dashboard_page()
elif page == "Real-Time Classifier":
    classifier_page()
