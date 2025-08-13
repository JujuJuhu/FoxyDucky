# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import io

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set Streamlit page configuration
st.set_page_config(
    page_title="JSON Anomaly Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("🚨 Anomaly Detection by FoxyDucky Task 3 Noctra Lupra")
st.markdown("""
Upload a **JSON file** (with each line being a JSON object) to perform anomaly detection using the **Isolation Forest** algorithm.
This app will preprocess your data, train a model, and provide a report with visualizations.
""")
st.info("The app expects a JSONL (JSON Lines) format, where each line is a valid JSON object.")

# --- Data Preprocessing Function ---
@st.cache_data
def analyze_json_data(file_content, contamination_rate):
    """
    Analyzes the JSON data for anomalies using a full preprocessing pipeline
    and Isolation Forest.
    """
    data = []
    try:
        # Decode file content from bytes to string and split by lines
        for line in file_content.decode('utf-8').splitlines():
            if line.strip():
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    except (json.JSONDecodeError, pd.errors.EmptyDataError) as e:
        return None, None, f"Error processing JSON file: {e}"

    if df.empty:
        return None, None, "Error: The provided JSON file is empty or invalid."
    
    st.write("---")
    st.subheader("Data Preprocessing Steps")

    # --- Step 1: Flatten Nested Dictionaries Iteratively ---
    st.info("Step 1: Flattening nested JSON objects...")
    # This loop now more robustly handles the mixed data types in log files
    for col in df.columns:
        if any(isinstance(x, dict) for x in df[col].dropna()):
            try:
                # Use a prefix to avoid column name conflicts
                flattened_df = pd.json_normalize(df[col][df[col].apply(lambda x: isinstance(x, dict))]).add_prefix(f"{col}_")
                df = df.drop(columns=[col]).join(flattened_df)
            except Exception as e:
                st.warning(f"Warning: Could not flatten column '{col}'. Error: {e}")
                
    st.success(f"DataFrame shape after flattening: **{df.shape}**")

    # --- Step 2: Handle Unhashable Types (Lists) ---
    st.info("Step 2: Handling unhashable list types...")
    for col in df.columns:
        if any(isinstance(x, list) for x in df[col].dropna()):
            # Convert lists to a string representation for one-hot encoding
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    st.success("Unhashable list types have been converted to strings.")

    # --- Step 3: Separate Features and Handle Missing Values ---
    st.info("Step 3: Separating features and handling missing values...")
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()

    exclude_cols = ['timestamp', 'src_ip', 'dest_ip', 'flow_id', 'in_iface', 'pkt_src', 'app_proto', 'proto']
    numerical_features = [col for col in numerical_features if col not in exclude_cols]
    categorical_features = [col for col in categorical_features if col not in exclude_cols]
    
    for col in numerical_features:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)
    st.success("Missing values in numerical columns handled by filling with the mean.")

    # --- Step 4: One-hot Encode Categorical Features ---
    st.info("Step 4: One-hot encoding categorical features...")
    if categorical_features:
        df_categorical_encoded = pd.get_dummies(df[categorical_features], dummy_na=True, drop_first=True, dtype=int)
    else:
        df_categorical_encoded = pd.DataFrame(index=df.index)
    st.success(f"Categorical features one-hot encoded. **{len(df_categorical_encoded.columns)}** new columns created.")

    # --- Step 5: Standardize Numerical Features ---
    st.info("Step 5: Standardizing numerical features...")
    scaler = StandardScaler()
    if numerical_features:
        df_numerical_scaled = pd.DataFrame(
            scaler.fit_transform(df[numerical_features]),
            columns=numerical_features,
            index=df.index
        )
    else:
        df_numerical_scaled = pd.DataFrame(index=df.index)
    st.success("Numerical features have been standardized.")

    # --- Step 6: Combine All Processed Features ---
    st.info("Step 6: Combining all processed features...")
    X = pd.concat([df_numerical_scaled, df_categorical_encoded], axis=1)

    if X.empty:
        return None, None, "Error: No features could be processed for analysis."
    st.success(f"Final preprocessed feature set has shape: **{X.shape}**")
    
    st.write("---")

    # --- Isolation Forest Model and Reporting ---
    st.subheader("Isolation Forest Model Analysis")
    model = IsolationForest(contamination=contamination_rate, random_state=42, n_jobs=-1)
    st.info(f"Training IsolationForest model with contamination={contamination_rate}...")
    model.fit(X)
    st.success("IsolationForest model trained successfully.")

    df['anomaly_prediction'] = model.predict(X)
    df['anomaly_score'] = model.decision_function(X)
    df['predicted_label'] = df['anomaly_prediction'].map({1: 0, -1: 1})

    return df, X, None

# --- Visualization Function ---
def generate_visualizations(df, X):
    """
    Generates and displays visualizations and a report based on the analysis results.
    """
    st.header("Analysis and Visualizations 📊")

    # 1. Anomaly Score Distribution
    st.subheader("📈 Distribution of Anomaly Scores")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['anomaly_score'], bins=50, kde=True, color='#1f77b4', ax=ax)
    ax.set_title('Distribution of Anomaly Scores', fontsize=16)
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.axvline(x=0, color='red', linestyle='--', label='Decision Boundary')
    ax.legend()
    ax.grid(axis='y', alpha=0.75)
    st.pyplot(fig)

    # 2. PCA Plot
    st.subheader("🗺️ Anomalies Visualized with PCA")
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        principal_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        principal_df['anomaly'] = df['predicted_label']

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            x=principal_df[principal_df['anomaly'] == 0]['PC1'],
            y=principal_df[principal_df['anomaly'] == 0]['PC2'],
            color='blue', label='Normal', alpha=0.6, ax=ax)
        sns.scatterplot(
            x=principal_df[principal_df['anomaly'] == 1]['PC1'],
            y=principal_df[principal_df['anomaly'] == 1]['PC2'],
            color='red', label='Anomaly', marker='X', s=100, ax=ax)
        ax.set_title('Anomalies Visualized with PCA', fontsize=16)
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("`Warning:` Too few features for PCA visualization. Skipping plot.")

    # 3. Anomaly Count by `event_type` (if exists)
    st.subheader("📊 Anomaly Count by Event Type")
    if 'event_type' in df.columns:
        plt.figure(figsize=(12, 6))
        palette = {-1: 'red', 1: 'blue'}
        fig, ax = plt.subplots()
        sns.countplot(x='event_type', hue='anomaly_prediction', data=df, palette=palette, ax=ax)
        ax.set_title('Anomaly Count by Event Type', fontsize=16)
        ax.set_xlabel('Event Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['Anomaly', 'Normal'], title='Prediction')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    else:
        st.warning("`Warning:` No 'event_type' column found in the original DataFrame. Skipping plot.")

# --- Streamlit UI Components ---
st.sidebar.header("User Input")
uploaded_file = st.sidebar.file_uploader("Choose a JSON file", type="json")
contamination_rate = st.sidebar.slider(
    "Set Anomaly Contamination Rate",
    min_value=0.01,
    max_value=0.5,
    value=0.05,
    step=0.01,
    help="An estimate of the proportion of outliers in the data. Adjusting this will change the number of anomalies detected."
)

if uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    
    if st.sidebar.button("Run Anomaly Detection"):
        with st.spinner("Processing data and training model... This may take a moment."):
            df, X, error_msg = analyze_json_data(file_content, contamination_rate)

            if error_msg:
                st.error(error_msg)
            else:
                st.success("Analysis complete!")
                
                # --- Anomaly Detection Results ---
                st.header("Anomaly Detection Results 🚨")
                num_anomalies = (df['predicted_label'] == 1).sum()
                num_normal = (df['predicted_label'] == 0).sum()
                anomaly_ratio = num_anomalies / len(df) if len(df) > 0 else 0

                st.markdown(f"""
                The Isolation Forest model's predictions revealed the following distribution:
                - **Total Data Points:** `{len(df)}`
                - **Detected Anomalies:** `{num_anomalies}`
                - **Normal Data Points:** `{num_normal}`
                - **Anomaly Ratio:** `{anomaly_ratio:.2%}`
                """)

                # --- Top 10 Anomalies ---
                st.subheader("Top 10 Anomalies by Score 📉")
                st.markdown("These are the data points with the lowest anomaly scores, indicating the highest likelihood of being an anomaly.")
                top_anomalies = df.sort_values(by='anomaly_score', ascending=True).head(10)
                
                # Check for key columns before displaying
                cols_to_display = [col for col in ['timestamp', 'anomaly_score', 'event_type', 'src_ip', 'dest_ip'] if col in top_anomalies.columns]
                st.dataframe(top_anomalies[cols_to_display])
                
                # --- Visualizations ---
                st.markdown("---")
                generate_visualizations(df, X)
else:
    st.write("Please upload a JSON file to begin the analysis.")