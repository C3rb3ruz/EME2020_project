import pandas as pd
import joblib
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model():
    print("Initializing Dropout Prediction Training Pipeline...\n")
    
    # 1. Load data with standard comma separator
    with tqdm(total=100, desc="Loading Dataset", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        df = pd.read_csv('students_dropout_academic_success.csv')
        pbar.update(100)

    # Clean column headers (strip tabs/spaces/quotes)
    df.columns = df.columns.str.strip().str.strip('"').str.strip("'")

    # 2. Preprocessing & Feature Filtering Steps
    with tqdm(total=100, desc="Preprocessing & Filtering Features", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        # Target mapping (handles 'target', 'Target', or 'Status')
        target_col = None
        for col in ['target', 'Target', 'Status']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            # Map target to binary (1 for Dropout, 0 for Non-Dropout)
            df[target_col] = df[target_col].apply(lambda x: 1 if str(x).strip().lower() == 'dropout' else 0)
        else:
            raise KeyError(f"Target column not found in dataset. Available columns: {list(df.columns)}")

        feature_cols = [
            'Marital status', 'Application mode', 'Application order', 'Course',
            'Daytime/evening attendance', 'Previous qualification',
            'Previous qualification (grade)', 'Nacionality', 'Admission grade',
            'Displaced', 'Educational special needs', 'Debtor',
            'Tuition fees up to date', 'Gender', 'Scholarship holder',
            'Age at enrollment', 'International',
            'Curricular units 1st sem (credited)',
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)',
            'Curricular units 1st sem (without evaluations)',
            'Curricular units 2nd sem (credited)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)',
            'Curricular units 2nd sem (without evaluations)',
            'Unemployment rate', 'Inflation rate', 'GDP'
        ]

        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].dropna()
        y = df.loc[X.index, target_col]
        pbar.update(100)

    # 3. Train-Test Split & Scaling
    with tqdm(total=100, desc="Scaling & Splitting Data", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pbar.update(100)

    # 4. Train Logistic Regression Model
    with tqdm(total=100, desc="Training Logistic Regression", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        pbar.update(100)

    # 5. Evaluate Model
    with tqdm(total=100, desc="Evaluating Model Metrics", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        y_pred = model.predict(X_test_scaled)
        pbar.update(100)

    print("\n" + "="*40)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("="*40)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save artifacts
    with tqdm(total=100, desc="Saving Model Artifacts", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        joblib.dump(model, 'model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(available_features, 'features.pkl')
        pbar.update(100)

    print("\n✨ Model, Scaler, and Features saved successfully!")

if __name__ == '__main__':
    train_and_save_model()