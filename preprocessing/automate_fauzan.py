import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
import json

class HeartDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.feature_columns = [
            'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
            'Sex_encoded', 'ChestPainType_encoded', 'RestingECG_encoded',
            'ExerciseAngina_encoded', 'ST_Slope_encoded'
        ]
        
    def load(self, filepath):
        print(f"📥 Loading dataset from: {filepath}")
        return pd.read_csv(filepath)
    
    def inspect(self, df):
        print("📊 Dataset Preview")
        print(df.head())
        print("\n🔎 Missing Values:")
        print(df.isnull().sum())
        print("\n🎯 Target Distribution:")
        print(df['HeartDisease'].value_counts())
    
    def fill_missing(self, df):
        print("🧹 Handling Missing Data...")
        df_clean = df.copy()
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

        for col in numeric_cols:
            if df_clean[col].isnull().sum():
                median = df_clean[col].median()
                df_clean[col].fillna(median, inplace=True)
                print(f"Filled {col} with median: {median}")

        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        for col in categorical_cols:
            if df_clean[col].isnull().sum():
                mode = df_clean[col].mode()[0]
                df_clean[col].fillna(mode, inplace=True)
                print(f"Filled {col} with mode: {mode}")

        return df_clean

    def encode(self, df):
        print("🔡 Encoding Categorical Columns...")
        df_enc = df.copy()
        cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

        for col in cat_cols:
            le = LabelEncoder()
            df_enc[col + '_encoded'] = le.fit_transform(df_enc[col])
            self.label_encoders[col] = le
            print(f"{col} → {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        df_enc['target'] = self.target_encoder.fit_transform(df_enc['HeartDisease'])
        print(f"\nTarget encoding: {dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))}")
        return df_enc

    def prepare(self, df):
        print("📦 Extracting Features and Target...")
        X = df[self.feature_columns]
        y = df['target']
        return X, y

    def scale(self, X_train, X_test):
        print("📏 Scaling Numerical Features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)

    def process(self, file_path, test_size=0.2, random_state=42):
        print("🚀 Starting Preprocessing Pipeline...")
        df = self.load(file_path)
        self.inspect(df)
        df_clean = self.fill_missing(df)
        df_encoded = self.encode(df_clean)
        X, y = self.prepare(df_encoded)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        X_train_scaled, X_test_scaled = self.scale(X_train, X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_all(self, X_train, X_test, y_train, y_test, output_path):
        print(f"💾 Saving Preprocessed Data to {output_path}")
        os.makedirs(output_path, exist_ok=True)

        # Save data
        X_train.to_csv(f"{output_path}/X_train.csv", index=False)
        X_test.to_csv(f"{output_path}/X_test.csv", index=False)
        y_train.to_csv(f"{output_path}/y_train.csv", index=False, header=["target"])
        y_test.to_csv(f"{output_path}/y_test.csv", index=False, header=["target"])

        # Save objects
        with open(f"{output_path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(f"{output_path}/label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)
        with open(f"{output_path}/target_encoder.pkl", "wb") as f:
            pickle.dump(self.target_encoder, f)

        # Metadata
        with open(f"{output_path}/feature_info.json", "w") as f:
            json.dump({
                "features": self.feature_columns,
                "target_classes": list(self.target_encoder.classes_)
            }, f, indent=2)

        print("✅ Data & objects saved successfully!")

# Main function
if __name__ == "__main__":
    print("❤️ Heart Disease Dataset Preprocessing")
    print("=" * 50)

    processor = HeartDataPreprocessor()
    raw_path = "../heart_raw.csv"  # ← sesuaikan jika perlu
    output_dir = "./dataset_preprocessing"

    try:
        X_train, X_test, y_train, y_test = processor.process(raw_path)
        processor.save_all(X_train, X_test, y_train, y_test, output_dir)

        print("\n🎉 Done! Dataset siap untuk training.")
        print(f"   - Train: {len(X_train)} samples")
        print(f"   - Test : {len(X_test)} samples")
        print(f"   - Features: {len(X_train.columns)}")
        print(f"   - Classes : {list(processor.target_encoder.classes_)}")

    except Exception as e:
        print(f"❌ Error: {e}")
