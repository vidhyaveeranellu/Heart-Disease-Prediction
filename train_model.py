import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Load dataset (correct file)
df = pd.read_csv(
    r"D:\HeartDisease\heart+disease\processed.cleveland.data",
    header=None,
    na_values="?"
)

# Assign column names
df.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

# Remove missing values
df = df.dropna()

# Convert target to binary (0 = no disease, 1 = disease)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Features & target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model trained and saved successfully!")