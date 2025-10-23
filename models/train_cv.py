import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --------------------------
# Load dataset
# --------------------------
data_path = "data/3_Region_Dataset.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip()

if "Region" not in data.columns:
    raise KeyError("Column 'Region' not found. Please check your CSV headers.")

print("📘 Loaded dataset with shape:", data.shape)
print("📊 Columns:", list(data.columns))

# --------------------------
# Prepare features and labels
# --------------------------
X = data[["Absorbance", "Concentration"]]
y = data["Region"].astype(str)

# --------------------------
# Encode labels
# --------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_names = encoder.classes_
print("🏷️ Classes:", list(class_names))

# --------------------------
# K-Fold Cross Validation Setup
# --------------------------
k = 10  # Adjust folds
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_accuracies = []
fold_confusions = []
fold = 1

# --------------------------
# Train and evaluate per fold
# --------------------------
for train_index, val_index in skf.split(X, y_encoded):
    print(f"\n🚀 Fold {fold}/{k} ---------------------------")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    fold_accuracies.append(acc)

    cm = confusion_matrix(y_val, y_pred)
    fold_confusions.append(cm)

    print(f"✅ Fold {fold} Accuracy: {acc * 100:.2f}%")
    print("📊 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))

    fold += 1

# --------------------------
# Summary metrics
# --------------------------
avg_acc = np.mean(fold_accuracies)
print("\n==============================")
print(f"📈 Average Cross-Validation Accuracy: {avg_acc * 100:.2f}%")
print(f"🏆 Best Fold Accuracy: {np.max(fold_accuracies) * 100:.2f}%")
print("==============================")

# --------------------------
# Save model and encoder
# --------------------------
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

best_model_index = np.argmax(fold_accuracies)
best_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
best_model.fit(X.iloc[skf.split(X, y_encoded).__next__()[0]], y_encoded[skf.split(X, y_encoded).__next__()[0]])

best_model_path = os.path.join(model_dir, "best_region_model.pkl")
encoder_path = os.path.join(model_dir, "region_encoder_cv.pkl")

joblib.dump(best_model, best_model_path)
joblib.dump(encoder, encoder_path)
print(f"\n💾 Best model saved to: {best_model_path}")
print(f"💾 Encoder saved to: {encoder_path}")

# --------------------------
# --------------------------
# VISUALIZATIONS (10 plots)
# --------------------------
import matplotlib
# Force matplotlib to use a default font
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
# Optional: increase font size for titles and labels
matplotlib.rcParams['font.size'] = 12
sns.set(style="whitegrid")

# 1️⃣ Dataset distribution by Region
plt.figure(figsize=(8,6))
sns.countplot(x=y)
plt.title("Dataset distribution by Region")
plt.show()

# 2️⃣ Scatter plot Absorbance vs Concentration
plt.figure(figsize=(8,6))
sns.scatterplot(x="Absorbance", y="Concentration", hue="Region", data=data, s=100)
plt.title("Absorbance vs Concentration by Region")
plt.show()

# 3️⃣ Boxplot Absorbance by Region
plt.figure(figsize=(8,6))
sns.boxplot(x="Region", y="Absorbance", data=data)
plt.title("Boxplot of Absorbance by Region")
plt.show()

# 4️⃣ Boxplot Concentration by Region
plt.figure(figsize=(8,6))
sns.boxplot(x="Region", y="Concentration", data=data)
plt.title("Boxplot of Concentration by Region")
plt.show()

# 5️⃣ Histogram of Absorbance
plt.figure(figsize=(8,6))
sns.histplot(data["Absorbance"], bins=10, kde=True, color='green')
plt.title("Histogram of Absorbance")
plt.show()

# 6️⃣ Histogram of Concentration
plt.figure(figsize=(8,6))
sns.histplot(data["Concentration"], bins=10, kde=True, color='orange')
plt.title("Histogram of Concentration")
plt.show()

# 7️⃣ Fold Accuracies
plt.figure(figsize=(8,6))
plt.plot(range(1, k+1), fold_accuracies, marker='o', color='blue')
plt.xticks(range(1, k+1))
plt.title("Fold-wise Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.show()

# 8️⃣ Confusion matrix of last fold
plt.figure(figsize=(8,6))
sns.heatmap(fold_confusions[-1], annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Confusion Matrix of Last Fold")
plt.show()

# 9️⃣ Pairplot of features
plt.figure(figsize=(8,6))
sns.scatterplot(x="Absorbance", y="Concentration", hue="Region", data=data, s=60, alpha=0.7)
plt.title("Pairplot (Absorbance vs Concentration)")
plt.show()

# 🔟 Feature importance from last trained RandomForest
plt.figure(figsize=(8,6))
importances = model.feature_importances_
sns.barplot(x=["Absorbance", "Concentration"], y=importances)
plt.title("Feature Importance (Last Fold)")
plt.show()