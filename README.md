# project-in-market-basket-analysis


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from ipywidgets import interact, widgets

df = pd.read_csv("/content/diversified_ecommerce_dataset(1).csv")
df = df.dropna()

np.random.seed(42)
df["Will Purchase Again"] = (((df["Return Rate"] < 35) & (df["Popularity Index"] > 50)) |
                             (np.random.rand(len(df)) > 0.90)).astype(int)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

df["Engagement Score"] = (100 - df["Return Rate"]) * df["Popularity Index"]

X = df.drop(columns=["Will Purchase Again", "Return Rate", "Popularity Index"])
y = df["Will Purchase Again"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


xgb.plot_importance(model, height=0.5)
plt.title("Feature Importance")
plt.show()


feature_names = X.columns

def predict_interactive(**kwargs):
    input_df = pd.DataFrame([kwargs])
    prediction = model.predict(input_df)[0]
    print("🔮 Prediction:", "✅ Will Purchase Again" if prediction == 1 else "❌ Will Not Purchase Again")

interact_args = {}
for feature in feature_names:
    f_min = int(X[feature].min())
    f_max = int(X[feature].max())
    if np.issubdtype(X[feature].dtype, np.integer):
        interact_args[feature] = widgets.IntSlider(min=f_min, max=f_max, value=(f_min + f_max) // 2)
    else:
        interact_args[feature] = widgets.FloatSlider(min=f_min, max=f_max, step=0.1, value=(f_min + f_max) / 2)

interact(predict_interactive, **interact_args)
