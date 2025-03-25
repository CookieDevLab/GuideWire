import pandas as pd
import xgboost as xgb
import pickle
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("kubernetes_issues.csv")

X = df.drop(columns=['Issue_Label'])
y = df['Issue_Label']

unique_labels = sorted(y.unique())
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
y = y.map(label_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(unique_labels), eval_metric='mlogloss')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

with open("kubernetes_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model saved successfully as kubernetes_model.pkl")

app = Flask(__name__)

with open("kubernetes_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

@app.route("/", methods=["GET"])
def home():
    return "Kubernetes Issue Prediction API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_features = pd.DataFrame([data])
    prediction = loaded_model.predict(input_features)[0]
    predicted_label = [key for key, value in label_mapping.items() if value == prediction][0]
    return jsonify({"prediction": predicted_label})

if __name__ == "__main__":
    app.run(debug=True)
