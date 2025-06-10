import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

class RiskDetectionModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        X = data[['Angry', 'Fear', 'Happy', 'Sad']]
        y = data['label_encoded']
        return X, y

    def train(self, X, y):
        self.model.fit(X, y)
        print("Model trained on all training data.")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to: {path}")

    def predict(self, X):
        return self.model.predict(X)

    def save_predictions(self, X, predictions, output_path):
        results = X.copy()
        results['predicted_label'] = predictions
        # Add a separate column with string labels
        results['predicted_label_str'] = results['predicted_label'].map({0: 'not at risk', 1: 'at risk'})
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")

model = RiskDetectionModel()
X, y = model.load_data('files/aggregated_risk_data.csv')

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model.train(X_train, y_train)
model.save_model('files/risk_model.pkl')

# Predict on test split
preds = model.predict(X_test)
model.save_predictions(X_test, preds, 'files/predicted_risk_labels_test_split.csv')

# Classification Report
print("Classification Report:\n", classification_report(y_test, preds))
#Graphical representation of the model's performance
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
# Display confusion matrix
ConfusionMatrixDisplay.from_estimator(
    model.model, X_test, y_test, cmap=plt.cm.Blues
)
plt.title("Confusion Matrix")
plt.show()
