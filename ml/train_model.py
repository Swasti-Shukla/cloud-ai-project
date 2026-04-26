import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

np.random.seed(42)
print("Creating training data...")

users, storage, budget, traffic, labels = [], [], [], [], []

for i in range(3000):
    u = np.random.randint(1, 60000)
    s = np.random.randint(5, 5000)
    b = np.random.randint(5, 300)
    t = np.random.randint(1, 1600)

    if u <= 100 and t <= 10:
        correct = 0
    elif u <= 500 and t <= 50:
        correct = 1
    elif u <= 2000 and t <= 200:
        correct = 2
    elif u <= 5000 and t <= 500:
        correct = 3
    else:
        correct = 4

    if np.random.random() < 0.04:
        correct = np.random.randint(0, 5)

    users.append(u)
    storage.append(s)
    budget.append(b)
    traffic.append(t)
    labels.append(correct)

data = pd.DataFrame({
    'users': users,
    'storage': storage,
    'budget': budget,
    'traffic': traffic,
    'label': labels
})

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training ML model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy * 100:.1f}%")

os.makedirs('../backend/model', exist_ok=True)
joblib.dump(model, '../backend/model/cloud_model.pkl')
print("Model saved to: ../backend/model/cloud_model.pkl")
print("DONE! Now run: python app.py from backend folder")