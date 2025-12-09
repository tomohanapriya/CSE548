# -*- coding: utf-8 -*-
# Updated FNN for NSL-KDD scenario files with proper encoding

import pandas as pd
import numpy as np

# ---------------------------
# CHOOSE SCENARIO: SA, SB, or SC
# ---------------------------
SCENARIO = "SA"

# Scenario file mapping
scenario_files = {
    "SA": {"train": "Training-a1-a3-a0.csv", "test": "Testing-a2-a4-a0.csv"},
    "SB": {"train": "Training-a1-a2-a0.csv", "test": "Testing-a1-a0.csv"},
    "SC": {"train": "Training-a1-a2-a0.csv", "test": "Testing-a1-a2-a3.csv"}
}

train_file = scenario_files[SCENARIO]["train"]
test_file  = scenario_files[SCENARIO]["test"]

print(f"\n=== Running Scenario {SCENARIO} ===")
print(f"Training File: {train_file}")
print(f"Testing File : {test_file}\n")

# ---------------------------
# Load train/test data
# ---------------------------
train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)

# Features and labels
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1].copy()
X_test  = test_df.iloc[:, :-1]
y_test  = test_df.iloc[:, -1].copy()

# ---------------------------
# Encode categorical features (protocol_type, service, flag)
# ---------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identify categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']
# If your CSV has different column names, adjust above

# Encode labels: normal=0, attack=1
y_train = np.where(y_train == 'normal', 0, 1)
y_test  = np.where(y_test == 'normal', 0, 1)

# Apply OneHotEncoder to categorical features
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), categorical_cols)],
    remainder='passthrough'
)
X_train = ct.fit_transform(X_train)
X_test  = ct.transform(X_test)

# ---------------------------
# Feature scaling
# ---------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# ---------------------------
# Build FNN
# ---------------------------
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = classifier.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

# ---------------------------
# Evaluate
# ---------------------------
loss, accuracy = classifier.evaluate(X_test, y_test)
print("\n=== MODEL PERFORMANCE ===")
print("Loss     :", loss)
print("Accuracy :", accuracy)

# Predict
y_pred_prob = classifier.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# ---------------------------
# Confusion Matrix
# ---------------------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ---------------------------
# Plot accuracy and loss
# ---------------------------
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('accuracy.png')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('loss.png')
plt.show()

