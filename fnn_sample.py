# -*- coding: utf-8 -*-
# Updated fnn_sample.py for Task 3 — Deep Learning using extracted CSV files
# Handles SA, SB, SC scenarios with proper preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# ---------------------------
# Choose Scenario Here: "SA", "SB", or "SC"
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
# Load train/test data (no headers)
# ---------------------------
train_df = pd.read_csv(train_file, header=None)
test_df  = pd.read_csv(test_file, header=None)

# Features and labels
X_train = train_df.iloc[:, :-1]
y_train = np.where(train_df.iloc[:, -1] == 'normal', 0, 1)
X_test  = test_df.iloc[:, :-1]
y_test  = np.where(test_df.iloc[:, -1] == 'normal', 0, 1)

# ---------------------------
# Encode categorical features: columns 1,2,3
# ---------------------------
categorical_cols = [1, 2, 3]  # protocol_type, service, flag
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), categorical_cols)],
    remainder='passthrough'
)
X_train = ct.fit_transform(X_train)
X_test  = ct.transform(X_test)

# ---------------------------
# Feature scaling
# ---------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# ---------------------------
# Build FNN
# ---------------------------
classifier = Sequential()
classifier.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1], kernel_initializer='uniform'))
classifier.add(Dense(units=32, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = classifier.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    verbose=1
)

# ---------------------------
# Evaluate model
# ---------------------------
loss, accuracy = classifier.evaluate(X_test, y_test)
print("\n=== MODEL PERFORMANCE ===")
print(f"Loss     : {loss}")
print(f"Accuracy : {accuracy}")

# Predict
y_pred_prob = classifier.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Calculate testing accuracy from confusion matrix
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
testing_accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Testing Accuracy: {testing_accuracy:.4f}")

# ---------------------------
# Plot accuracy and loss
# ---------------------------
plt.figure()
plt.plot(history.history['accuracy'])
plt.title(f'Model Accuracy - Scenario {SCENARIO}')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig(f'accuracy_{SCENARIO}.png')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.title(f'Model Loss - Scenario {SCENARIO}')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig(f'loss_{SCENARIO}.png')
plt.show()
