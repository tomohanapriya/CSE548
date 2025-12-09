# -*- coding: utf-8 -*-
# Fully fixed FNN code for Task 3 – handles categorical features + scenarios

import pandas as pd
import numpy as np

# ---------------------------
# SCENARIO SELECTION
# ---------------------------
SCENARIO = "SA"    # Change to SA / SB / SC

scenario_files = {
    "SA": {
        "train": "Training-a1-a3-a0.csv",
        "test":  "Testing-a2-a4-a0.csv"
    },
    "SB": {
        "train": "Training-a1-a2-a0.csv",
        "test":  "Testing-a1-a0.csv"
    },
    "SC": {
        "train": "Training-a1-a2-a0.csv",
        "test":  "Testing-a1-a2-a3.csv"
    }
}

train_file = scenario_files[SCENARIO]["train"]
test_file  = scenario_files[SCENARIO]["test"]

print(f"\n=== Running Scenario {SCENARIO} ===")
print(f"Training: {train_file}")
print(f"Testing : {test_file}\n")

# ---------------------------
# LOAD CSV FILES
# ---------------------------
train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)

# Separate features & labels
y_train = train_df.iloc[:, -1]
y_test  = test_df.iloc[:, -1]

X_train = train_df.iloc[:, :-1]
X_test  = test_df.iloc[:, :-1]

# ---------------------------
# FIX CATEGORICAL COLUMNS → One-hot encode
# ---------------------------
print("\n[INFO] Converting categorical features...\n")

X_train = pd.get_dummies(X_train)
X_test  = pd.get_dummies(X_test)

# Ensure train and test have identical columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Convert to numpy
X_train = X_train.values
X_test  = X_test.values

# ---------------------------
# ENCODE LABELS
# ---------------------------
y_train = y_train.apply(lambda x: 0 if x == 'normal' else 1).values
y_test  = y_test.apply(lambda x: 0 if x == 'normal' else 1).values

# ---------------------------
# NORMALIZATION
# ---------------------------
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# ---------------------------
# BUILD FNN MODEL
# ---------------------------
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classifier

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---------------------------
# TRAIN
# ---------------------------
print("\n[INFO] Training model...\n")

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    verbose=1
)

# ---------------------------
# EVALUATE
# ---------------------------
print("\n[INFO] Evaluating model...\n")
loss, accuracy = model.evaluate(X_test, y_test)
print("\n=== MODEL PERFORMANCE ===")
print("Loss     :", loss)
print("Accuracy :", accuracy)

# Predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# ---------------------------
# CONFUSION MATRIX
# ---------------------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ---------------------------
# PLOTS
# ---------------------------
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
