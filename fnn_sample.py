# -*- coding: utf-8 -*-
# Modified for Task 3 – Deep Learning using extracted CSV files

import pandas as pd
import numpy as np

# ---------------------------
# CHOOSE SCENARIO HERE:
# SA, SB, or SC
# ---------------------------
SCENARIO = "SA"

# Scenario file mapping
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
print(f"Training File: {train_file}")
print(f"Testing File : {test_file}\n")

# ---------------------------
# Load train/test data
# ---------------------------
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Features and labels
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test  = test_df.iloc[:, :-1].values
y_test  = test_df.iloc[:, -1].values

# ---------------------------
# Normalize
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

classifier.add(Dense(units=64, kernel_initializer='uniform',
                     activation='relu', input_dim=X_train.shape[1]))
classifier.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = classifier.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    verbose=1
)

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
