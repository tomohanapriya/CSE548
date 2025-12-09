# -*- coding: utf-8 -*-
# Fully Fixed Version – Runs SA, SB, SC and Handles Categorical Features

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

# ---------------------------
# Scenario File Definitions
# ---------------------------
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

summary_results = []

# Create output folder
os.makedirs("results", exist_ok=True)

# ---------------------------
# Function: Run One Scenario
# ---------------------------
def run_scenario(name, train_file, test_file):

    print(f"\n=== Running Scenario {name} ===")
    print(f"Training File: {train_file}")
    print(f"Testing File : {test_file}\n")

    # Load data
    train_df = pd.read_csv(train_file)
    test_df  = pd.read_csv(test_file)

    # ---------------------------
    # Encode Categorical Columns
    # ---------------------------
    print("[INFO] Checking for categorical columns ...")

    # Select all non-numeric columns
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()

    if len(cat_cols) > 0:
        print("[INFO] Converting categorical columns:", cat_cols)

        # Combine for consistent encoding
        full = pd.concat([train_df, test_df], axis=0)
        full = pd.get_dummies(full, columns=cat_cols)

        # Split back
        train_df = full.iloc[:len(train_df), :]
        test_df  = full.iloc[len(train_df):, :]

    else:
        print("[INFO] No categorical features found.")

    # Extract features + labels
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test  = test_df.iloc[:, :-1].values
    y_test  = test_df.iloc[:, -1].values

    # ---------------------------
    # Normalize
    # ---------------------------
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)

    # ---------------------------
    # Build Model
    # ---------------------------
    model = Sequential()
    model.add(Dense(64, activation='relu', kernel_initializer='uniform', input_dim=X_train.shape[1]))
    model.add(Dense(32, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("[INFO] Training model ...")
    history = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

    print("\n[INFO] Evaluating model ...")
    loss, acc = model.evaluate(X_test, y_test, verbose=1)

    # ---------------------------
    # Prediction + Confusion Matrix
    # ---------------------------
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    # Save matrix
    cm_file = f"results/confusion_{name}.txt"
    with open(cm_file, "w") as f:
        f.write(str(cm))

    # ---------------------------
    # Plot Accuracy and Loss
    # ---------------------------
    plt.plot(history.history['accuracy'])
    plt.title(f'Model Accuracy – {name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f"results/accuracy_{name}.png")
    plt.clf()

    plt.plot(history.history['loss'])
    plt.title(f'Model Loss – {name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"results/loss_{name}.png")
    plt.clf()

    # Add to summary table
    summary_results.append([name, loss, acc, cm.tolist()])

    print(f"[DONE] Scenario {name} complete. Results saved under /results/\n")


# ---------------------------
# Run All Scenarios
# ---------------------------
for scenario in ["SA", "SB", "SC"]:
    run_scenario(
        scenario,
        scenario_files[scenario]["train"],
        scenario_files[scenario]["test"],
    )

# ---------------------------
# Print Final Summary
# ---------------------------
print("\n=========== FINAL SUMMARY ===========")
for row in summary_results:
    name, loss, acc, cm = row
    print(f"\nScenario {name}:")
    print(f"  Loss     : {loss}")
    print(f"  Accuracy : {acc}")
    print(f"  Confusion Matrix : {cm}")

print("\nAll results stored in the 'results/' folder.\n")
