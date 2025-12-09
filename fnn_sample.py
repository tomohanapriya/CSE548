# -*- coding: utf-8 -*-
# Run all 3 scenarios (SA, SB, SC) and save results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense

# ---------------------------------------------------------
# Scenario file mapping
# ---------------------------------------------------------
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

# To store results for summary table
summary_rows = []

# ---------------------------------------------------------
# Run all scenarios
# ---------------------------------------------------------
for SCENARIO in ["SA", "SB", "SC"]:

    print("\n====================================")
    print(f"=== Running Scenario {SCENARIO} ===")
    print("====================================")

    train_file = scenario_files[SCENARIO]["train"]
    test_file  = scenario_files[SCENARIO]["test"]

    print(f"Training File: {train_file}")
    print(f"Testing File : {test_file}\n")

    # ---------------------------
    # Load Data
    # ---------------------------
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

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

    # Evaluate
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
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Save confusion matrix
    cm_df = pd.DataFrame(cm)
    cm_filename = f"confusion_matrix_{SCENARIO}.csv"
    cm_df.to_csv(cm_filename, index=False)
    print(f"[SAVED] {cm_filename}")

    # ---------------------------
    # Plot accuracy
    # ---------------------------
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.title(f'Model Accuracy - {SCENARIO}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    acc_filename = f'accuracy_{SCENARIO}.png'
    plt.savefig(acc_filename)
    plt.close()
    print(f"[SAVED] {acc_filename}")

    # ---------------------------
    # Plot loss
    # ---------------------------
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title(f'Model Loss - {SCENARIO}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    loss_filename = f'loss_{SCENARIO}.png'
    plt.savefig(loss_filename)
    plt.close()
    print(f"[SAVED] {loss_filename}")

    # Add to summary table
    summary_rows.append([SCENARIO, loss, accuracy])

# ---------------------------------------------------------
# Create Summary Results Table
# ---------------------------------------------------------
summary_df = pd.DataFrame(summary_rows, columns=["Scenario", "Loss", "Accuracy"])
summary_df.to_csv("summary_results.csv", index=False)

print("\n====================================")
print("ALL SCENARIOS COMPLETED SUCCESSFULLY")
print("Summary saved to summary_results.csv")
print("====================================")
