# fnn_task3.py
# Updated FNN model for Task 3 (SA, SB, SC)

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------
# Scenario configuration
# ----------------------------
scenarios = {
    'SA': {
        'train': 'Training-a1-a3-a0.csv',
        'test':  'Testing-a2-a4-a0.csv',
        'unseen_classes': [2, 4]
    },
    'SB': {
        'train': 'Training-a1-a2-a0.csv',
        'test':  'Testing-a1-a0.csv',
        'unseen_classes': []
    },
    'SC': {
        'train': 'Training-a1-a2-a0.csv',
        'test':  'Testing-a1-a2-a3.csv',
        'unseen_classes': [3]
    }
}

# ----------------------------
# Run Scenario Function
# ----------------------------
def run_scenario(name, cfg):

    print("\n==========================")
    print(f" Running Scenario {name} ")
    print("==========================")

    # Load CSV
    train_df = pd.read_csv(cfg['train'])
    test_df  = pd.read_csv(cfg['test'])

    # Separate features and labels
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test  = test_df.iloc[:, :-1].values
    y_test  = test_df.iloc[:, -1].values

    # ----------------------------
    # Build FNN Model (simple)
    # ----------------------------
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # ----------------------------
    # Train
    # ----------------------------
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        verbose=1,
        validation_data=(X_test, y_test)
    )

    # ----------------------------
    # Predictions
    # ----------------------------
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # ----------------------------
    # Overall Accuracy
    # ----------------------------
    overall = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {overall:.4f}")

    # ----------------------------
    # Unseen class accuracy (Task 3 requirement)
    # ----------------------------
    unseen = cfg["unseen_classes"]
    if unseen:
        mask = test_df.iloc[:, -1].isin(unseen)
        if mask.sum() > 0:
            unseen_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"Accuracy on unseen attack classes {unseen}: {unseen_acc:.4f}")
        else:
            print("No unseen class samples found.")
    else:
        print("No unseen classes in this scenario.")

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

# ----------------------------
# Main: run all scenarios
# ----------------------------
for scenario_name, cfg in scenarios.items():
    run_scenario(scenario_name, cfg)
