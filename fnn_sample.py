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
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()_
