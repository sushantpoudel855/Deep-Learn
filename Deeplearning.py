# ================================
# DWIT 202 Assignment
# Need of Deep Learning (Where ML Fails)
# Dataset: Kaggle MNIST CSV (single file)
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical


# -------------------------------
# STEP 1: Load dataset
# -------------------------------

df = pd.read_csv("mnist_test.csv")   # <-- make sure filename is correct
print("Dataset shape:", df.shape)

# -------------------------------
# STEP 2: Split features & labels
# -------------------------------

X = df.drop("label", axis=1)
y = df["label"]

# -------------------------------
# STEP 3: Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# =====================================================
# TASK 1: MACHINE LEARNING MODEL (Logistic Regression)
# =====================================================

ml_model = LogisticRegression(max_iter=1000)
ml_model.fit(X_train, y_train)

y_pred_ml = ml_model.predict(X_test)
ml_accuracy = accuracy_score(y_test, y_pred_ml)

print("\nLogistic Regression Accuracy:", ml_accuracy)

# =====================================================
# TASK 2: DEEP LEARNING MODEL (Neural Network)
# =====================================================

# Normalize pixel values
X_train = X_train.values / 255.0
X_test = X_test.values / 255.0

# Reshape into 28x28 images
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# One-hot encode labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build Neural Network
dl_model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
dl_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
dl_model.fit(
    X_train,
    y_train_cat,
    epochs=15,
    batch_size=32
)

# Evaluate model
loss, dl_accuracy = dl_model.evaluate(X_test, y_test_cat)

print("\nNeural Network Accuracy:", dl_accuracy)

# -------------------------------
# END OF CODE
# -------------------------------
