import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb
import joblib
import pandas as pd

from ift6758.features import load_train_val_test_x_y

def train_and_val_model(
        model,
        model_params,
        model_slug: str,
        model_name: str,
        project: str = "ift6758-milestone-2",
        use_wandb=True,
        close_wandb=True
):
    """
    Train and test a scikit learn model

    Returns:
        model: The trained model
        y_pred: The predicted values
        y_proba: The predicted probabilities
        y_val: The true values
    """
    # Load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_cache()

    # Print the columns names
    print("Columns names:", X_train.columns)

    # Init wandb if needed
    if use_wandb:
        wandb.init(
            project=project,
            name=model_slug,
            config={
                "architecture": model_name,
                "dataset": "all features",
                **model_params
        })

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, f"artifacts/{model_slug}.pkl")

    # Evaluate the model
    y_proba = model.predict_proba(X_val)
    y_pred = np.argmax(y_proba, axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_proba[:, 1])

    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc_score}")

    if use_wandb:
        # Send the model to wandb
        artifact = wandb.Artifact(model_slug, type="model")
        artifact.add_file(f"artifacts/{model_slug}.pkl")

        # Log the model and accuracy
        wandb.log_artifact(artifact)
        wandb.log({"accuracy": accuracy})
        wandb.log({"auc": auc_score})

        # Finish the run
        if close_wandb:
            wandb.finish()

    return model, y_pred, y_proba, y_val


def train_and_test_model(
        model,
        model_slug: str,
):
    # Load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_cache()

    # Join the train and validation sets
    X_train = pd.concat([X_train, X_val])
    y_train = pd.concat([y_train, y_val])

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, f"artifacts/{model_slug}.pkl")

    # Evaluate the model
    y_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba[:, 1])

    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc_score}")

    # Save the results to npz
    np.savez(f"artifacts/{model_slug}.npz", y_pred_proba=y_proba[:, 1], y_test=y_test)


def load_data_from_cache():
    """
    Load the data once and save it
    Load the data and split it into train, validation and test sets
    """
    # Load the data once and save it
    folder = "./cache"
    if not os.path.exists(folder):
        os.makedirs(folder)
        # Load the data
        X_train, y_train, X_val, y_val, X_test, y_test = load_train_val_test_x_y(test_size=0.2)
        # Save the data
        joblib.dump(X_train, f"{folder}/X_train.pkl")
        joblib.dump(y_train, f"{folder}/y_train.pkl")
        joblib.dump(X_val, f"{folder}/X_val.pkl")
        joblib.dump(y_val, f"{folder}/y_val.pkl")
        joblib.dump(X_test, f"{folder}/X_test.pkl")
        joblib.dump(y_test, f"{folder}/y_test.pkl")
    else:
        # Load the data
        X_train = joblib.load(f"{folder}/X_train.pkl")
        y_train = joblib.load(f"{folder}/y_train.pkl")
        X_val = joblib.load(f"{folder}/X_val.pkl")
        y_val = joblib.load(f"{folder}/y_val.pkl")
        X_test = joblib.load(f"{folder}/X_test.pkl")
        y_test = joblib.load(f"{folder}/y_test.pkl")

    return X_train, y_train, X_val, y_val, X_test, y_test