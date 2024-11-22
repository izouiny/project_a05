from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import wandb
import joblib

from ift6758.features import load_train_val_test_x_y, get_preprocessing_pipeline

def train_and_test_random_forest(use_wandb=True):
    """
    Train and test a random forest model

    Returns:
        model: The trained model
        y_pred: The predicted values
        y_val: The true values
    """

    # Model slug name
    model_slug = "random_forest"

    # Load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_train_val_test_x_y(test_size=0.2)

    # Arguments for the model
    model_args = {
        "n_estimators": 100,
        "random_state": 42
    }

    # Create the model
    model = Pipeline([
        ('preprocessor', get_preprocessing_pipeline()),
        ('classifier', RandomForestClassifier(**model_args))
    ])

    # Init wandb if needed
    if use_wandb:
        wandb.init(
            project="ift6758-milestone-2",
            name=model_slug,
            config={
                "architecture": "Random Forest",
                "dataset": "all features",
                **model_args
        })

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, f"artifacts/{model_slug}.pkl")

    # Evaluate the model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_pred, y_val)

    if use_wandb:
        # Send the model to wandb
        artifact = wandb.Artifact(model_slug, type="model")
        artifact.add_file(f"artifacts/{model_slug}.pkl")

        # Log the model and accuracy
        wandb.log_artifact(artifact)
        wandb.log({"accuracy": accuracy})

        # Finish the run
        wandb.finish()

    return model, y_pred, y_val

#-----------------------------------------------------
#-----------------------------------------------------
if __name__ == "__main__":
    train_and_test_random_forest()