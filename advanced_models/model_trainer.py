from sklearn.metrics import accuracy_score
import wandb
import joblib

from ift6758.features import load_train_val_test_x_y


def train_and_val_model(model, model_params, model_slug: str, model_name: str, use_wandb=True):
    """
    Train and test a scikit learn model

    Returns:
        model: The trained model
        y_pred: The predicted values
        y_val: The true values
    """
    # Load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_train_val_test_x_y(test_size=0.2)

    # Print the columns names
    print(X_train.columns)

    # Init wandb if needed
    if use_wandb:
        wandb.init(
            project="ift6758-milestone-2",
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
