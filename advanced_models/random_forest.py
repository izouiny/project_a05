from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from model_trainer import train_and_val_model
from ift6758.features import get_preprocessing_pipeline
from ift6758.visualizations import four_graphs

def train_and_test_random_forest(use_wandb=True):
    """
    Train and test a random forest model

    Returns:
        model: The trained model
        y_pred: The predicted values
        y_val: The true values
    """
    # Arguments for the model
    model_params = {
        "n_estimators": 100,
        "random_state": 42
    }

    # Create the model
    model = Pipeline([
        ('preprocessor', get_preprocessing_pipeline()),
        ('classifier', RandomForestClassifier(**model_params))
    ])

    return train_and_val_model(
        model=model,
        model_slug="random_forest",
        model_name="Random Forest",
        model_params=model_params,
        use_wandb=use_wandb
    )

#-----------------------------------------------------
#-----------------------------------------------------
if __name__ == "__main__":
    model, y_pred, y_val = train_and_test_random_forest()
    four_graphs(y_pred, y_val, "random_forest", save_wandb=False)