from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import wandb

from model_trainer import train_and_val_model
from ift6758.features import get_preprocessing_pipeline
from ift6758.visualizations import four_graphs

from os import cpu_count

def train_and_test_random_forest(n_estimators = 300, max_depth = 12, use_wandb=True, close_wandb=True):
    """
    Train and test a random forest model

    Returns:
        model: The trained model
        y_pred: The predicted values
        y_proba: The predicted probabilities
        y_val: The true values
    """
    # Arguments for the model
    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": None,
        "random_state": 42,
        "n_jobs": max(1, cpu_count() - 2)
    }

    # Create the model
    model = Pipeline([
        ('preprocessor', get_preprocessing_pipeline()),
        ('classifier', RandomForestClassifier(**model_params))
    ])

    return train_and_val_model(
        project="random_forest",
        model=model,
        model_slug="random_forest",
        model_name="Random Forest",
        model_params=model_params,
        use_wandb=use_wandb,
        close_wandb=close_wandb
    )

#-----------------------------------------------------
#-----------------------------------------------------
if __name__ == "__main__":
    use_wandb = True

    n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    max_depth = [8, 10, 12, 14, 16, 18, 20]

    for n in n_estimators:
        for d in max_depth:
            model, y_pred, y_proba, y_val = train_and_test_random_forest(n_estimators=n, max_depth=d, use_wandb=use_wandb, close_wandb=False)
            four_graphs(y_proba[:, 1], y_val, f"random_forest", save_wandb=use_wandb)
            if use_wandb:
                wandb.finish()

    # model, y_pred, y_proba, y_val = train_and_test_random_forest(n_estimators=1000, max_depth=18, use_wandb=use_wandb, close_wandb=False)
    # four_graphs(y_proba[:, 1], y_val, "random_forest", save_wandb=use_wandb)
    #
    # if use_wandb:
    #     wandb.finish()
