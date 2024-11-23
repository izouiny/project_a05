import sys
sys.path.append('.')
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from model_trainer import train_and_val_model
from ift6758.features import get_preprocessing_pipeline
from ift6758.visualizations import four_graphs
from os import cpu_count

def train_and_test_mlp(use_wandb=True):
    """
    Train and test a MLP model

    Returns:
        model: The trained model
        y_pred: The predicted values
        y_val: The true values
    """
    # Arguments for the model
    model_params = {
        "hidden_layer_sizes": 100,
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "learning_rate": "constant"
    }

    # Create the model
    model = Pipeline([
        ('preprocessor', get_preprocessing_pipeline()),
        ('classifier', MLPClassifier(**model_params))
    ])

    return train_and_val_model(
        model=model,
        model_slug="MLP",
        model_name="MLP",
        model_params=model_params,
        use_wandb=use_wandb
    )

#-----------------------------------------------------
#-----------------------------------------------------
if __name__ == "__main__":
    model, y_proba, y_pred, y_val = train_and_test_mlp(use_wandb=False)
    auc = four_graphs(y_proba, y_val, "MLP", save_wandb=False)
    print(f"AUC : {auc}")