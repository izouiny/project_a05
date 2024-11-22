import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import wandb

from model_trainer import train_and_val_model, train_and_test_model
from ift6758.features import get_preprocessing_pipeline
from ift6758.visualizations import four_graphs

def train_and_test_svm(C = 1.0, kernel = "rbf", use_wandb=True, close_wandb=True):
    """
    Train and test a SVM model

    Returns:
        model: The trained model
        y_pred: The predicted values
        y_proba: The predicted probabilities
        y_val: The true values
    """
    # Arguments for the model
    model_params = {
        "C": C,
        "kernel": kernel,
        "class_weight": None,
        "max_iter": 2000,
        "random_state": 42,
        "probability": True
    }

    # Create the model
    model = Pipeline([
        ('preprocessor', get_preprocessing_pipeline()),
        ('classifier', SVC(**model_params))
    ])

    return train_and_val_model(
        project="svm",
        model=model,
        model_slug="svm",
        model_name="SVM",
        model_params=model_params,
        use_wandb=use_wandb,
        close_wandb=close_wandb
    )

#-----------------------------------------------------
#-----------------------------------------------------
# if __name__ == "__main__":
#     use_wandb = True
#
#     # C_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
#     C_values = np.linspace(0.55, 0.95, 10)
#     # kernel_values = ["linear", "poly", "rbf", "sigmoid"]
#     kernel_values = ["rbf"]
#
#     for c in C_values:
#         for k in kernel_values:
#             model, y_pred, y_proba, y_val = train_and_test_svm(C=c, kernel=k, use_wandb=use_wandb, close_wandb=False)
#             four_graphs(y_proba[:, 1], y_val, f"svm", save_wandb=use_wandb)
#             if use_wandb:
#                 wandb.finish()


# if __name__ == "__main__":
#     use_wandb = True
#
#     model, y_pred, y_proba, y_val = train_and_test_svm(use_wandb=use_wandb, close_wandb=False)
#     four_graphs(y_proba[:, 1], y_val, "svm", save_wandb=use_wandb)
#
#     if use_wandb:
#         wandb.finish()


if __name__ == "__main__":
    train_and_test_model(
        model=Pipeline([
            ('preprocessor', get_preprocessing_pipeline()),
            ('classifier', SVC(C=0.55, kernel="rbf", class_weight=None, max_iter=2000, random_state=42, probability=True))
        ]),
        model_slug="svm"
    )