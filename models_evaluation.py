import wandb
import joblib
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from advanced_models.model_trainer import load_data_from_cache
from ift6758.features import load_train_test_x_y
from ift6758.visualizations import four_graphs, four_graphs_multiple_models

def load_model_from_artifact(artifact_name, model_filename, model_type='model'):
    """Load a model from a W&B artifact, downloading only if not already present."""
    # Define a local directory for caching downloaded artifacts
    local_cache_dir = './artifacts_cache'
    os.makedirs(local_cache_dir, exist_ok=True)
    
    # Local path to check if the model already exists
    local_model_path = os.path.join(local_cache_dir, model_filename)
    
    if os.path.exists(local_model_path):
        print(f"Model already exists locally: {local_model_path}")
        return joblib.load(local_model_path)
    else:
        # Download artifact if not already cached
        artifact = run.use_artifact(artifact_name, type=model_type)
        artifact_dir = artifact.download(root=local_cache_dir)
        model_path = os.path.join(artifact_dir, model_filename)
        
        # Cache the model to avoid re-downloading
        os.rename(model_path, local_model_path)
        print(f"Downloaded and cached model: {local_model_path}")
        return joblib.load(local_model_path)

# Ensure X_test or X_val only has the columns expected by the model
def align_features(model, X):
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
        missing_features = [f for f in expected_features if f not in X.columns]
        extra_features = [f for f in X.columns if f not in expected_features]

        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        if extra_features:
            print(f"Extra features in input data (ignored): {extra_features}")
            X = X[expected_features]  # Align the features in the correct order

    return X

def evaluate_models(X_test, y_test, models, model_names):

    #Empty dictionary for storing models results
    models_results = {}

    # Inside your loop where models are being evaluated:
    for i, model in enumerate(models):
        print(f"Evaluating {model_names[i]}...")
        X_test_aligned = align_features(model, X_test)

        # Convert DataFrame to DMatrix for XGBoost
        if isinstance(model, xgb.Booster):  # Check if it's an XGBoost model
            X_test_aligned = xgb.DMatrix(X_test_aligned)  # Convert to DMatrix
        
        # Predict labels
        y_pred = model.predict(X_test_aligned)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {model_names[i]}: {accuracy}")

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_aligned)
            positive_class_index = list(model.classes_).index(1)  # Confirm index of positive class
            models_results[model_names[i]] = (y_pred_proba[:, positive_class_index], y_test)
        else:
            print(f"{model_names[i]} does not support probability prediction.")
        
    return(models_results)

# Initialize wandb run
run = wandb.init()

# Load models
regression_distance = load_model_from_artifact(
    'IFT6758-2024-A05/simple_model_logistic_regression/goal_distance_regression:v0', 
    'goal_distance_regression.joblib'
)

regression_angle = load_model_from_artifact(
    'IFT6758-2024-A05/simple_model_logistic_regression/goal_angle_regression:v0', 
    'goal_angle_regression.joblib'
)

regression_both = load_model_from_artifact(
    'IFT6758-2024-A05/simple_model_logistic_regression/goal_distance_and_angle_regression:v0', 
    'goal_distance_and_angle_regression.joblib'
)

# xgboost = load_model_from_artifact(
#     'IFT6758-2024-A05/xgboost_feature_selection/artifacts/model/xgboost_identity:v0', 
#     'xgboost_identity.pkl',
#     model_type='model'
        
# )

svm = load_model_from_artifact(
    'IFT6758-2024-A05/ift6758-milestone-2/svm:v10', 
    'svm.pkl'
)

random_forest = load_model_from_artifact(
    'IFT6758-2024-A05/ift6758-milestone-2/random_forest:v79', 
    'random_forest.pkl'
)

MLP = load_model_from_artifact(
    'IFT6758-2024-A05/ift6758-milestone-2/MLP:v0', 
    'MLP.pkl'
)

advanced_regression = load_model_from_artifact(
    'IFT6758-2024-A05/advanced_logistic_regression/advanced_regression:v0', 
    'advanced_regression.joblib',
    model_type='model'
)

#Finish the run
wandb.finish()

def main():

    models = [regression_distance,regression_angle,regression_both,svm,random_forest,MLP,advanced_regression] #regression_distance,regression_angle,regression_both,svm,random_forest,MLP,advanced_regression
    model_names = ["Logistic regression distance","Logistic regression angle","Logistic regression both","SVM","Random forest","MLP","Advanced regression"] #"Logistic regression distance","Logistic regression angle","Logistic regression both","SVM","Random forest","MLP","Advanced regression"

    #Load data from cache
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_cache()

    #For validation set
    #Evaluate advanced models 
    results = evaluate_models(X_val, y_val, models=models, model_names=model_names)

    #Select specific models
    results = {name: results[name] for name in ["SVM","Random forest","MLP","Advanced regression"]}

    #Creates four graphs of performance
    four_graphs_multiple_models(results, folder_name="try_your_best_models")


    #Load data separately according to game_id
    #For training set "regular" games
    #Select for only game_type 03
    X_train, y_train, X_test, y_test = load_train_test_x_y(game_type=3)
    results = evaluate_models(X_test, y_test, models=models, model_names=model_names)

    #Select specific models
    results = {name: results[name] for name in ["Logistic regression distance","Logistic regression angle","Logistic regression both","Random forest"]}

    #Load the .npy file for regular games for XGBoost model
    y_pred_proba =  np.load(os.path.join("data", "Y_proba_saison.npy"))
    y_test = np.load(os.path.join("data", "Y_label_saison.npy"))
    results["XGBoost"] = (y_pred_proba, y_test)
    
    #Creates four graphs of performance
    four_graphs_multiple_models(results, folder_name="best_models_all_data")

    #For training set "brut match series"
    #Select for only game_type 02
    X_train, y_train, X_test, y_test = load_train_test_x_y(game_type=2)
    results = evaluate_models(X_test, y_test, models=models, model_names=model_names)

    #Select specific models
    results = {name: results[name] for name in ["Logistic regression distance","Logistic regression angle","Logistic regression both","Random forest"]}

    #Load the .npy file for series games for XGBoost
    y_pred_proba =  np.load(os.path.join("data", "Y_proba_serie.npy"))
    y_test = np.load(os.path.join("data", "Y_label_serie.npy"))
    results["XGBoost"] = (y_pred_proba, y_test)

    #Creates four graphs of performance
    four_graphs_multiple_models(results, folder_name="best_models_brut_games")

if __name__ == "__main__":
    main()











