import os
import wandb
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from advanced_models.model_trainer import train_and_val_model
from advanced_models.model_trainer import load_data_from_cache
from ift6758.features import get_preprocessing_pipeline
from ift6758.visualizations import four_graphs

def advanced_logistic_regression(X_train, y_train, X_test, y_test, model_name, save_wandb = False):

    #Tranform model_name variable for folder name and wandb (slug)
    model_slug = model_name.replace(" ", "_").lower()

    # Create the base pipeline
    model = Pipeline([
        ('preprocessor', get_preprocessing_pipeline()),  # preprocessing step
        ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),  # Add polynomial features
        ('classifier',LogisticRegression(C=5, solver='saga', penalty='l1', max_iter=100, n_jobs=-1))  # Logistic Regression step
    ])

    # Fit the model with GridSearchCV
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    #Define dir where to save models
    model_dir = os.path.join("advanced_models", "artifacts")
    model_path = os.path.join(model_dir, f"{model_slug}.joblib")

    #Save model locally
    joblib.dump(model, model_path)

    if(save_wandb):

        #Initialize a wandb run
        wandb.init(project="advanced_logistic_regression",name=model_name, reinit=True)

        #Log model as an artifact
        artifact = wandb.Artifact(model_slug, type='model')    
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        wandb.log({"accuracy": accuracy})

        #Plot 4 graphs for each model and save then in wandb
        four_graphs(y_pred_proba[:, 1], y_test, model_name=model_name,save_wandb=save_wandb)

        #Save predicted probabilities and actuals labels to file
        data_path = os.path.join(model_dir, f"{model_slug}.npz")
        np.savez(data_path, y_pred_proba=y_pred_proba[:, 1], y_test=y_test)

        # Create a W&B artifact
        artifact = wandb.Artifact("data", type="dataset")
        artifact.add_file(data_path)
        wandb.log_artifact(artifact)

        #Finish the run
        wandb.finish()

# Load data from cache
X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_cache()

# Create a boolean mask for filtering goal_angle over 90
mask_train = X_train['goal_angle'].between(-90, 90)
mask_val = X_val['goal_angle'].between(-90, 90)

# Filter X_train and y_train using the same mask
X_train = X_train[mask_train]
y_train = y_train[mask_train]

# Filter X_val and y_val similarly
X_val = X_val[mask_val]
y_val = y_val[mask_val]

print("lauching model")
#Run advance logistic regression model
advanced_logistic_regression(X_train, y_train, X_val, y_val, "advanced_regression", save_wandb=True)
