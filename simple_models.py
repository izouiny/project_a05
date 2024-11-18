from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibrationDisplay
#from sklearn.model_selection import train_test_split
import pandas as pd
import wandb
import joblib
import matplotlib.pyplot as plt


# Load training set CSV
training_set = pd.read_csv('data_for_models/training_set.csv')

# Load validation set CSV
validation_set = pd.read_csv('data_for_models/validation_set.csv')

def simple_logistic_regression(feature):

    # Prepare the data
    x_train = training_set[[feature]] 
    x_val = validation_set[[feature]]  
    y_train = training_set['is_goal']
    y_val = validation_set['is_goal']

    # Train the logistic regression model
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Predict labels for validation set
    y_pred = clf.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy is: ")
    print(accuracy)

    #Predicted labels
    y_pred = clf.predict(x_val)

    #Predicted probabilities for each type of label
    y_pred_proba = clf.predict_proba(x_val)

    #Combine predicted class labels and probabilities into a dataframe for easier interpretation
    summary_df = pd.DataFrame({
        'real_label':y_val,
        'predicted_label': y_pred,
        'probability_no_goal': y_pred_proba[:, 0],
        'probability_goal': y_pred_proba[:, 1],
        feature: validation_set[feature]
    })

    #Save dataframe
    summary_df.to_csv(feature+"_model/logistic_regression_goal_distance_model.csv")

    #Create the calibration display from the validation set
    CalibrationDisplay.from_estimator(clf, x_val, y_val, n_bins=10, strategy='uniform')

    # Add plot titles and labels
    plt.title('Reliability Diagram (Calibration Curve)', fontsize=16)
    plt.xlabel('Mean Predicted Probability', fontsize=14)
    plt.ylabel('Fraction of Positives', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()# Fit the logistic regression model

    #Save model locally
    joblib.dump(clf, feature+'_model/simple_logistic_regression.joblib')

    #Initialize a wandb run
    wandb.init(project="ift6758-milestone-2", name=(feature+"_logistic_regression"), reinit=True)

    #Log model as an artifact
    artifact = wandb.Artifact('simple_logistic_regression', type='model')
    artifact.add_file(feature+'_model/simple_logistic_regression.joblib')
    wandb.log_artifact(artifact)
    wandb.log({"accuracy": accuracy})

    #Finish the run
    wandb.finish()


simple_logistic_regression('goal_distance')
simple_logistic_regression('goal_angle')