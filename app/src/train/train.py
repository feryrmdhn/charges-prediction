import argparse
import os
import pandas as pd
import numpy as np
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

# def input_fn(request_body, request_content_type):
#     if request_content_type == 'application/json':
#         input_data = json.loads(request_body)
#         return np.array(input_data)
#     else:
#         raise ValueError(f"Unsupported content type: {request_content_type}")

# def predict_fn(input_data, model):
#     return model.predict(input_data)

# def output_fn(prediction, accept):
#     return json.dumps(prediction.tolist())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--random_state', type=int, default=42)
    
    # SageMaker parameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args, _ = parser.parse_known_args()
    
    # Set hyperparameters
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split
    random_state = args.random_state
    
    # Load data train
    training_dir = args.train
    train_data_path = os.path.join(training_dir, "insurance_data_cleaned.csv")
    
    data = pd.read_csv(train_data_path)
    
    # Split features and target
    # Target -> "charges"
    X = data.drop('charges', axis=1)
    y = data['charges']
    
    # Split training and test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Print metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)

    print("\n" + "-" * 40)
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Validation R²: {val_r2:.4f}")
    
    # Save model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path, protocol=4)
    print("\n" + "-" * 40)
    print(f"Model Saved!")