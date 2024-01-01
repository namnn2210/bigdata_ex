import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load dataset
df = pd.read_csv('df_gas.csv')  # Replace with your file path
df = df[['year', 'annual_consume']]  # Select relevant columns

# Split the dataset
train = df[df['year'].isin([2018, 2019])]
test = df[df['year'] == 2020]
X_train = train.drop('annual_consume', axis=1)
y_train = train['annual_consume']
X_test = test.drop('annual_consume', axis=1)
y_test = test['annual_consume']

# Define models and their hyperparameters
models = {
    'RandomForest': RandomForestRegressor,
    'GradientBoosting': GradientBoostingRegressor
}
params = {
    'RandomForest': [{'n_estimators': 100}, {'n_estimators': 200}, {'n_estimators': 300}],
    'GradientBoosting': [{'n_estimators': 100, 'learning_rate': 0.1}, {'n_estimators': 200, 'learning_rate': 0.05}, {'n_estimators': 300, 'learning_rate': 0.01}]
}

# Set up MLflow experiment
mlflow.set_experiment('gas_consumption_modeling')

# Training and logging with MLflow
for model_name, model in models.items():
    for param in params[model_name]:
        with mlflow.start_run():
            # Initialize and train the model
            clf = model(**param)
            clf.fit(X_train, y_train)

            # Make predictions and calculate metrics
            predictions = clf.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            rmse = mean_squared_error(y_test, predictions, squared=False)

            # Log parameters, metrics, and the model
            mlflow.log_params(param)
            mlflow.log_metrics({'mae': mae, 'r2': r2, 'rmse': rmse})
            mlflow.sklearn.log_model(clf, model_name)


