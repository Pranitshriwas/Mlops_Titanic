import pandas as pd 
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load prepared data:
df = pd.read_csv('data/prepared.csv')
print(df.head())

# dropping non-numerical column
non_numeric_cols = df.select_dtypes(include=['object']).columns
print("Dropping non-numeric columns:", list(non_numeric_cols))
df.drop(columns=non_numeric_cols, inplace=True)


X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# starting with mlflow

with mlflow.start_run():
    # log params
    mlflow.log_param("model_type","LogisticRegression")

    # Train model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train,y_train)

    # model prediction
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred,y_test)

    # log model
    mlflow.sklearn.log_model(model,"model")

    # log metric
    mlflow.log_metric("accuracy",accuracy)

    print("Training completed and logged with the help of mlflow")
    print(f'Accuracy {accuracy:.3f}')


   # Save as pickle file
    import joblib
    joblib.dump(model, "model.pkl")
    print("Model also saved as model.pkl")