import pandas as pd
import os
import mlflow
df = pd.read_csv('tested.csv')
print(df.head())

# checking for null values in the dataset.
print(df.isnull().sum())

print(len(df))
print(df.columns)

# filling missing values in Age column with median of the same column:

print(df['Age'])
df['Age'].fillna(df['Age'].median(), inplace=True)
print(df.isnull().sum())

# filling missing values in fare column with median of the same column:

print(df['Fare'])
df['Fare'].fillna(df['Fare'].median(), inplace=True)
print(df.isnull().sum())

# filling missing values in Cabin column with mode:

#but in dataset we have 418 records but in 1 column (Cabin).
# we have 327 missing values. So, we can handle it by two ways.
# 1.drop the complete column. OR fill with median or mode  

print(df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True))
print(df.isnull().sum())

# dropping columns which are not needed:

print(df.drop(['PassengerId','Name','SibSp','Embarked'], axis=1, inplace=True))
print(df.columns)

# convert (sex) column into male = 0 and female = 1:

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)

print(df['Sex'].head())
print(df['Sex'].dtype)

# saving prepared data:

os.makedirs('data', exist_ok=True)
df.to_csv('data/prepared.csv', index=False)

print("data preparation completed and saved to data/prepared.csv")

with mlflow.start_run(run_name="Data Preparation"):
    mlflow.set_tag("step", "data_preprocessing")

    mlflow.log_param("original_rows", len(df))
    mlflow.log_param("original_columns", df.shape[1])

    mlflow.log_param("age_fill", "median")
    mlflow.log_param("age_median_value", df['Age'].median())

    mlflow.log_param("fare_fill", "median")
    mlflow.log_param("fare_median_value", df['Fare'].median())

    mlflow.log_param("cabin_fill", "mode")
    mlflow.log_param("cabin_mode_value", df['Cabin'].mode()[0])

    mlflow.log_param("dropped_columns", "PassengerId, Name, SibSp, Embarked")
    mlflow.log_param("sex_encoding", "male=0, female=1")

    mlflow.log_metric("final_rows", len(df))
    mlflow.log_metric("final_columns", df.shape[1])

    mlflow.log_artifact("data/prepared.csv")