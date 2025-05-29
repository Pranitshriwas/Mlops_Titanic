import requests
import json

url = "http://127.0.0.1:1234/invocations"
headers = {"Content-Type": "application/json"}

# Updated data without 'SibSp' and 'Embarked'
data = {
    "dataframe_split": {
        "columns": ["Pclass", "Sex", "Age", "Parch", "Fare"],
        "data": [[3, 1, 72.0, 2, 43.432]]  # example input
    }
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print("Prediction:", response.json())
