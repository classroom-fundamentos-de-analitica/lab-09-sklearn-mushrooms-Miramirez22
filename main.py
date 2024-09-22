import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load datasets
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

# Preprocess data
def preprocess_data(data):
    # Encode categorical variables
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
    return data, label_encoders

train_data, label_encoders = preprocess_data(train_data)
test_data, _ = preprocess_data(test_data)

# Split features and target
X_train = train_data.drop('type', axis=1)
y_train = train_data['type']
X_test = test_data.drop('type', axis=1)
y_test = test_data['type']

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)