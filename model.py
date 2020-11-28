import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
import pickle


data = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Dockers/master/BankNote_Authentication.csv')
print(data.head())

# Data is preprocessed

X = data.drop(['class'], axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Implementing Random Forest Classifier
rf = RFC(max_depth=10)
rf.fit(X_train, y_train)

# Prediction
y_pred = rf.predict(X_test)

# Evaluation
score = accuracy_score(y_test, y_pred)
print(score)

### Creating a Pickle file using serialization
pickle_out = open("classifier.pkl", "wb")
pickle.dump(rf, pickle_out)
pickle_out.close()

print(rf.predict([[2, 3, 4, 1]]))