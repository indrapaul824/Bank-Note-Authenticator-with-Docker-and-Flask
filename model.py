import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import pickle

# set random seed
seed = 90

data = pd.read_csv('data/BankNoteAuth.csv')
print(data.head())

# Data is preprocessed

X = data.drop(['class'], axis=1)
sc = StandardScaler()
X = sc.fit_transform(X)

y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Implementing Random Forest Classifier
rf = RFC(max_depth=10)
rf.fit(X_train, y_train)

# Prediction
y_pred = rf.predict(X_test)

# Evaluation
#score = accuracy_score(y_test, y_pred)
#print(score)

### Report training set scores
train_score = rf.score(X_train, y_train) * 100
#### Report testing set scores
test_score = rf.score(X_test, y_test) * 100

# write scores to a file
with open("metrics.txt", "w") as f:
    f.write("Training accuracy score: %2.2f%%\n" % train_score)
    f.write("Test accuracy score: %2.2f%%\n" % test_score)

##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################
# Calculate feature importance in random forest
importances = rf.feature_importances_
labels = data.columns
feature_data = pd.DataFrame(list(zip(labels, importances)), columns=["feature", "importance"])
feature_data = feature_data.sort_values(by='importance', ascending=False,)

# image formatting
axis_fs = 18  # fontsize
title_fs = 22  # fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_data)
ax.set_xlabel('Importance', fontsize=axis_fs)
ax.set_ylabel('Feature', fontsize=axis_fs)  # ylabel
ax.set_title('Random forest\nfeature importance', fontsize=title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()


##########################################
############ PLOT CONFUSION MATRIX  #############
##########################################

plot_confusion_matrix(rf, X_test, y_test)

plt.tight_layout()
plt.savefig("confmat.png", dpi=120)
    

### Creating a Pickle file using serialization
pickle_out = open("classifier.pkl", "wb")
pickle.dump(rf, pickle_out)
pickle_out.close()

