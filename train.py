# Standard imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import joblib
# from pprint import pprint

# Load data and check
data = pd.read_csv('data/iris.csv')
# pprint(data[:10])

# Train test split
train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

# Train model
mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)
mod_dt.fit(X_train,y_train)
y_pred = mod_dt.predict(X_test)

# Create model training report and metrics
print('The accuracy of the Decision Tree is',"{:.3f}".format(accuracy_score(y_pred,y_test)))
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).T
# report_df.to_csv("metrics.csv")
report_df.to_csv('metrics.csv', mode='a', header=False) 

# Write out trained model
joblib.dump(mod_dt, "models/dtree.joblib")
