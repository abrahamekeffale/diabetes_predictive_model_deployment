
import pandas as pd

df=pd.read_csv(r'C:\Users\HP\Desktop\training 5\diabetes.csv')

print(df.head())

df.columns

df.info

df.isnull().sum()

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Separate features (X) and target variable (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

#
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Create a logistic regression model
model1 = LogisticRegression()

# Train the model
model1.fit(X_train, y_train)


# Create a decision tree model
model2 = DecisionTreeClassifier()

# Train the model
model2.fit(X_train, y_train)

#
# Create a random forest model
model3 = RandomForestClassifier(n_estimators=100)

# Train the model
model3.fit(X_train, y_train)

#
# Make predictions on the test set for each model
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)


# Evaluate each model's performance
print("Logistic Regression:")
print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))

print("Decision Tree:")
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))

print("Random Forest:")
print(classification_report(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have y_test (true labels) and y_pred (predicted labels)

cm = confusion_matrix(y_test, y_pred1)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


pickle.dump(model2,open('model.pkl','wb'))


