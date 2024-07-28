import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
dataset = pd.read_csv('/heart.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


dataset.head(3)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Training the built-in Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegressionCV
classifier_cv = LogisticRegressionCV(cv=5, random_state=0)
classifier_cv.fit(X_train, y_train)

# Predicting the Test set results with the built-in Logistic Regression model
y_pred_cv = classifier_cv.predict(X_test)

# Making the Confusion Matrix for the built-in Logistic Regression model
cm_cv = confusion_matrix(y_test, y_pred_cv)
print("Confusion Matrix (Built-in Logistic Regression):")
print(cm_cv)
print("Accuracy Score (Built-in Logistic Regression):", accuracy_score(y_test, y_pred_cv))


# Random Forest
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest")
print("Confusion Matrix:")
print(cm_rf)
print("Accuracy Score:", acc_rf)

# Support Vector Machine
svm_classifier = SVC(kernel = 'linear', random_state = 0)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("\nSupport Vector Machine (SVM)")
print("Confusion Matrix:")
print(cm_svm)
print("Accuracy Score:", acc_svm)


