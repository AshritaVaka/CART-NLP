import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv('/winequality-red.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling for SVR
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# Random Forest Regression
regressor_rf = RandomForestRegressor(n_estimators=10, random_state=0)
regressor_rf.fit(X_train, y_train)

y_pred_rf = regressor_rf.predict(X_test)

y_pred_rf_rounded = np.round(y_pred_rf)

# Random Forest Regression
cm_rf = confusion_matrix(y_test, y_pred_rf_rounded)
accuracy_rf = accuracy_score(y_test, y_pred_rf_rounded)
print("Random Forest Regression Confusion Matrix:\n", cm_rf)
print("Random Forest Regression Accuracy:", accuracy_rf)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X[:, 0], y, color='red')
plt.plot(X_grid, regressor_rf.predict(np.c_[X_grid, np.zeros((len(X_grid), X.shape[1] - 1))]), color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Feature')
plt.ylabel('Quality')
plt.show()

# Decision Tree Regression
regressor_dt = DecisionTreeRegressor(random_state=0)
regressor_dt.fit(X_train, y_train)

y_pred_dt = regressor_dt.predict(X_test)

y_pred_dt_rounded = np.round(y_pred_dt)

# Decision Tree Regression
cm_dt = confusion_matrix(y_test, y_pred_dt_rounded)
accuracy_dt = accuracy_score(y_test, y_pred_dt_rounded)
print("Decision Tree Regression Confusion Matrix:\n", cm_dt)
print("Decision Tree Regression Accuracy:", accuracy_dt)

# Visualising the Decision Tree Regression results (higher resolution)
plt.scatter(X[:, 0], y, color='red')
plt.plot(X_grid, regressor_dt.predict(np.c_[X_grid, np.zeros((len(X_grid), X.shape[1] - 1))]), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Feature')
plt.ylabel('Quality')
plt.show()


# Support Vector Regression (SVR)
regressor_svr = SVR(kernel='rbf')
regressor_svr.fit(X_train_scaled, y_train_scaled)

y_pred_svr = sc_y.inverse_transform(regressor_svr.predict(X_test_scaled).reshape(-1, 1)).ravel()

y_pred_svr_rounded = np.round(y_pred_svr)

# Support Vector Regression
cm_svr = confusion_matrix(y_test, y_pred_svr_rounded)
accuracy_svr = accuracy_score(y_test, y_pred_svr_rounded)
print("Support Vector Regression Confusion Matrix:\n", cm_svr)
print("Support Vector Regression Accuracy:", accuracy_svr)

# Visualising the SVR results (higher resolution)
plt.scatter(sc_X.inverse_transform(X_train)[:, 0], sc_y.inverse_transform(y_train.reshape(-1, 1)), color='red')
plt.plot(X_grid, sc_y.inverse_transform(regressor_svr.predict(sc_X.transform(np.c_[X_grid, np.zeros((len(X_grid), X.shape[1] - 1))])).reshape(-1, 1)), color='blue')
plt.title('Support Vector Regression (High Resolution)')
plt.xlabel('Feature')
plt.ylabel('Quality')
plt.show()


