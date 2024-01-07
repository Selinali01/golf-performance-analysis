import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set variables
# Import either PGA or LPGA data
gender = "lpga" 
# Choose Model: linear / random_forest / gradient_boosting / neural_network
model = "random_forest"

if (gender == "pga"):
    from pga import X_train, X_test, y_train, y_test, X, y, X_test_scaled, X_train_scaled
else:
    from lpga import X_train, X_test, y_train, y_test, X, y, X_test_scaled, X_train_scaled

# Fit Model depending 
if (model == "linear"):
    # Linear regression model 
    lr = LinearRegression()
    print(X_train_scaled)

    # Check if X_train_scaled or X_test_scaled has Nan values
    #Fit on scaled training data
    lr.fit(X_train_scaled, y_train)
    
    # Print linear regression model coefficients in table
    print('Intercept: ', lr.intercept_)
    # Label coefficients with feature names
    coefficients = pd.DataFrame(data=lr.coef_, index=X.columns, columns=['Coefficients'])
    print(coefficients)
    #print('Coefficients: ', lr.coef_)

    # Make predictions on the test data

    y_pred = lr.predict(X_test_scaled)

elif (model == "random_forest"):
    rf = RandomForestRegressor()
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

elif(model == "gradient_boosting"):
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor()
    gb.fit(X_train_scaled, y_train)
    y_pred = gb.predict(X_test_scaled)

elif(model == "neural_network"):
    from sklearn.neural_network import MLPRegressor
    nn = MLPRegressor()
    nn.fit(X_train_scaled, y_train)
    y_pred = nn.predict(X_test_scaled)


# Model Analysis
# Calculate the MSE and MAE and RMSE and R^2
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error: ', mse)
print('Mean Absolute Error: ', mae)
print('Root Mean Squared Error: ', rmse)
print('R^2: ', r2)

# Residual plot
import matplotlib.pyplot as plt
import seaborn as sns

# new empty plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.hlines(y=0, xmin = y_pred.min(), xmax = y_pred.max(), colors = 'r')
#plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.show()

# 2. Prediction vs Actual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# feature importance plot
feature_importance = pd.DataFrame(data={
    'Feature': X.columns,
    'Importance': lr.coef_
})
feature_importance = feature_importance.sort_values(by='Importance', key=abs, ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Check each feature's statistical significance
import statsmodels.api as sm
X_train_scaled = sm.add_constant(X_train_scaled)
model = sm.OLS(y_train, X_train_scaled)
results = model.fit()
print(results.summary())


# Print feature importance in table
print(feature_importance)

# Check for overfitting with cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr, X_train_scaled, y_train, cv=5)
print(scores)



