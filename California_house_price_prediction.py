import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
# 1. Load the dataset
housing = pd.read_csv("housing.csv")

# 2. Create a stratified test set
housing['income_cat'] = pd.cut(housing["median_income"], 
                               bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1) # We will work on this data
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1) # Set aside the test data

# We will work on the copy of training data 
housing = strat_train_set.copy()

# 3. Seperate features and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# print(housing, housing_labels)

# 4. List the numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. Lets make the pipeline 

# For numerical columns
num_pipline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# For categorical columns
cat_pipline = Pipeline([ 
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Construct the full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipline, num_attribs), 
    ('cat', cat_pipline, cat_attribs)
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds= lin_reg.predict(housing_prepared)
# lin_rmse= root_mean_squared_error(housing_labels , lin_preds)
lin_rmses = -cross_val_score(lin_reg , housing_prepared ,  housing_labels ,scoring="neg_root_mean_squared_error" , cv=10)
# print(f" the neg root mean squared error of linear regression is {lin_rmses.mean()}")
 
# Decision Tree
dec_reg = DecisionTreeRegressor(random_state=42)
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)

# print(f"the root mean sqaure error of decision tree is {dec_rmse}")
dec_rmses = -cross_val_score(dec_reg , housing_prepared, housing_labels, scoring='neg_root_mean_squared_error' , cv=10) 
# print(f"the root mean sqaure error of decsion tree  is {dec_rmses.mean()}")

# Random Forest
random_forest_reg = RandomForestRegressor(random_state=42)
random_forest_reg.fit(housing_prepared, housing_labels)
random_forst_pred =random_forest_reg.predict(housing_prepared)
# random_forest_rmse = root_mean_squared_error(housing_labels , random_forst_pred)
forest_rmses = -cross_val_score(
    random_forest_reg,
    housing_prepared,
    housing_labels,
    scoring='neg_root_mean_squared_error',
    cv=10
)

# print("Random Forest RMSE:", forest_rmses.mean())

# ================= TEST SET EVALUATION =================

# 1. Separate test features and labels
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# 2. Transform test data using SAME pipeline (no fit!)
X_test_prepared = full_pipeline.transform(X_test)

# 3. Use best model (Random Forest)
final_model = random_forest_reg

# 4. Predict on test set
final_preds = final_model.predict(X_test_prepared)

# 5. Calculate RMSE
final_rmse = root_mean_squared_error(y_test, final_preds)

# print("Final Test RMSE:", final_rmse)
# print(y_test.mean())

comparison = pd.DataFrame({
    "Actual :" : y_test[:20],
    "Predicted :" : final_preds[:20]
})
print(comparison)


 
