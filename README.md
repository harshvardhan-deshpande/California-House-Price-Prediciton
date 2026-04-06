# California-House-Price-Prediciton
End-to-end machine learning project for housing price prediction using the California Housing dataset, featuring data preprocessing pipelines, stratified sampling, model comparison, and evaluation using cross-validation (best RMSE ≈ 47K with Random Forest).

⚙️ Workflow & Implementation
1️⃣ Data Loading
Loaded dataset using Pandas
Dataset contains housing features like median income, location, and house value
2️⃣ Stratified Sampling
Created income_cat using median_income
Used StratifiedShuffleSplit to:
Maintain distribution of income categories
Create train and test sets (80/20 split)


3️⃣ Feature & Label Separation
Target variable: median_house_value
Features: all other columns
4️⃣ Data Preprocessing Pipeline

Built using:

Pipeline
ColumnTransformer
🔹 Numerical Pipeline
Missing value handling using SimpleImputer (median)
Feature scaling using StandardScaler
🔹 Categorical Pipeline
Encoding using OneHotEncoder
Handled unknown categories safely
5️⃣ Data Transformation
Applied full pipeline using:
full_pipeline.fit_transform(housing)

6️⃣ Model Training
Trained and compared 3 models:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
7️⃣ Model Evaluation (Cross-Validation)
Used 10-fold Cross Validation
Metric: RMSE (Root Mean Squared Error)

8️⃣ Best Model Selection
Random Forest performed best
Handles non-linearity and reduces overfitting
9️⃣ Final Test Evaluation
Test set was kept completely unseen during training


Final predictions made using Random Forest
📊 Results
Model	RMSE (approx)
Linear Regression	~69K
Decision Tree	~69K
Random Forest	~49K (CV)
Final Test RMSE	~47K ✅
📌 Interpretation
Model error ≈ 47K
Mean house price ≈ 206K
Error ≈ 23%

🔍 Sample Predictions
Actual	Predicted
204600	205706
337400	338403
162500	221708
367400	280961

👉 Model performs well on most values but shows larger errors on extreme cases
