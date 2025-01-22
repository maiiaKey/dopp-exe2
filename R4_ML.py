import tarfile
#from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
# Set the backend to TkAgg
matplotlib.use('TkAgg')
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import numpy as np
from scipy.stats import boxcox
import seaborn as sns
import os


tar_file_path = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\data\airbnb_london\train.tar"
extracted_folder = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\data\airbnb_london\extracted"
csv_file_path = os.path.join(extracted_folder, "train.csv")

# Check if the train.csv file exists
if not os.path.exists(csv_file_path):
    print("train.csv not found. Extracting from the tar file...")
    print("Please download data from link in readme file and define location of tar file accordingly if not done so already...")

    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(path=extracted_folder)
        print(f"Files extracted to {extracted_folder}")

    for root, dirs, files in os.walk(extracted_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_file_path = os.path.join(root, file)
                print(f"Found dataset: {csv_file_path}")


dataset_path = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\data\airbnb_london\extracted\train.csv"
df = pd.read_csv(dataset_path)


df = df.drop(columns=["country"])
df = df[df["city"].str.lower() == "london"]
df = df.drop(columns=[
    "city", "latitude", "longitude", "id", "total_amenities",
    "minimum_maximum_nights", "maximum_maximum_nights",
    "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
    "is_superhost", "is_host_has_pic", "is_host_id_verified",
    "is_available", "host_for_log", "host_listings_log",
    "host_listings_homes_log", "host_response_time_ord",
    "host_response_rate_percent", "host_acceptance_rate_percent"
])


london_rows = df.shape[0]
print(f"Number of datapoints (rows) after filtering for 'london': {london_rows}")

# Remove outliers from 'price_per_night_dollar'
Q1 = df["price_per_night_dollar"].quantile(0.25)
Q3 = df["price_per_night_dollar"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df["price_per_night_dollar"] >= lower_bound) & (df["price_per_night_dollar"] <= upper_bound)]

rows_after_outlier_removal = df.shape[0]
print(f"Number of datapoints (rows) after outlier removal: {rows_after_outlier_removal}")


average_price = df["price_per_night_dollar"].mean()
print(f"The average price per night is: ${average_price:.2f}")

plt.figure(figsize=(10, 6))
plt.boxplot(df["price_per_night_dollar"], vert=False, patch_artist=True)
plt.title("Boxplot of Price Per Night (Dollar)")
plt.xlabel("Price Per Night (Dollar)")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()



# Summarize the columns ending with _500m, _1000m, _5000m
poi_500m_cols = [col for col in df.columns if col.endswith('_500m')]
poi_1000m_cols = [col for col in df.columns if col.endswith('_1000m')]
poi_5000m_cols = [col for col in df.columns if col.endswith('_5000m')]

# Create new summarized columns
df['sum_poi_500m'] = df[poi_500m_cols].sum(axis=1)
df['sum_poi_1000m'] = df[poi_1000m_cols].sum(axis=1)
df['sum_poi_5000m'] = df[poi_5000m_cols].sum(axis=1)

# Drop the original POI columns
df = df.drop(columns=poi_500m_cols + poi_1000m_cols + poi_5000m_cols)


# Calculate the correlation between new POI columns and the target variable
poi_columns = ['sum_poi_500m', 'sum_poi_1000m', 'sum_poi_5000m']
correlations = {}

for col in poi_columns:
    correlations[col] = df[col].corr(df["price_per_night_dollar"])

print("Correlation between POI columns and target variable:")
for col, corr_value in correlations.items():
    print(f"{col}: {corr_value:.4f}")

correlation_data = df[poi_columns + ["price_per_night_dollar"]].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_data,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    square=True,
    cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title("Correlation Matrix: POI Columns and Target Variable")
plt.show()

########################################
######## Start ML Pipeline ########
########################################

target_column = "price_per_night_dollar"
X = df.drop(columns=[target_column])
y = df[target_column]


"""
### did not improve the model 
# Apply skewness correction only to numeric features
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns  # Select only numeric columns

for col in numeric_columns:
    if np.abs(X[col].skew()) > 0.75:  # Threshold for skewness
        if (X[col] > 0).all():  # Ensure all values are positive for Box-Cox
            X[col], _ = boxcox(X[col])
        else:
            X[col] = np.log1p(X[col] - X[col].min() + 1)  # Log-transform adjusted for non-positive values

"""

# Correlation Matrix for all features and dropping highly correlated ones
### does also not improve results but looks nice
correlation_matrix = X.corr(method='pearson')

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix")
plt.show()

# Identify highly correlated features (threshold > 0.9)
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]

# Drop highly correlated features
X = X.drop(columns=high_corr_features)

print(f"Features dropped due to high correlation: {high_corr_features}")


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling for preprocessing
#scaler = StandardScaler()
scaler = MinMaxScaler()

# Scale the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
#did not improve results
# Select top 20 features using mutual information
selector = SelectKBest(score_func=mutual_info_regression, k=20)
X_train_scaled = selector.fit_transform(X_train_scaled, y_train)
X_test_scaled = selector.transform(X_test_scaled)
"""

# -------------------------
# Linear Regression Model
# -------------------------
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_lr = linear_model.predict(X_test_scaled)

# Evaluate Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
explained_var_lr = explained_variance_score(y_test, y_pred_lr)

print("\nLinear Regression Results:")
print(f"Mean Squared Error (MSE): {mse_lr:.2f}")
print(f"Mean Absolute Error (MAE): {mae_lr:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_lr * 100:.2f}%")
print(f"R-squared (R2): {r2_lr:.2f}")
print(f"Explained Variance Score: {explained_var_lr:.2f}")

# -------------------------
# Ridge Regression
# -------------------------

ridge_params = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga']
}

ridge_model = Ridge()

# Perform Grid Search for Ridge Regression
ridge_grid = GridSearchCV(
    estimator=ridge_model,
    param_grid=ridge_params,
    scoring='neg_mean_squared_error',
    cv=5,  # 5-fold cross-validation
    verbose=1
)

ridge_grid.fit(X_train_scaled, y_train)

# Best parameters and score
print("\nBest parameters for Ridge Regression:", ridge_grid.best_params_)
print("Best score for Ridge Regression (MSE):", -ridge_grid.best_score_)

# Use the best model to make predictions
best_ridge_model = ridge_grid.best_estimator_
y_pred_ridge = best_ridge_model.predict(X_test_scaled)

# Evaluate Ridge Regression with multiple metrics
print("\nRidge Regression Evaluation:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_ridge):.2f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_ridge):.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(y_test, y_pred_ridge) * 100:.2f}%")
print(f"R-squared (R2): {r2_score(y_test, y_pred_ridge):.2f}")
print(f"Explained Variance Score: {explained_variance_score(y_test, y_pred_ridge):.2f}")

# -------------------------
# XGBoost Model
# -------------------------
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    verbosity=1
)

xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluate XGBoost
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
explained_var_xgb = explained_variance_score(y_test, y_pred_xgb)

print("\nXGBoost Results:")
print(f"Mean Squared Error (MSE): {mse_xgb:.2f}")
print(f"Mean Absolute Error (MAE): {mae_xgb:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_xgb * 100:.2f}%")
print(f"R-squared (R2): {r2_xgb:.2f}")
print(f"Explained Variance Score: {explained_var_xgb:.2f}")

# Calculate feature importance for XGBoost
xgb_importance = xgb_model.feature_importances_

# Create a DataFrame for visualization
xgb_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_importance
}).sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(12, 10))
plt.barh(xgb_importance_df['Feature'], xgb_importance_df['Importance'], color='skyblue')
plt.title("Feature Importance for XGBoost", fontsize=16)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=8)
plt.gca().invert_yaxis()
plt.xticks(fontsize=10)
plt.yticks(fontsize=8)
plt.tight_layout()

plt.yticks(rotation=45)

plt.show()

"""
# -------------------------
# Random Forest Regressor with GridSearchCV
# -------------------------
# Define the parameter grid for Random Forest
rf_params = {
    'n_estimators': [50, 150],        
    'max_depth': [5, 15, None],       
    'min_samples_split': [2, 10],      
    'min_samples_leaf': [1, 4],        
}


rf_model = RandomForestRegressor(random_state=42)

rf_grid = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_params,
    scoring='neg_mean_squared_error',  # Optimization objective
    cv=5,  # 5-fold cross-validation
    verbose=1,  # Show progress
    n_jobs=-1  # Use all available cores
)


rf_grid.fit(X_train_scaled, y_train)


print("\nBest parameters for Random Forest:", rf_grid.best_params_)
print("Best score for Random Forest (MSE):", -rf_grid.best_score_)

best_rf_model = rf_grid.best_estimator_
y_pred_rf_tuned = best_rf_model.predict(X_test_scaled)

print("\nTuned Random Forest Evaluation:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_rf_tuned):.2f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_rf_tuned):.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(y_test, y_pred_rf_tuned) * 100:.2f}%")
print(f"R-squared (R2): {r2_score(y_test, y_pred_rf_tuned):.2f}")
print(f"Explained Variance Score: {explained_variance_score(y_test, y_pred_rf_tuned):.2f}")
"""

##### Feature Importance ####

lr_importance = np.abs(linear_model.coef_)
lr_importance = lr_importance / lr_importance.sum()

ridge_importance = np.abs(best_ridge_model.coef_)
ridge_importance = ridge_importance / ridge_importance.sum()

xgb_importance = xgb_model.feature_importances_
xgb_importance = xgb_importance / xgb_importance.sum()

# Calculate feature importance for Random Forest
#rf_importance = best_rf_model.feature_importances_
#rf_importance = rf_importance / rf_importance.sum()

# Create a DataFrame for comparison
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Linear Regression': lr_importance,
    'Ridge Regression': ridge_importance,
    'XGBoost': xgb_importance,
    #'Random Forest': rf_importance
}).set_index('Feature')

# Get the top 10 features by average importance across models
importance_df['Average Importance'] = importance_df.mean(axis=1)
top_features = importance_df.sort_values(by='Average Importance', ascending=False).head(10)

top_features.plot(kind='barh', figsize=(12, 8), title="Top 10 Features by Importance Across Models")
plt.xlabel("Normalized Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()