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
matplotlib.use('TkAgg')
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from scipy.stats import boxcox
import seaborn as sns
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from scipy.stats import entropy
import shap
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.svm import SVR

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

# Saving the filtered data
output_csv_path = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\data\airbnb_london\processed_data.csv"
df.to_csv(output_csv_path, index=False)
print(f"Processed data saved to {output_csv_path}")

############ pipeline can be run from here

# Loading  the saved processed DataFrame
input_csv_path = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\data\airbnb_london\processed_data.csv"
df = pd.read_csv(input_csv_path)
print(f"Processed data loaded from {input_csv_path}")



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


# Calculate the average price
average_price = df["price_per_night_dollar"].mean()
print(f"The average price per night is: ${average_price:.2f}")


output_path_boxplot = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\boxplot_price_per_night.png"


plt.figure(figsize=(10, 6))
plt.boxplot(df["price_per_night_dollar"], vert=False, patch_artist=True)
plt.title("Boxplot of Price Per Night (Dollar)")
plt.xlabel("Price Per Night (Dollar)")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_path_boxplot)
plt.show()

print(f"Boxplot saved at: {output_path_boxplot}")



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

# Calculate the correlation data
correlation_data = df[poi_columns + ["price_per_night_dollar"]].corr()

output_path_heatmap = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\correlation_heatmap.png"

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
plt.tight_layout()
plt.savefig(output_path_heatmap)
plt.show()

print(f"Heatmap saved at: {output_path_heatmap}")




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

# Calculate the correlation matrix for all features
#does not improve prediction
correlation_matrix = X.corr(method='pearson')

# Define the output path for saving the plot
output_path_corr_matrix = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\correlation_matrix.png"

# Create and save the heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(output_path_corr_matrix)
plt.show()

print(f"Correlation matrix saved at: {output_path_corr_matrix}")


# Identify highly correlated features (threshold > 0.9)
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]

# Drop highly correlated features
X = X.drop(columns=high_corr_features)

print(f"Features dropped due to high correlation: {high_corr_features}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Choose Scaler
#scaler = StandardScaler()
scaler = MinMaxScaler()

# Scaling
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
# Linear Regression with 5-Fold Cross-Validation
# -------------------------
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

linear_model = LinearRegression()

# cross validation
cv_results_mae = cross_val_score(
    linear_model, X_train_scaled, y_train, scoring='neg_mean_absolute_error', cv=kfold
)
cv_results_mse = cross_val_score(
    linear_model, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=kfold
)
cv_results_r2 = cross_val_score(
    linear_model, X_train_scaled, y_train, scoring='r2', cv=kfold
)

# Calculate the average metrics from cross-validation
avg_mae_cv = -np.mean(cv_results_mae)
avg_mse_cv = -np.mean(cv_results_mse)
avg_r2_cv = np.mean(cv_results_r2)


linear_model.fit(X_train_scaled, y_train)
y_pred_lr = linear_model.predict(X_test_scaled)


mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr) * 100
r2_lr = r2_score(y_test, y_pred_lr)
explained_var_lr = explained_variance_score(y_test, y_pred_lr)


print("\nLinear Regression Cross-Validation Results (5-Fold):")
print(f"Average MAE (CV): {avg_mae_cv:.2f}")
print(f"Average MSE (CV): {avg_mse_cv:.2f}")
print(f"Average R-squared (CV): {avg_r2_cv:.2f}")

print("\nLinear Regression Test Set Results:")
print(f"Mean Squared Error (MSE): {mse_lr:.2f}")
print(f"Mean Absolute Error (MAE): {mae_lr:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_lr:.2f}%")
print(f"R-squared (R2): {r2_lr:.2f}")
print(f"Explained Variance Score: {explained_var_lr:.2f}")


output_path_lr_eval = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\linear_regression_cv_evaluation.png"

metrics = ['MSE (Test)', 'MAE (Test)', 'MAPE (Test)', 'R2 (Test)', 'Explained Variance (Test)',
           'MAE (CV)', 'MSE (CV)', 'R2 (CV)']
values = [mse_lr, mae_lr, mape_lr, r2_lr, explained_var_lr, avg_mae_cv, avg_mse_cv, avg_r2_cv]

plt.figure(figsize=(12, 8))
plt.barh(metrics, values, color='skyblue')
plt.title("Linear Regression Evaluation Metrics (Test Set and Cross-Validation)", fontsize=16)
plt.xlabel("Values", fontsize=14)
plt.ylabel("Metrics", fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_path_lr_eval)
plt.show()

print(f"Linear Regression evaluation plot (CV and Test) saved at: {output_path_lr_eval}")



# -------------------------
# Ridge Regression
# -------------------------

ridge_params = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga']
}
"""
ridge_params = {
    'alpha': [0.01],
    'solver': ['auto']
}
"""

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
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mape_ridge = mean_absolute_percentage_error(y_test, y_pred_ridge) * 100
r2_ridge = r2_score(y_test, y_pred_ridge)
explained_var_ridge = explained_variance_score(y_test, y_pred_ridge)


print("\nRidge Regression Evaluation:")
print(f"Mean Squared Error (MSE): {mse_ridge:.2f}")
print(f"Mean Absolute Error (MAE): {mae_ridge:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_ridge:.2f}%")
print(f"R-squared (R2): {r2_ridge:.2f}")
print(f"Explained Variance Score: {explained_var_ridge:.2f}")

output_path_ridge_eval = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\ridge_evaluation.png"


metrics = ['MSE', 'MAE', 'MAPE', 'R2', 'Explained Variance']
values = [mse_ridge, mae_ridge, mape_ridge, r2_ridge, explained_var_ridge]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color='skyblue')
plt.title("Ridge Regression Evaluation Metrics", fontsize=14)
plt.ylabel("Values", fontsize=12)
plt.tight_layout()
plt.savefig(output_path_ridge_eval)
plt.show()


################################
# Manual Grid Search for XGBoost
################################

n_estimators_options = [100, 1000]
learning_rate_options = [0.01, 0.1]
max_depth_options = [3, 10]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

best_params = None
best_mae = float('inf')

for n_estimators in n_estimators_options:
    for learning_rate in learning_rate_options:
        for max_depth in max_depth_options:
            mae_scores = []

            #cross validation
            for train_idx, val_idx in kfold.split(X_train_scaled):
                X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]


                xgb_model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42,
                    verbosity=0
                )
                xgb_model.fit(X_train_cv, y_train_cv)


                y_pred_cv = xgb_model.predict(X_val_cv)
                mae_scores.append(mean_absolute_error(y_val_cv, y_pred_cv))

            # average MAE across folds
            avg_mae = np.mean(mae_scores)

            # check if current one is better
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_params = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth
                }

# Train the best XGBoost model on the entire training set
best_xgb_model = xgb.XGBRegressor(**best_params, random_state=42)
best_xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = best_xgb_model.predict(X_test_scaled)

# Evaluate the best model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb) * 100
r2_xgb = r2_score(y_test, y_pred_xgb)
explained_var_xgb = explained_variance_score(y_test, y_pred_xgb)

# Print results
print("\nBest Parameters for XGBoost:", best_params)
print(f"Best MAE from Cross-Validation: {best_mae:.2f}")
print("\nXGBoost Regressor Evaluation:")
print(f"Mean Squared Error (MSE): {mse_xgb:.2f}")
print(f"Mean Absolute Error (MAE): {mae_xgb:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_xgb:.2f}%")
print(f"R-squared (R2): {r2_xgb:.2f}")
print(f"Explained Variance Score: {explained_var_xgb:.2f}")




################################
# XGBoost
################################

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05, #0.1
    max_depth=10, #6
    random_state=42,
    verbosity=1
)

xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

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

xgb_importance = xgb_model.feature_importances_

# Create a DataFrame for visualization
xgb_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_importance
}).sort_values(by='Importance', ascending=False)


output_path_xgb_importance = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\xgb_feature_importance.png"

plt.figure(figsize=(12, 10))
plt.barh(xgb_importance_df['Feature'], xgb_importance_df['Importance'], color='skyblue')
plt.title("Feature Importance for XGBoost", fontsize=16)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=8)
plt.gca().invert_yaxis()
plt.xticks(fontsize=10)
plt.yticks(fontsize=8)
plt.tight_layout()


plt.savefig(output_path_xgb_importance)

plt.show()

################################
# K-Nearest Neighbors (KNN) Regressor
################################


knn_params = {
    'n_neighbors': [3, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

knn_model = KNeighborsRegressor()


knn_grid = GridSearchCV(
    estimator=knn_model,
    param_grid=knn_params,
    scoring='neg_mean_squared_error',
    cv=5,  # 5-fold cross-validation
    verbose=1
)

knn_grid.fit(X_train_scaled, y_train)

print("\nBest parameters for KNN Regressor:", knn_grid.best_params_)
print("Best score for KNN Regressor (MSE):", -knn_grid.best_score_)


best_knn_model = knn_grid.best_estimator_
y_pred_knn = best_knn_model.predict(X_test_scaled)

mse_knn = mean_squared_error(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mape_knn = mean_absolute_percentage_error(y_test, y_pred_knn) * 100
r2_knn = r2_score(y_test, y_pred_knn)
explained_var_knn = explained_variance_score(y_test, y_pred_knn)


print("\nKNN Regressor Evaluation:")
print(f"Mean Squared Error (MSE): {mse_knn:.2f}")
print(f"Mean Absolute Error (MAE): {mae_knn:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape_knn:.2f}%")
print(f"R-squared (R2): {r2_knn:.2f}")
print(f"Explained Variance Score: {explained_var_knn:.2f}")


output_path_knn_eval = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\knn_evaluation.png"


metrics = ['MSE', 'MAE', 'MAPE', 'R2', 'Explained Variance']
values = [mse_knn, mae_knn, mape_knn, r2_knn, explained_var_knn]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color='skyblue')
plt.title("KNN Regressor Evaluation Metrics", fontsize=14)
plt.ylabel("Values", fontsize=12)
plt.tight_layout()
plt.savefig(output_path_knn_eval)
plt.show()



##### Feature Importance ####

lr_importance = np.abs(linear_model.coef_)
lr_importance = lr_importance / lr_importance.sum()

ridge_importance = np.abs(best_ridge_model.coef_)
ridge_importance = ridge_importance / ridge_importance.sum()

xgb_importance = xgb_model.feature_importances_
xgb_importance = xgb_importance / xgb_importance.sum()


# Create a DataFrame for comparison
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Linear Regression': lr_importance,
    'Ridge Regression': ridge_importance,
    'XGBoost': xgb_importance,
}).set_index('Feature')

# Get the top 10 features by average importance across models
importance_df['Average Importance'] = importance_df.mean(axis=1)
top_features = importance_df.sort_values(by='Average Importance', ascending=False).head(10)


output_path_top_features = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\top_features_importance.png"


top_features.plot(kind='barh', figsize=(12, 8), title="Top 10 Features by Importance Across Models")
plt.xlabel("Normalized Importance")
plt.ylabel("Feature")
plt.tight_layout()


plt.savefig(output_path_top_features)

plt.show()


########### Summary of results ########

mae_values = {
    'Linear Regression': mae_lr,
    'Ridge Regression': mean_absolute_error(y_test, y_pred_ridge),
    'XGBoost': mean_absolute_error(y_test, y_pred_xgb),
    'KNN Regressor': mean_absolute_error(y_test, y_pred_knn)  # Include KNN MAE
}


mae_df = pd.DataFrame(list(mae_values.items()), columns=['Model', 'MAE'])


output_path_mae_comparison = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\mae_comparison.png"


plt.figure(figsize=(10, 6))
colors = ['skyblue', 'lightgreen', 'salmon', 'orange']  # Add a color for KNN
plt.bar(mae_df['Model'], mae_df['MAE'], color=colors)
plt.title("Mean Absolute Error (MAE) Comparison Across Models", fontsize=16)
plt.xlabel("Model", fontsize=14)
plt.ylabel("MAE", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()


plt.savefig(output_path_mae_comparison)

plt.show()

##########################################
######### SHAP Values ##################
#########################################

# Linear Regression
explainer_lr = shap.Explainer(linear_model.predict, X_train_scaled)
shap_values_lr = explainer_lr(X_test_scaled)

# Ridge Regression
explainer_ridge = shap.Explainer(best_ridge_model.predict, X_train_scaled)
shap_values_ridge = explainer_ridge(X_test_scaled)

# XGBoost
explainer_xgb = shap.Explainer(xgb_model, X_train_scaled)
shap_values_xgb = explainer_xgb(X_test_scaled)

output_path_shap_summary_lr = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\shap_summary_plot_lr.png"
output_path_shap_summary_ridge = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\shap_summary_plot_ridge.png"
output_path_shap_summary_xgb = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\shap_summary_plot_xgb.png"
output_path_shap_decision_xgb = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\shap_decision_plot_xgb.png"
output_path_shap_comparison = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\shap_comparison.png"

# Visualizations
shap.summary_plot(
    shap_values_lr,
    X_test_scaled,
    feature_names=X.columns,
    show=False
)
plt.tight_layout()
plt.savefig(output_path_shap_summary_lr)
plt.show()
print(f"SHAP summary plot for Linear Regression saved at: {output_path_shap_summary_lr}")


shap.summary_plot(
    shap_values_ridge,
    X_test_scaled,
    feature_names=X.columns,
    show=False
)
plt.tight_layout()
plt.savefig(output_path_shap_summary_ridge)
plt.show()
print(f"SHAP summary plot for Ridge Regression saved at: {output_path_shap_summary_ridge}")


shap.summary_plot(
    shap_values_xgb,
    X_test_scaled,
    feature_names=X.columns,
    show=False
)
plt.tight_layout()
plt.savefig(output_path_shap_summary_xgb)
plt.show()
print(f"SHAP summary plot for XGBoost saved at: {output_path_shap_summary_xgb}")
