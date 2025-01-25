import kagglehub
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from data_loading import merged_df
from data_loading import housing_df_monthly
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests


#### checking if prices of Airbnbs change over the years #####
grouped_df = merged_df.groupby(["year"])["price"].mean()
grouped_df_districts = merged_df.groupby(["year", "neighbourhood"])["price"].mean()
print(merged_df[['year', 'neighbourhood', 'price']].info())
print(merged_df[['year', 'neighbourhood', 'price']].head())
print("This is the grouped dataframe per year: ", grouped_df)

#### plotting the price changes over the years
# Filter years between 2010 and 2024
grouped_df_filtered = grouped_df[(grouped_df.index >= 2010) & (grouped_df.index <= 2024)]

# Save the visualization of average prices over the years (2010-2024)
output_path_avg_prices_year = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\avg_prices_by_year_2010_2024.png"

plt.figure(figsize=(10, 6))
plt.plot(grouped_df_filtered.index, grouped_df_filtered.values, marker='o', linestyle='-', linewidth=2, color='skyblue')
plt.title("Average Prices by Year (2010-2024)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Price", fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_path_avg_prices_year)
plt.show()

print(f"Average prices plot (2010-2024) saved at: {output_path_avg_prices_year}")

print(f"Visualization of average prices by year saved at: {output_path_avg_prices_year}")

# Save the visualization of average Airbnb prices by year per district
output_path_avg_prices_district = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\avg_prices_by_year_per_district.png"

print(grouped_df_districts)
plt.figure(figsize=(20, 12))
for district in grouped_df_districts.index.get_level_values('neighbourhood').unique():
    district_data = grouped_df_districts.xs(district, level='neighbourhood')
    plt.plot(district_data.index, district_data.values, marker='o', linestyle='-', linewidth=2, label=district)

plt.title("Average Prices Airbnb by Year per Districts", fontsize=18)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Price", fontsize=14)
plt.legend(title="Districts", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12, rotation=45)  # Rotated x-axis labels
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(output_path_avg_prices_district)
plt.show()

print(f"Visualization of average prices by year per district saved at: {output_path_avg_prices_district}")

####################################
######## housing data ##############
###################################

#### total London
housing_merged = housing_df_monthly.groupby(["year"])["average_price"].mean()

print(housing_merged)

output_path_total_london = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\housing_prices_total_london.png"

plt.figure(figsize=(12, 6))
plt.plot(housing_merged.index, housing_merged.values, marker='o', linestyle='-', linewidth=2, color='skyblue')
plt.title("Average Housing Prices by Year in London", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Price", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(output_path_total_london)
plt.show()

print(f"Total London housing prices visualization saved at: {output_path_total_london}")


#### per district
housing_merged_districts = housing_df_monthly.groupby(["year", "area"])["average_price"].mean()


# Save the visualization of housing prices by district as a PNG file
output_path_districts = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\housing_prices_districts.png"

plt.figure(figsize=(20, 12))
for area in housing_merged_districts.index.get_level_values('area').unique():
    area_data = housing_merged_districts.xs(area, level='area')
    plt.plot(area_data.index, area_data.values, marker='o', linestyle='-', linewidth=2, label=area)

plt.title("Average Housing Prices by Year per District in London", fontsize=18)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Price", fontsize=14)
plt.legend(title="Districts", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(output_path_districts)
plt.show()

print(f"Housing prices by district visualization saved at: {output_path_districts}")

###################################################################################################################
################################    RQ1: What is the influence of hotels and Airbnbs influences     ###############
################################   the local housing markets and economic activities?     ##########################
###################################################################################################################

print("Starting calculations for RQ1 -> What is the influence of hotels and Airbnbs influences the local housing markets and economic activities?")


#################################################
######### correlating housing and airbnb #########
##################################################

#######################
#### per London ########
########################

merged_df_housing_bnb = pd.merge(housing_merged, grouped_df, on="year", how="inner")
print(merged_df_housing_bnb)

correlation = merged_df_housing_bnb['average_price'].corr(merged_df_housing_bnb['price'])

print(f"Correlation between housing prices and Airbnb prices per year over whole London: {correlation:.2f}")

print(housing_df_monthly)

####### lagged correlation #########

lagged_correlations_london = {}

# Test lags from 1 to 3 years
for lag in range(1, 4):
    # Shift housing prices by the current lag
    merged_df_housing_bnb[f'housing_lag_{lag}'] = merged_df_housing_bnb['average_price'].shift(lag)

    # Calculate correlation between lagged housing prices and Airbnb prices
    correlation_housing = merged_df_housing_bnb[f'housing_lag_{lag}'].corr(merged_df_housing_bnb['price'])
    lagged_correlations_london[f'Housing Prices Lag {lag}'] = correlation_housing

    # Shift Airbnb prices by the current lag
    merged_df_housing_bnb[f'airbnb_lag_{lag}'] = merged_df_housing_bnb['price'].shift(lag)

    correlation_airbnb = merged_df_housing_bnb[f'airbnb_lag_{lag}'].corr(merged_df_housing_bnb['average_price'])
    lagged_correlations_london[f'Airbnb Prices Lag {lag}'] = correlation_airbnb

print("Lagged Correlations (Whole London):")
for key, value in lagged_correlations_london.items():
    print(f"{key}: {value:.2f}")


correlation_df_london = pd.DataFrame(list(lagged_correlations_london.items()), columns=['Lag', 'Correlation'])

output_path = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\lagged_correlations_london.png"

plt.figure(figsize=(10, 6))
plt.bar(correlation_df_london['Lag'], correlation_df_london['Correlation'], color='skyblue')
plt.title("Lagged Correlations Between Housing and Airbnb Prices (Whole London)", fontsize=16)
plt.xlabel("Lag", fontsize=14)
plt.ylabel("Correlation", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_path)
plt.show()

print(f"Lagged correlation visualization saved at: {output_path}")

##############################################
###### Granger Test for whole London #########
##############################################
granger_data = merged_df_housing_bnb[['average_price', 'price']].dropna()


# For testing if Airbnb prices Granger-cause housing prices
print("For testing if Airbnb prices Granger-cause housing prices")
grangercausalitytests(granger_data[['average_price', 'price']], maxlag=3)

# For testing if housing prices Granger-cause Airbnb prices
print("Testing if housing prices Granger-cause Airbnb prices")
grangercausalitytests(granger_data[['price', 'average_price']], maxlag=3)


granger_results = {}

# Test Airbnb prices -> housing prices
airbnb_to_housing = grangercausalitytests(granger_data[['average_price', 'price']], maxlag=3, verbose=False)
granger_results['Airbnb -> Housing'] = airbnb_to_housing

# Test housing prices -> Airbnb prices
housing_to_airbnb = grangercausalitytests(granger_data[['price', 'average_price']], maxlag=3, verbose=False)
granger_results['Housing -> Airbnb'] = housing_to_airbnb


visualization_data = []
test_stats = ["ssr_chi2test", "ssr_ftest", "lrtest", "params_ftest"]

for direction, results in granger_results.items():
    for lag, lag_results in results.items():
        for test_stat in test_stats:
            # Extract p-value for the test_stat
            try:
                if isinstance(lag_results[0], dict) and test_stat in lag_results[0]:
                    p_value = lag_results[0][test_stat][1]  # Extract p-value
                    visualization_data.append({
                        "Direction": direction,
                        "Lag": lag,
                        "Test Statistic": test_stat,
                        "p-value": p_value
                    })
            except Exception as e:
                print(f"Error processing {test_stat} for Lag {lag}: {e}")


####### Visualize granger results over London


# Save visualization of Granger causality results over London
output_path_granger_results = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\granger_causality_results_london.png"

visualization_df = pd.DataFrame(visualization_data)

plt.figure(figsize=(16, 8))
for direction in visualization_df['Direction'].unique():
    for test_stat in visualization_df['Test Statistic'].unique():
        subset = visualization_df[(visualization_df['Direction'] == direction) &
                                   (visualization_df['Test Statistic'] == test_stat)]
        plt.plot(
            subset['Lag'],
            subset['p-value'],
            marker='o',
            label=f"{direction} ({test_stat})"
        )

# Add threshold line
plt.axhline(0.05, color='red', linestyle='--', label='Significance Threshold (p=0.05)')

plt.xticks(range(1, 4), labels=[f"Lag {i}" for i in range(1, 4)], fontsize=12)
plt.xlabel("Lag", fontsize=14)
plt.ylabel("p-value", fontsize=14)
plt.title("Granger Causality Results (All Test Statistics) for Whole London", fontsize=16)
plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_path_granger_results)
plt.show()

print(f"Visualization of Granger causality results saved at: {output_path_granger_results}")



##############################################################
######### correlating housing and airbnb per District #########
################################################################


housing_df_monthly['area'] = housing_df_monthly['area'].str.lower()
merged_df['neighbourhood'] = merged_df['neighbourhood'].str.lower()

grouped_housing_districts = housing_df_monthly.groupby(["year", "area"])["average_price"].mean().reset_index()
grouped_airbnb_districts = merged_df.groupby(["year", "neighbourhood"])["price"].mean().reset_index()

merged_districts = pd.merge(
    grouped_housing_districts,
    grouped_airbnb_districts,
    left_on=["year", "area"],
    right_on=["year", "neighbourhood"],
    how="inner"
)

print("Merged dataset per district:")
print(merged_districts)

district_correlations = (
    merged_districts.groupby("area")
    .apply(lambda df: df["average_price"].corr(df["price"]))
    .reset_index(name="correlation")
)

# Save correlation bar plot between housing and Airbnb prices per district
output_path_district_correlation = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\district_correlation_barplot.png"

print("\nCorrelation between housing and Airbnb prices per district:")
print(district_correlations)

plt.figure(figsize=(12, 8))
sns.barplot(
    x="correlation",
    y="area",
    data=district_correlations.sort_values("correlation", ascending=False),
    palette="coolwarm"
)
plt.title("Correlation between Housing and Airbnb Prices per District", fontsize=16)
plt.xlabel("Correlation", fontsize=14)
plt.ylabel("District (Area)", fontsize=14)
plt.tight_layout()
plt.savefig(output_path_district_correlation)
plt.show()

print(f"Bar plot saved at: {output_path_district_correlation}")

# Save heatmap for correlations between years
output_path_heatmap = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\housing_airbnb_correlation_heatmap.png"

heatmap_data = merged_districts.pivot_table(
    index="area",
    columns="year",
    values=["average_price", "price"]
).corr().loc["average_price", "price"]

plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={'label': 'Correlation'},
    xticklabels=heatmap_data.columns,  # Columns correspond to Airbnb prices
    yticklabels=heatmap_data.index  # Rows correspond to Housing prices
)
plt.title("Correlation Heatmap: Housing vs Airbnb Prices per Year", fontsize=16)
plt.xlabel("Airbnb Prices (Year)", fontsize=14)
plt.ylabel("Housing Prices (Year)", fontsize=14)
plt.tight_layout()
plt.savefig(output_path_heatmap)
plt.show()

print(f"Heatmap saved at: {output_path_heatmap}")

######### calculating lagged correlation per districts

data = merged_districts[['year', 'average_price', 'price']].dropna()


lagged_correlations = {}

for lag in range(1, 4):
    # Shift housing prices by the current lag
    data[f'housing_lag_{lag}'] = data['average_price'].shift(lag)

    # Calculate correlation between lagged housing prices and Airbnb prices
    correlation = data[f'housing_lag_{lag}'].corr(data['price'])
    lagged_correlations[f'Housing Prices Lag {lag}'] = correlation

    # Shift Airbnb prices by the current lag
    data[f'airbnb_lag_{lag}'] = data['price'].shift(lag)

    # Calculate correlation between lagged Airbnb prices and housing prices
    correlation = data[f'airbnb_lag_{lag}'].corr(data['average_price'])
    lagged_correlations[f'Airbnb Prices Lag {lag}'] = correlation

print("Lagged Correlations:")
for key, value in lagged_correlations.items():
    print(f"{key}: {value:.2f}")


##### lagged correlations per district

lagged_correlations_per_district = {}


for district in merged_districts['area'].unique():
    district_data = merged_districts[merged_districts['area'] == district][['year', 'average_price', 'price']].dropna()

    district_lagged_corr = {}

    # Test lags from 1 to 3 years
    for lag in range(1, 4):
        # Shift housing prices by the current lag
        district_data[f'housing_lag_{lag}'] = district_data['average_price'].shift(lag)

        corr_housing = district_data[f'housing_lag_{lag}'].corr(district_data['price'])
        district_lagged_corr[f'Housing Prices Lag {lag}'] = corr_housing

        district_data[f'airbnb_lag_{lag}'] = district_data['price'].shift(lag)

        corr_airbnb = district_data[f'airbnb_lag_{lag}'].corr(district_data['average_price'])
        district_lagged_corr[f'Airbnb Prices Lag {lag}'] = corr_airbnb

    lagged_correlations_per_district[district] = district_lagged_corr


print("Lagged Correlations per District:")
for district, correlations in lagged_correlations_per_district.items():
    print(f"\nDistrict: {district}")
    for lag, value in correlations.items():
        print(f"  {lag}: {value:.2f}")


# Convert lagged correlations dictionary to a DataFrame
lagged_corr_df = pd.DataFrame.from_dict(lagged_correlations_per_district, orient='index')

# Extract data for Lag 1
lag_1_data = lagged_corr_df[['Housing Prices Lag 1', 'Airbnb Prices Lag 1']].reset_index()
lag_1_data.rename(columns={'index': 'District'}, inplace=True)

# Extract data for Lag 2
lag_2_data = lagged_corr_df[['Housing Prices Lag 2', 'Airbnb Prices Lag 2']].reset_index()
lag_2_data.rename(columns={'index': 'District'}, inplace=True)

# Save plot for Lag 1
output_path_lag_1 = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\lag_1_correlations.png"

plt.figure(figsize=(15, 8))
lag_1_data.set_index('District').plot(kind='bar', figsize=(15, 8))
plt.title("Lagged Correlations (Lag 1 Year)", fontsize=16)
plt.xlabel("District", fontsize=14)
plt.ylabel("Correlation", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.legend(title="Lag 1 Correlations", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_path_lag_1)
plt.show()

print(f"Lag 1 correlation plot saved at: {output_path_lag_1}")

# Save plot for Lag 2
output_path_lag_2 = r"C:\Users\kmallinger\Documents\GitHub\dopp-exe2\results\lag_2_correlations.png"

plt.figure(figsize=(15, 8))
lag_2_data.set_index('District').plot(kind='bar', figsize=(15, 8))
plt.title("Lagged Correlations (Lag 2 Years)", fontsize=16)
plt.xlabel("District", fontsize=14)
plt.ylabel("Correlation", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.legend(title="Lag 2 Correlations", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_path_lag_2)
plt.show()

print(f"Lag 2 correlation plot saved at: {output_path_lag_2}")
