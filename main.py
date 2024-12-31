import kagglehub
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from data_loading import merged_df
from data_loading import housing_df_monthly

#### checking if prices of Airbnbs change over the years #####
grouped_df = merged_df.groupby(["year"])["price"].mean()
grouped_df_districts = merged_df.groupby(["year", "neighbourhood"])["price"].mean()
print(merged_df[['year', 'neighbourhood', 'price']].info())
print(merged_df[['year', 'neighbourhood', 'price']].head())
print("This is the grouped dataframe per year: ", grouped_df)

#### plotting the price changes over the years
plt.figure(figsize=(10, 6))
plt.plot(grouped_df.index, grouped_df.values, marker='o', linestyle='-', linewidth=2)
plt.title("Average Prices by Year", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Price", fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(grouped_df_districts)
plt.figure(figsize=(20, 12))
for district in grouped_df_districts.index.get_level_values('neighbourhood').unique():
    district_data = grouped_df_districts.xs(district, level='neighbourhood')
    plt.plot(district_data.index, district_data.values, marker='o', linestyle='-', linewidth=2, label=district)

plt.title("Average Priceso AirBNB by Year per Districts", fontsize=18)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Price", fontsize=14)
plt.legend(title="Districts", fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12, rotation=45)  # Rotated x-axis labels
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


######## housing data ##############

housing_merged = housing_df_monthly.groupby(["year"])["average_price"].mean()

print(housing_merged)

plt.figure(figsize=(12, 6))
plt.plot(housing_merged.index, housing_merged.values, marker='o', linestyle='-', linewidth=2)
plt.title("Average Housing Prices by Year in London", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Price", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()