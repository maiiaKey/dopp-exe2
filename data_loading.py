import kagglehub
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

base_data_folder = "data"
housing_folder = os.path.join(base_data_folder, "housing_london")
os.makedirs(housing_folder, exist_ok=True)  # Create the folder if it doesn't exist

################ Housing Data Set ##################

path_housing = kagglehub.dataset_download("justinas/housing-in-london")
print("Path to dataset files housing:", path_housing)

housing_file_path = f"{path_housing}/housing_in_london_monthly_variables.csv"
housing_df_monthly = pd.read_csv(housing_file_path)

# Extract the year from the date and create a new 'year' column
housing_df_monthly['date'] = pd.to_datetime(housing_df_monthly['date'], errors='coerce')
housing_df_monthly['year'] = housing_df_monthly['date'].dt.year

output_path = os.path.join(housing_folder, "housing_in_london_monthly_variables.csv")
housing_df_monthly.to_csv(output_path, index=False)
print(f"Housing data saved to {output_path}")

print("Data of the housing DataFrame:")
print(housing_df_monthly.columns)
print(housing_df_monthly.head())
print(housing_df_monthly["area"].unique())

#path_airbnb = kagglehub.dataset_download("thedevastator/airbnb-prices-in-european-cities") #year of airbnbs not clear, therefore not really usable
#print("Path to dataset files airbnb:", path_airbnb)


################ Airbnb Dataset ###################

##### load local data for London airbnb files (loaded from: https://insideairbnb.com/get-the-data/)
folder_path = "data/airbnb_london"

listings_path = os.path.join(folder_path, "listings.csv")
listings_df = pd.read_csv(listings_path)

reviews_path = os.path.join(folder_path, "reviews.csv")
reviews_df = pd.read_csv(reviews_path)

# Check if merged file exists
merged_output_path = os.path.join(folder_path, "merged_listings_reviews.csv")
if not os.path.exists(merged_output_path):
    # Rename "listing_id" to "id" in reviews.csv to enable merge
    if "listing_id" in reviews_df.columns:
        reviews_df.rename(columns={"listing_id": "id"}, inplace=True)

    # Merge the data with reviews to have the dates on the listings. This can help to assess price changes over time per districts (RQ1)
    # Not sure how trustworthy the dates and prices are. In this merge, the prices from listings.csv are taken and merged with the dates of the review.
    merged_df = pd.merge(listings_df, reviews_df, on="id", how="outer")

    # Save the merged DataFrame
    merged_df.to_csv(merged_output_path, index=False)
    print(f"Merging complete. Saved to {merged_output_path}")
else:
    print(f"Merged file already exists at {merged_output_path}")
    merged_df = pd.read_csv(merged_output_path)

print("Data for the airbnb dataframe:")
print(merged_df.columns)
print(merged_df[['neighbourhood', 'price']].head())

if 'year' not in merged_df.columns:
    merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
    merged_df['year'] = merged_df['date'].dt.year
    merged_df['year'] = merged_df['year'].interpolate(method='linear').astype(int)
    print("Date column converted to datetime, and year column extracted.")
else:
    merged_df['year'] = merged_df['year'].interpolate(method='linear').astype(int)




