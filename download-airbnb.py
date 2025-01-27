import os
import pandas as pd
import requests
import gzip
import shutil
import argparse

# Define the function to download and preprocess data
def download_and_preprocess(url_file, output_file, drop_columns, download_folder):
    # Create the download folder if it doesn't exist
    os.makedirs(download_folder, exist_ok=True)

    # Read URLs from the input text file
    with open(url_file, 'r') as f:
        urls = f.read().splitlines()

    combined_data = []  # List to store processed data

    for url in urls:
        try:
            # Extract country and city from the URL
            parts = url.split('/')
            country = parts[3]
            city = parts[5] if len(parts) > 7 else None

            # Construct the filename
            compressed_file = os.path.join(download_folder, f"{country}_{city if city else 'data'}.csv.gz")

            # Skip download if the file already exists
            if os.path.exists(compressed_file):
                print(f"Skipping download, file already exists: {compressed_file}")
            else:
                # Download the file
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(compressed_file, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {compressed_file}")
                else:
                    print(f"Failed to download {url}. HTTP status code: {response.status_code}")
                    continue

            # Unzip the file
            extracted_file = compressed_file.replace('.gz', '')
            if not os.path.exists(extracted_file):
                with gzip.open(compressed_file, 'rb') as f_in:
                    with open(extracted_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Extracted: {extracted_file}")

            # Load the CSV and preprocess
            df = pd.read_csv(extracted_file)
            
            # Drop unwanted columns
            df = df.drop(columns=drop_columns, errors='ignore')

            # Drop rows where the price field is NaN
            df = df.dropna(subset=['price'])

            # Add country and city columns
            df['country'] = country
            df['city'] = city if city else 'Unknown'

            # Append to the combined data
            combined_data.append(df)
        except Exception as e:
            print(f"Error processing {url}: {e}")

    # Combine all data into a single DataFrame
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess Airbnb data.")
    parser.add_argument("url_file", type=str, help="Path to the text file containing URLs.")
    parser.add_argument("output_file", type=str, help="Path to save the combined CSV file.")
    parser.add_argument("download_folder", type=str, help="Folder to save downloaded files.")
    drop_columns= ["host_location","host_verifications","calendar_updated","first_review","last_review","license","host_about","listing_url","scrape_id","last_scraped","source","name","description","neighborhood_overview","picture_url","host_id","host_url","host_name","host_thumbnail_url","host_picture_url","host_neighbourhood","neighbourhood","neighbourhood_cleansed","neighbourhood_group_cleansed","bathrooms_text","calendar_last_scraped"]

    args = parser.parse_args()

    download_and_preprocess(args.url_file, args.output_file, drop_columns, args.download_folder)
