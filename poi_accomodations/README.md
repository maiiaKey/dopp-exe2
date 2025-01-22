# Setup
Download all interesting osm data that you want to process. Tested on https://download.geofabrik.de/europe.html with all sub regions as files.

Python version: 3.13

Install requirements.txt

Install Osmium Tool https://osmcode.org/osmium-tool/ (tested with version 1.16.0)

# Extracting the data
Place data in a geodata folder and run the extract-data.sh and wait for it to complete (this might take a while)
If you get errors when download the airbnb listings, you might need to fetch the most up to date links from https://insideairbnb.com/get-the-data/ and replace the existing ones in the airbnb-download-links.txt, they release quarterly updates also deprecating old links.

Afterwards use the process-airbnb-listings.ipynb to create the final train.csv (Note: concurrent processing might take up to 24 Gb of ram)

OR:

Download the final training dataset from https://drive.google.com/file/d/1_09yOZDQGuexwTwlmoIUDShtt-lxHLzF/view?usp=sharing