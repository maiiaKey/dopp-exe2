# Setup
Download all interesting osm data that you want to process. Tested on https://download.geofabrik.de/europe.html with all sub regions as files.

Python version: 3.13

Install requirements.txt

Install Osmium Tool https://osmcode.org/osmium-tool/ (tested with version 1.16.0)

# Extracting the data
Place data in a geodata folder and run the extract-data.sh and wait for it to complete (this might take a while)

Afterwards use the merge-hotels.pois.ipynb to create the final train.csv