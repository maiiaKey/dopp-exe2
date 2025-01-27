#!/bin/bash

./extract-keys.sh
python3 download-airbnb.py airbnb-download-links.txt airbnb-listings.csv airbnbdata
./extract-poi.sh -tags-at-once 12
python3 consolidate_geojson.py ./pois/ consolidated_pois.geojson 4