#!/bin/bash

./extract-keys.sh
python download-airbnb.py airbnb-download-links.txt airbnb-listings.csv airbnbdata
./extract-poi.sh -tags-at-once 12
python3 dedupe-and-unify.py ./pois/ consolidated_pois.geojson 4