#!/bin/bash

./extract-keys.sh
./extract-accomodations.sh
./extract-poi.sh -tags-at-once 12
python3 dedupe-and-unify.py ./pois/ consolidated_pois.geojson 4
python3 dedupe-and-unify.py ./accomodations/ consolidated_accomodations.geojson 4