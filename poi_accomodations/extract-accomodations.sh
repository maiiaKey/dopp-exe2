#!/bin/bash

DATA_DIR="keys"
OUTPUT_DIR="accomodations"
mkdir -p "$OUTPUT_DIR"

# Default number of parallel tags
TAGS_AT_ONCE=1

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -tags-at-once) TAGS_AT_ONCE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if ! [[ "$TAGS_AT_ONCE" =~ ^[0-9]+$ ]]; then
    echo "Error: -tags-at-once must be a positive integer."
    exit 1
fi
TOURISM_TAGS=(
  "tourism=hotel"
  "tourism=apartment"
  "tourism=chalet"
  "tourism=guest_house"
  "tourism=hostel"
)

for FILE in "$DATA_DIR"/*.osm.pbf; do
  BASENAME=$(basename "$FILE" .osm.pbf)
  echo "Processing: $BASENAME"
    # Generate extracts for each tourism tag
    for TAG in "${TOURISM_TAGS[@]}"; do
        TAG_FILENAME=$(echo "$TAG" | sed 's/_/-/g' | sed 's/=/-/g')  # Replace underscores with hyphens for filenames
        EXTRACT_PBF="$OUTPUT_DIR/${BASENAME}-${TAG_FILENAME}.osm.pbf"
        INTERMEDIARY_PBF="$OUTPUT_DIR/${BASENAME}-${TAG_FILENAME}-intermediary.osm.pbf"
        
        # Perform the extraction and conversion
        osmium tags-filter "$FILE" "$TAG" -O -o "$INTERMEDIARY_PBF"
        osmium tags-filter "$INTERMEDIARY_PBF" /name -O -o "$EXTRACT_PBF"
        osmium export "$EXTRACT_PBF" -u type_id -O -o "$OUTPUT_DIR/${BASENAME}-${TAG_FILENAME}.geojson"

        # Clean up intermediate files
        rm "$INTERMEDIARY_PBF" "$EXTRACT_PBF"
    done
  echo "Processed: $BASENAME"
done