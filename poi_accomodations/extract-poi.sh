#!/bin/bash

DATA_DIR="keys"
OUTPUT_DIR="pois2"
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

POI_TAGS=(
    "aerialway=station"
    "aeroway=terminal"
    "amenity=restaurant"
    "amenity=bar"
    "amenity=pub"
    "amenity=cafe"
    "amenity=surf_school"
    "amenity=university"
    "amenity=boat_rental"
    "amenity=bus_station"
    "amenity=bicycle_rental"
    "amenity=car_rental"
    "amenity=charging_station"
    "amenity=taxi"
    "amenity=atm"
    "amenity=bank"
    "amenity=casino"
    "amenity=cinema"
    "amenity=conference_centre"
    "amenity=event_venue"
    "amenity=exhibition_centre"
    "amenity=fountain"
    "amenity=music_venue"
    "amenity=nightclub"
    "amenity=stage"
    "amenity=theatre"
    "amenity=police"
    "amenity=post_box"
    "amenity=post_office"
    "amenity=recycling"
    "amenity=clock"
    "amenity=funeral_hall"
    "amenity=grave_yard"
    "amenity=internet_cafe"
    "amenity=monastery"
    "amenity=place_of_worship"
    "amenity=townhall"
    "amenity=refugee_site"
    "building=commercial"
    "building=industrial"
    "building=office"
    "building=retail"
    "building=supermarket"
    "building=religious"
    "building=cathedral"
    "building=bakehouse"
    "building=bridge"
    "building=civic"
    "building=government"
    "building=museum"
    "building=public"
    "building=train_station"
    "building=university"
    "building=barn"
    "building=conservatory"
    "building=stable"
    "building=bakehouse"
    "building=grandstand"
    "building=stadium"
    "building=bunker"
    "building=castle"
    "building=military"
    "building=pagoda"
    "building=ruins"
    "building=tower"
    "building=windmill"
    "craft=atelier"
    "craft=beekeeper"
    "craft=blacksmith"
    "craft=brewery"
    "craft=clockmaker"
    "craft=confectionery"
    "craft=distillery"
    "craft=glassblower"
    "craft=goldsmith"
    "craft=handicraft"
    "craft=jeweller"
    "craft=mint"
    "craft=pottery"
    "craft=shoemaker"
    "craft=tailor"
    "craft=watchmaker"
    "craft=winery"
    "geological=fault"
    "geological=fold"
    "geological=volcanic_vent"
    "geological=meteor_crater"
    "geological=columnar_jointing"
    "geological=tor"
    "geological=sinkhole"
    "historic=aircraft"
    "historic=anchor"
    "historic=aqueduct"
    "historic=archaeological_site"
    "historic=battlefield"
    "historic=building"
    "historic=cannon"
    "historic=castle"
    "historic=church"
    "historic=city_gate"
    "historic=creamery"
    "historic=farm"
    "historic=fort"
    "historic=gallows"
    "historic=house"
    "historic=high_cross"
    "historic=locomotive"
    "historic=machine"
    "historic=manor"
    "historic=memorial"
    "historic=mine"
    "historic=monastery"
    "historic=monument"
    "historic=ruins"
    "historic=ship"
    "historic=tank"
    "historic=temple"
    "historic=tomb"
    "historic=tower"
    "historic=wreck"
    "natural=bay"
    "natural=beach"
    "natural=geyser"
    "natural=glacier"
    "natural=hot_spring"
    "natural=arch"
    "natural=peak"
    "natural=valley"
    "natural=volcano"
    "public_transport=station"
    "railway=subway_entrance"
    "shop=gift"
    "sport=motor"
    "tourism=aquarium"
    "tourism=artwork"
    "tourism=attraction"
    "tourism=gallery"
    "tourism=museum"
    "tourism=theme_park"
    "tourism=viewpoint"
    "tourism=zoo"
    "water=lake"
    "waterway=dam"
    "waterway=waterfall"
  )

process_tag() {
    local FILE="$1"
    local TAG="$2"
    local BASENAME="$3"

    TAG_FILENAME=$(echo "$TAG" | sed 's/_/-/g' | sed 's/=/-/g')
    EXTRACT_PBF="$OUTPUT_DIR/${BASENAME}-${TAG_FILENAME}.osm.pbf"
    INTERMEDIARY_PBF="$OUTPUT_DIR/${BASENAME}-${TAG_FILENAME}-intermediary.osm.pbf"

    osmium tags-filter "$FILE" "$TAG" --no-progress -O -o "$INTERMEDIARY_PBF"
    osmium tags-filter "$INTERMEDIARY_PBF" /name -O -o "$EXTRACT_PBF"
    osmium export "$EXTRACT_PBF" -u type_id --no-progress -O -o "$OUTPUT_DIR/${BASENAME}-${TAG_FILENAME}.geojson"

    rm "$INTERMEDIARY_PBF" "$EXTRACT_PBF"
}

export -f process_tag  # Export function for parallel use in xargs
export OUTPUT_DIR

for FILE in "$DATA_DIR"/*.osm.pbf; do
    BASENAME=$(basename "$FILE" .osm.pbf)
    echo "Processing: $BASENAME"

    # Use xargs for parallel processing
    printf "%s\n" "${POI_TAGS[@]}" | xargs -I{} -P"$TAGS_AT_ONCE" bash -c 'process_tag "$0" "$1" "$2"' "$FILE" "{}" "$BASENAME"

    echo "Processed: $BASENAME"
done