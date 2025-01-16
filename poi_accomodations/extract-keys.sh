#!/bin/bash
DATA_DIR="geodata"
OUTPUT_DIR="keys"
mkdir -p "$OUTPUT_DIR"

for FILE in "$DATA_DIR"/*.osm.pbf; do
  BASENAME=$(basename "$FILE" .osm.pbf)
  KEYS_EXTRACT_PBF="$OUTPUT_DIR/${BASENAME}-keys.osm.pbf"
  echo "Processing: $BASENAME"
  osmium tags-filter "$FILE" -O building=commercial,industrial,office,retail,supermarket,religious,cathedral,bakehouse,bridge,civic,government,museum,public,train_station,university,barn,conservatory,stable,bakehouse,grandstand,stadium,bunker,castle,military,pagoda,ruins,tower,windmill amenity=restaurant,bar,pub,cafe,surf_school,university,boat_rental,bus_station,bicycle_rental,car_rental,charging_station,taxi,atm,bank,casino,cinema,conference_centre,event_venue,exhibition_centre,fountain,music_venue,nightclub,stage,theatre,police,post_box,post_office,recycling,clock,funeral_hall,grave_yard,internet_cafe,monastery,place_of_worship,townhall,refugee_site tourism=aquarium,artwork,attraction,gallery,museum,theme_park,viewpoint,zoo,hotel,apartment,chalet,guest_house,hostel historic=aircraft,anchor,aqueduct,archaeological_site,battlefield,building,cannon,castle,church,city_gate,creamery,farm,fort,gallows,house,high_cross,locomotive,machine,manor,memorial,mine,monastery,monument,ruins,ship,tank,temple,tomb,tower,wreck water=lake craft=atelier,beekeeper,blacksmith,brewery,clockmaker,confectionery,distillery,glassblower,goldsmith,handicraft,jeweller,mint,pottery,shoemaker,tailor,watchmaker,winery waterway=dam,waterfall aerialway=station public_transport=station shop=gift railway=subway_entrance sport=motor aeroway=terminal geological=fault,fold,volcanic_vent,meteor_crater,columnar_jointing,tor,sinkhole natural=bay,beach,geyser,glacier,hot_spring,arch,peak,valley,volcano -o "$KEYS_EXTRACT_PBF"
  echo "Processed: $BASENAME"
done