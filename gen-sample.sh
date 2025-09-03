#!/bin/bash
python cli/main.py \
  --width 1400 \
  --height 800 \
  --cells 50000 \
  --seed continents3 \
  --template continents \
  --preview-width 1400 \
  --preview-height 800 \
  --burgs-number 3000 \
  --min-river-flux 20 \
  --precip-mult 1.5 \
  --snap-to-coast-steps 5

  # --export-map out/geojson/continents.map
  # --burgs-number: target number of towns; 1000 means “auto” based on map size. Set a larger explicit number to force more towns.
  # --states-number: number of capitals/states to seed.
  # --town-spacing-base: lower this to reduce minimum spacing between towns (more towns fit).
  # --town-spacing-power: adjust spacing scaling; lower slightly to allow more towns.
  # --urbanization-rate: controls population share in towns (doesn’t change count directly, but affects sizes).
  # --min-river-flux: keep default 30.0 (you can lower to increase headwaters)
  # --precip-mult: e.g., 1.2–1.5 to increase flux and river persistence
  # --snap-to-coast-steps: default 3; increase to 4–5 if you still see inland terminations near the coast
  ## Examples

  ## More towns at 10k cells, same states:
  # --burgs-number 800 --town-spacing-base 120
  ## Many more towns and states:
  # --states-number 45 --burgs-number 1200 --town-spacing-base 120 --town-spacing-power 0.6