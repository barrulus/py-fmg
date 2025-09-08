#!/usr/bin/env bash

set -euo pipefail

# ------------------------------------------------------------
# py-fmg CLI switches reference (single source of truth)
# ------------------------------------------------------------
# Core map params
# --width <float>              : Map width (px) (default: 1000)
# --height <float>             : Map height (px) (default: 800)
# --cells <int>                : Target number of cells (default: 10000)
# --seed <str>                 : Random seed (string) (default: None)
# --out <path>                 : Output directory root (default: out)
# --template <str>             : Heightmap template name (default: continents)
# --target-land <0..1>         : Target land fraction; auto-shift sea level (default: None)

# Preview
# --preview [basename]         : Generate Leaflet layers preview (default size from --width/--height). Optional basename; defaults to {template}_{timestamp}. (default: disabled)
# --preview-scale <float>      : Scale factor for preview size (default: 1.0)
# --no-relax                   : Disable Lloyd relaxation (flag) (default: false)

# FMG .map export
# --export-map [path]          : Export FMG .map (optional path). Defaults to {template}_{timestamp}.map under --out when flag is present (default: disabled)
# --export-map-minimal         : Minimal .map with safe defaults (flag) (default: false)

# Hydrology
# --min-river-flux <float>     : Minimum flux to form a visible river (default: 30.0)
# --precip-mult <float>        : Multiplier for precipitation in hydrology (default: 1.0)
# --snap-to-coast-steps <int>  : Steps to extend river mouths toward ocean (default: 3; 0 = unlimited)
# --resolve-steps <int>        : Max iterations for depression resolution (default: 100)

# Climate tuning
# --equator-temp <float>       : Sea-level temperature at equator (°C) (default: None)
# --tropical-gradient <float>  : Temperature drop per degree latitude in tropics (°C/°) (default: None)
# --itcz-width <float>         : ITCZ half-width around equator (degrees) (default: None)
# --itcz-boost <float>         : ITCZ precipitation multiplier within band (default: None)

# GeoJSON export
# --geojson [basename]         : Write GeoJSON artifacts (optional basename; defaults to {template}_{timestamp}). If omitted, no GeoJSON is written (default: disabled)

# Settlements and states
# --states-number <int>        : Target number of states (capitals) (default: 30)
# --burgs-number <int>         : Target number of towns (1000 = auto) (default: 1000)
# --town-spacing-base <int>    : Base divisor for town spacing (lower = more towns) (default: 150)
# --town-spacing-power <float> : Power adjustment for town spacing (lower = more towns) (default: 0.7)
# --urbanization-rate <float>  : Urbanization rate (0..1) used for settlement sizing (default: 0.1)

# Notes:
# - Boolean flags have no value (e.g., --no-relax, --export-map-minimal).
# - Combine hydrology and climate switches to shape river patterns (e.g., ITCZ boost).

# Use module invocation to run the CLI within Poetry env
poetry run python -m cli.main \
  --width 1400 \
  --height 800 \
  --cells 50000 \
  --seed 987656789 \
  --template continents \
  --preview \
  --snap-to-coast-steps 0 \
  --burgs-number 2000 \
  --min-river-flux 35 \
  --precip-mult 0.8
  # --export-map

  # Optional example toggles
  # --target-land 0.45 \
  # --no-relax \
  # --equator-temp 23 \
  # --tropical-gradient 0.15 \
  # --itcz-width 15 \
  # --itcz-boost 1.8 \
  # --states-number 45 \
  # --town-spacing-base 120 \
  # --town-spacing-power 0.65 \
  # --urbanization-rate 0.12 \
  # --export-map  # writes to default {template}_{timestamp}.map under --out
  # --precreated europe  # use a bundled precreated heightmap instead of --template
  # --list-precreated    # list valid precreated heightmap IDs and exit
  # --parity-mode        # disable enhanced flow and coast snapping for FMG parity

# Quick tips
# - For more headwaters: lower --min-river-flux (e.g., 20) and increase --precip-mult (1.2–1.5).
# - For wetter tropics: increase --itcz-boost (1.5–2.0) and widen --itcz-width (12–20).
# - For more towns: increase --burgs-number and lower --town-spacing-base.
# - To reduce inland river endings: set --snap-to-coast-steps to a larger value or 0 (unlimited) and increase --resolve-steps.
