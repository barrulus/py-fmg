#!/bin/bash

curl -X POST "http://localhost:8000/maps/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "seed": "123456789",
    "width": 1600,
    "height": 800,
    "cells_desired": 50000,
    "map_name": "1k_first_new",
    "template": "volcano"
  }'