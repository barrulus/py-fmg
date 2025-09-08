"""
Leaflet preview generator that embeds GeoJSON inline.

Generates a minimal HTML file with a Leaflet map and a single layer loaded
from an inlined GeoJSON object. Avoids external fetch so it can be opened
locally without a web server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"


def write_inline_leaflet(
    geojson: Dict[str, Any],
    out_html: str | Path,
    title: str = "Cells Preview",
    width: int = 800,
    height: int = 600,
) -> Path:
    """Write a standalone HTML file with Leaflet and inlined GeoJSON."""
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    geojson_str = json.dumps(geojson)
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{title}</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <link rel=\"stylesheet\" href=\"{LEAFLET_CSS}\" />
  <style>
    #map {{ width: {width}px; height: {height}px; }}
    body {{ margin: 0; padding: 0; }}
  </style>
  <script src=\"{LEAFLET_JS}\"></script>
</head>
<body>
  <div id=\"map\"></div>
  <script>
    const data = {geojson_str};
    // Use a simple, unprojected CRS so coordinates are treated as planar
    const map = L.map('map', {{ crs: L.CRS.Simple }});
    // Fit bounds to data if present, else default view
    let layer = L.geoJSON(data, {{
      style: function(feature) {{
        const h = feature.properties && feature.properties.height;
        // simple grayscale based on height if present
        const v = (typeof h === 'number') ? Math.max(0, Math.min(255, h)) : 128;
        const col = `rgb(${{v}}, ${{v}}, ${{v}})`;
        return {{ stroke: false, fillColor: col, fillOpacity: 0.6 }};
      }}
    }}).addTo(map);
    if (layer.getLayers().length) {{
      map.fitBounds(layer.getBounds().pad(0.05));
    }} else {{
      map.setView([0, 0], 1);
    }}
  </script>
  </body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_inline_leaflet_multi(
    layers: Dict[str, Any],
    out_html: str | Path,
    title: str = "Map Preview",
    width: int = 1200,
    height: int = 800,
) -> Path:
    """Write a Leaflet HTML with multiple inlined GeoJSON layers."""
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare inline JS objects for each layer
    layer_vars = []
    var_names = []
    for name, fc in layers.items():
        var = name.replace(" ", "_").replace("-", "_")
        var_names.append((name, var))
        layer_vars.append(f"const {var} = {json.dumps(fc)};")

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{title}</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <link rel=\"stylesheet\" href=\"{LEAFLET_CSS}\" />
  <style>
    #map {{ width: {width}px; height: {height}px; }}
    body {{ margin: 0; padding: 0; }}
  </style>
  <script src=\"{LEAFLET_JS}\"></script>
</head>
<body>
  <div id=\"map\"></div>
  <script>
    {'\n'.join(layer_vars)}
    const map = L.map('map', {{ crs: L.CRS.Simple }});

    function styleFor(name) {{
      if (name === 'coastlines') {{
        return function(_) {{ return {{ color: '#0066cc', weight: 1.5, fillOpacity: 0 }}; }}
      }}
      if (name === 'biomes') {{
        return function(f) {{
          const c = (f.properties && f.properties.color) || '#888888';
          return {{ color: '#333', weight: 0.2, fillColor: c, fillOpacity: 0.75 }};
        }}
      }}
      if (name === 'provinces') {{
        return function(f) {{
          const c = (f.properties && f.properties.color) || '#a8c0ff';
          return {{ color: '#444', weight: 0.6, fillColor: c, fillOpacity: 0.35 }};
        }}
      }}
      if (name === 'topography') {{
        return function(f) {{
          const p = f.properties || {{}};
          const c = p.color || '#cccccc';
          return {{ color: '#333', weight: 0.1, fillColor: c, fillOpacity: 0.85 }};
        }}
      }}
      if (name === 'cultures_cells') {{
        return function(f) {{
          const c = (f.properties && f.properties.color) || '#cccccc';
          return {{ color: '#333', weight: 0.2, fillColor: c, fillOpacity: 0.65 }};
        }}
      }}
      if (name === 'climate') {{
        return function(f) {{
          const t = (f.properties && f.properties.temperature) || 0;
          const v = Math.max(-30, Math.min(40, t));
          const r = Math.round(255 * (v + 30) / 70);
          const b = 255 - r;
          return {{ color: '#555', weight: 0.2, fillColor: `rgb(${{r}},120,${{b}})`, fillOpacity: 0.6 }};
        }}
      }}
      if (name === 'rivers' || name === 'rivers_smooth') {{
        return function(f) {{
          const w = (f.properties && f.properties.width) || 1.0;
          const lw = Math.max(1, Math.min(6, 1 + Math.log2(1 + w)));
          return {{ color: '#1e90ff', weight: lw, opacity: 0.9, lineCap: 'round', lineJoin: 'round' }};
        }}
      }}
      if (name === 'routes') {{
        return function(f) {{
          const cls = (f.properties && f.properties.class) || 'road';
          let color = '#8b4513';
          let dashArray = null;
          let weight = 2.0;
          if (cls === 'highway') {{ weight = 3.5; dashArray = null; }}
          else if (cls === 'road') {{ weight = 2.5; dashArray = '6 6'; }}
          else {{ weight = 2.0; dashArray = '2 6'; }}
          return {{ color, weight, dashArray, opacity: 0.9, lineCap: 'round', lineJoin: 'round' }};
        }}
      }}
      if (name === 'sea_routes') {{
        return function(f) {{
          const cls = (f.properties && f.properties.class) || 'coastal';
          let color = '#0055aa';
          let dashArray = cls === 'sea-lane' ? '10 6' : '4 8';
          let weight = cls === 'sea-lane' ? 3.0 : 2.0;
          return {{ color, weight, dashArray, opacity: 0.85, lineCap: 'round', lineJoin: 'round' }};
        }}
      }}
      if (name === 'burgs') {{
        return function(f) {{
          return {{ color: '#000', weight: 1, fillColor: '#ffcc00', fillOpacity: 0.9 }};
        }}
      }}
      if (name === 'watermask') {{
        return function(f) {{
          const p = f.properties || {{}};
          if (p.is_ocean) return {{ stroke: false, fillColor: '#69a7ff', fillOpacity: 0.6 }};
          if (p.is_lake) return {{ stroke: false, fillColor: '#8fc6ff', fillOpacity: 0.7 }};
          if (p.is_coast) return {{ stroke: false, fillColor: '#e8d9b5', fillOpacity: 0.6 }};
          return {{ stroke: false, fillColor: '#e2d3a8', fillOpacity: 0.6 }};
        }}
      }}
      return function(f) {{
        const h = f.properties && f.properties.height;
        const v = (typeof h === 'number') ? Math.max(0, Math.min(255, h)) : 128;
        const col = `rgb(${{v}}, ${{v}}, ${{v}})`;
        return {{ stroke: false, fillColor: col, fillOpacity: 0.6 }};
      }}
    }}

    function pointToLayerFor(name) {{
      if (name === 'markers') {{
        return function(feature, latlng) {{
          const p = feature.properties || {{}};
          const iconHtml = `<div style=\"font-size:${{(p.px || 16)}}px; line-height:1; transform: translate(${{p.dx||0}}px, ${{p.dy||0}}px);\">${{p.icon || '•'}}</div>`;
          const icon = L.divIcon({{ html: iconHtml, className: 'marker-emoji', iconSize: [p.px||16, p.px||16]}});
          return L.marker(latlng, {{ icon }});
        }}
      }}
      if (name === 'regiments') {{
        return function(feature, latlng) {{
          const p = feature.properties || {{}};
          const iconHtml = `<div style=\"font-size:16px; line-height:1;\">${{p.icon || '⚔️'}}</div>`;
          const icon = L.divIcon({{ html: iconHtml, className: 'regiment-emoji', iconSize: [16,16]}});
          return L.marker(latlng, {{ icon }});
        }}
      }}
      return null;
    }}

    function onEachFeatureFor(name) {{
      return function(feature, layer) {{
        const p = feature.properties || {{}};
        if (name === 'markers') {{
          const title = p.name || p.type || 'Marker';
          const legend = p.legend || '';
          layer.bindPopup(`<b>${{title}}</b><br/>${{legend}}`);
        }} else if (name === 'regiments') {{
          const title = p.name || 'Regiment';
          const total = p.total != null ? ` — ${{p.total}}` : '';
          layer.bindPopup(`<b>${{title}}${{total}}</b>`);
        }} else if (name === 'routes' || name === 'sea_routes') {{
          const cls = p.class || '';
          const dist = p.distance != null ? ` (${{(p.distance.toFixed ? p.distance.toFixed(1) : p.distance)}})` : '';
          layer.bindPopup(`<b>${{name}}</b> — ${{cls}}${{dist}}`);
        }}
      }}
    }}

    const layerObjs = {{}};
    let anyLayer = null;
    {''.join([f"layerObjs['{name}'] = L.geoJSON({var}, {{style: styleFor('{name}'), pointToLayer: pointToLayerFor('{name}'), onEachFeature: onEachFeatureFor('{name}')}}).addTo(map);\n" for name,var in var_names])}
    {''.join([f"if (!anyLayer && layerObjs['{name}'].getLayers().length) anyLayer = layerObjs['{name}'];\n" for name,_ in var_names])}
    if (anyLayer) {{ map.fitBounds(anyLayer.getBounds().pad(0.05)); }} else {{ map.setView([0,0],1); }}
    L.control.layers({{}}, layerObjs, {{ collapsed: false }}).addTo(map);
  </script>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    return out_path
