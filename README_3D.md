# Visualisation 3D avec pg2b3dm

Ce guide explique comment utiliser la visualisation 3D intégrée dans py-fmg avec pg2b3dm et Cesium.

## 🚀 Démarrage Rapide

### 1. Génération de Carte

Assurez-vous d'avoir une carte générée :

```bash
# Démarrer les services
docker-compose up -d web db

# Générer une carte via l'API
curl -X POST "http://localhost:9888/maps/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "map_name": "Test 3D Map",
    "width": 800,
    "height": 600,
    "seed": 12345
  }'
```

### 2. Configuration de la Base de Données

Exécuter le script de configuration 3D :

```bash
# Se connecter à la base de données et exécuter le setup
docker exec -i py-fmg-db psql -U postgres -d py_fmg < setup_3d_views.sql
```

Ou utiliser le script Python :

```bash
python generate_3d_tiles.py --setup-only
```

### 3. Génération des Tuiles 3D

#### Option A : Script Bash (Recommandé)

```bash
# Générer toutes les tuiles pour la dernière carte
./generate_tiles_batch.sh

# Générer pour une carte spécifique
./generate_tiles_batch.sh "map-uuid-here"
```

#### Option B : Docker Compose

```bash
# Utiliser la configuration spécialisée 3D
docker-compose -f docker-compose-3d.yaml up

# Vérifier les tuiles générées
docker-compose -f docker-compose-3d.yaml run tile-checker
```

#### Option C : Script Python

```bash
python generate_3d_tiles.py
```

### 4. Visualisation

```bash
# Démarrer le viewer Cesium
docker-compose up cesium-viewer

# Ouvrir dans le navigateur
open http://localhost:8081
```

## 🏗️ Architecture

### Services Docker

```yaml
services:
  web: py-fmg-app (API FastAPI)
  db: PostGIS database
  pg2b3dm-*: Services de génération de tuiles par couche
  cesium-viewer: Viewer 3D Nginx + Cesium
```

### Couches 3D Disponibles

1. **Terrain** (`terrain_3d`)
   - Extrusion basée sur l'altitude
   - Couleurs par type de terrain
   - Géométrie : Polygones extrudés

2. **Établissements** (`settlements_3d`)
   - Taille selon le type (capitale > ville > village)
   - Couleurs distinctives par importance
   - Géométrie : Cylindres/cubes

3. **Rivières** (`rivers_3d`)
   - Largeur basée sur le débit
   - Élévation pour visibilité
   - Géométrie : Lignes extrudées

4. **Cultures** (`cultures_3d`)
   - Frontières culturelles en 3D
   - Couleurs par culture
   - Géométrie : Murs de frontière

5. **États** (`states_3d`)
   - Frontières politiques
   - Plus hautes que les frontières culturelles
   - Géométrie : Murs politiques

### Structure des Tuiles

```
tiles/
├── terrain/
│   ├── tileset.json
│   └── *.b3dm
├── settlements/
│   ├── tileset.json
│   └── *.b3dm
├── rivers/
│   ├── tileset.json
│   └── *.b3dm
├── cultures/
│   ├── tileset.json
│   └── *.b3dm
├── states/
│   ├── tileset.json
│   └── *.b3dm
└── generation_report.json
```

## 🎮 Contrôles du Viewer

### Interface Cesium

- **Terrain** : Afficher le relief de base
- **Settlements** : Afficher les établissements
- **Rivers** : Afficher les rivières
- **Cultures** : Afficher les frontières culturelles
- **Show All** : Afficher toutes les couches
- **Reset View** : Retour à la vue d'ensemble

### Navigation

- **Clic gauche + glisser** : Rotation de la caméra
- **Clic droit + glisser** : Panoramique
- **Molette** : Zoom avant/arrière
- **Clic milieu + glisser** : Inclinaison

## ⚙️ Configuration Avancée

### Paramètres pg2b3dm

```bash
# Erreur géométrique (niveau de détail)
-g 2000  # Plus bas = plus de détails

# Features par tuile
--max_features_per_tile 500  # Plus bas = tuiles plus petites

# Colonne de géométrie
--geometrycolumn "geom"

# Colonne d'attributs (couleur)
--attributescolumn "color"
```

### Personnalisation des Couleurs

Modifier dans `setup_3d_views.sql` :

```sql
-- Terrain
CASE 
    WHEN NOT COALESCE(is_land, false) THEN '#4169E1'  -- Bleu eau
    WHEN COALESCE(height, 0) > 100 THEN '#8B4513'     -- Brun montagnes
    WHEN COALESCE(height, 0) > 50 THEN '#228B22'      -- Vert collines
    ELSE '#90EE90'  -- Vert clair plaines
END as color

-- Établissements
CASE 
    WHEN COALESCE(is_capital, false) THEN '#FF0000'   -- Rouge capitales
    WHEN settlement_type = 'city' THEN '#FFA500'      -- Orange villes
    ELSE '#8B4513'  -- Brun villages
END as color
```

### Optimisation Performance

```sql
-- Index spatiaux pour performance
CREATE INDEX idx_terrain_3d_geom ON voronoi_cells USING GIST (geometry);
CREATE INDEX idx_settlements_3d_geom ON settlements USING GIST (geometry);
```

## 🔧 Dépannage

### Problèmes Courants

#### 1. Tuiles vides

```bash
# Vérifier les données
docker exec py-fmg-db psql -U postgres -d py_fmg -c "SELECT * FROM check_3d_setup();"

# Vérifier les géométries
docker exec py-fmg-db psql -U postgres -d py_fmg -c "SELECT COUNT(*) FROM terrain_3d WHERE geom IS NOT NULL;"
```

#### 2. Erreurs pg2b3dm

```bash
# Vérifier les logs
docker logs py-fmg-3d-terrain

# Tester la connexion DB
docker run --rm --network host geodan/pg2b3dm:latest -h localhost -U postgres -d py_fmg --test
```

#### 3. Viewer ne charge pas

```bash
# Vérifier les fichiers de tuiles
ls -la tiles/*/tileset.json

# Vérifier les permissions
chmod -R 755 tiles/
```

### Logs et Debugging

```bash
# Logs des services
docker-compose logs pg2b3dm-terrain
docker-compose logs cesium-viewer

# Vérification des tuiles
cat tiles/generation_report.json

# Test des vues 3D
docker exec py-fmg-db psql -U postgres -d py_fmg -c "
SELECT 
    'terrain' as layer, COUNT(*) as features 
FROM terrain_3d WHERE geom IS NOT NULL
UNION ALL
SELECT 
    'settlements' as layer, COUNT(*) as features 
FROM settlements_3d WHERE geom IS NOT NULL;
"
```

## 📊 Métriques de Performance

### Temps de Génération Typiques

- **Terrain** (5000 cellules) : ~2-3 minutes
- **Établissements** (50 villes) : ~30 secondes
- **Rivières** (20 rivières) : ~45 secondes
- **Cultures** (10 cultures) : ~1 minute
- **États** (5 états) : ~45 secondes

### Taille des Fichiers

- **Terrain** : 10-50 MB (selon complexité)
- **Établissements** : 1-5 MB
- **Rivières** : 2-10 MB
- **Cultures** : 1-3 MB
- **États** : 1-3 MB

### Optimisations

```bash
# Compression des tuiles
find tiles/ -name "*.b3dm" -exec gzip {} \;

# Nettoyage des tuiles anciennes
rm -rf tiles/*/
./generate_tiles_batch.sh
```

## 🔮 Fonctionnalités Avancées

### Intégration avec l'Éditeur

```python
# Régénération automatique après édition
import requests

# Modifier le terrain
edit_response = requests.post(f"/maps/{map_id}/edit/terrain", json={
    "cell_indices": [100, 101, 102],
    "operation": "set_height", 
    "value": 150.0
})

# Régénérer les tuiles terrain
if edit_response.json()["regenerate_required"]:
    subprocess.run(["./generate_tiles_batch.sh", map_id])
```

### Export pour Autres Plateformes

```bash
# Conversion pour Unity/Unreal
# (Nécessite des outils additionnels)
python convert_tiles_to_unity.py tiles/

# Export GeoJSON pour GIS
python export_to_geojson.py tiles/
```

### Streaming en Temps Réel

```javascript
// WebSocket pour mises à jour en temps réel
const ws = new WebSocket('ws://localhost:9888/ws/tiles');
ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    if (update.type === 'tileset_updated') {
        // Recharger le tileset dans Cesium
        viewer.scene.primitives.removeAll();
        loadTileset(update.layer, update.url);
    }
};
```

## 📚 Ressources

- [Documentation pg2b3dm](https://github.com/Geodan/pg2b3dm)
- [Cesium 3D Tiles](https://cesium.com/learn/cesiumjs/ref-doc/Cesium3DTileset.html)
- [PostGIS 3D Functions](https://postgis.net/docs/reference.html#PostGIS_3D_Functions)
- [3D Tiles Specification](https://github.com/CesiumGS/3d-tiles)

---

*Documentation mise à jour le 2 août 2025*

