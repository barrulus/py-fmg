from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.features import Features, Feature, LAND_COAST

# Create the same test scenario
config = GridConfig(150, 150, 300)
graph = generate_voronoi_graph(config, seed="lake_breach_test")

# Initialize with all high land
graph.heights[:] = 50

# Create ocean on left side
for i in range(len(graph.points)):
    x, y = graph.points[i]
    if x < 40:
        graph.heights[i] = 10  # Ocean

# Create low land barrier between ocean and lake
for i in range(len(graph.points)):
    x, y = graph.points[i]
    if 40 <= x <= 50 and 65 <= y <= 85:
        graph.heights[i] = 21  # Low land that can be breached

# Run initial markup
features = Features(graph)
features.markup_grid()

# Create a small lake near the ocean
lake_center_x, lake_center_y = 60, 75
lake_cells = []
for i in range(len(graph.points)):
    x, y = graph.points[i]
    dist = ((x - lake_center_x) ** 2 + (y - lake_center_y) ** 2) ** 0.5
    if dist < 10:
        lake_cells.append(i)

# Manually create the lake
lake_id = len(features.features)
for cell in lake_cells:
    graph.heights[cell] = 19  # Lake water
    features.feature_ids[cell] = lake_id
    features.distance_field[cell] = -1  # WATER_COAST

# Add lake feature
lake_feature = Feature(
    id=lake_id,
    type="lake",
    land=False,
    border=False,
    cells=len(lake_cells),
    first_cell=lake_cells[0],
)
features.features.append(lake_feature)

# Update graph attributes
graph.distance_field = features.distance_field
graph.feature_ids = features.feature_ids
graph.features = features.features

# Debug: Check if there are any LAND_COAST cells
coast_cells = []
for i in range(len(graph.points)):
    if features.distance_field[i] == LAND_COAST:
        coast_cells.append(i)

print(f"Number of LAND_COAST cells: {len(coast_cells)}")
print(f"Lake cells: {len(lake_cells)}")

# Check lake neighbors
print("\nChecking lake cell neighbors:")
checked = set()
for lake_cell in lake_cells[:5]:  # Check first 5 lake cells
    print(f"\nLake cell {lake_cell} at {graph.points[lake_cell]}:")
    for neighbor in graph.cell_neighbors[lake_cell]:
        if neighbor in checked:
            continue
        checked.add(neighbor)
        height = graph.heights[neighbor]
        dist_field = features.distance_field[neighbor]
        feature_id = features.feature_ids[neighbor]
        print(f"  Neighbor {neighbor}: height={height}, distance_field={dist_field}, feature={feature_id}")
        
        # Check if this is a potential breach point
        if features.distance_field[neighbor] == LAND_COAST and height <= 22:
            print(f"    -> Potential breach point!")
            # Check its neighbors for ocean
            for nn in graph.cell_neighbors[neighbor]:
                if features.feature_ids[nn] == 1:  # Ocean feature
                    print(f"      -> Has ocean neighbor {nn}!")

# Find ocean feature
ocean_id = None
for f in features.features:
    if f and f.type == "ocean":
        ocean_id = f.id
        print(f"\nOcean feature ID: {ocean_id}")
        break

# Try the breach
print(f"\nBefore breach: Lake type = {features.features[lake_id].type}")
features.open_near_sea_lakes(breach_limit=22)
print(f"After breach: Lake type = {features.features[lake_id].type}")