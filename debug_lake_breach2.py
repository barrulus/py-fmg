from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.features import Features, LAND_COAST

# Create the same test scenario
config = GridConfig(150, 150, 300)
graph = generate_voronoi_graph(config, seed="lake_breach_test")

# Initialize with all high land
graph.heights[:] = 50

# Create ocean on left side
ocean_cells = []
for i in range(len(graph.points)):
    x, y = graph.points[i]
    if x < 40:
        graph.heights[i] = 10  # Ocean
        ocean_cells.append(i)

# Create a small lake near the ocean
lake_center_x, lake_center_y = 60, 75
lake_cells = []
for i in range(len(graph.points)):
    x, y = graph.points[i]
    dist = ((x - lake_center_x) ** 2 + (y - lake_center_y) ** 2) ** 0.5
    if dist < 10:
        graph.heights[i] = 19  # Lake water
        lake_cells.append(i)

# Create low land barrier between ocean and lake
barrier_cells = []
for i in range(len(graph.points)):
    x, y = graph.points[i]
    if 40 <= x <= 50 and 65 <= y <= 85:
        graph.heights[i] = 21  # Low land that can be breached
        barrier_cells.append(i)

print(f"Ocean cells: {len(ocean_cells)}")
print(f"Lake cells: {len(lake_cells)}")
print(f"Barrier cells: {len(barrier_cells)}")

# Run markup after all terrain is set up
features = Features(graph)
features.markup_grid()

# Find features
ocean_id = None
lake_id = None
for f in features.features:
    if f and f.type == "ocean":
        ocean_id = f.id
        print(f"\nOcean feature: id={f.id}, cells={f.cells}")
    elif f and f.type == "lake":
        lake_id = f.id
        print(f"Lake feature: id={f.id}, cells={f.cells}")

# Check coastline cells
coast_cells = []
for i in range(len(graph.points)):
    if features.distance_field[i] == LAND_COAST:
        coast_cells.append(i)

print(f"\nTotal LAND_COAST cells: {len(coast_cells)}")

# Check if any barrier cells are marked as coast
barrier_coast = []
for bc in barrier_cells:
    if features.distance_field[bc] == LAND_COAST:
        barrier_coast.append(bc)
        print(f"Barrier cell {bc} at {graph.points[bc]} is LAND_COAST, height={graph.heights[bc]}")

print(f"\nBarrier cells marked as coast: {len(barrier_coast)}")

# Now check the exact breach logic
print("\n--- Checking breach logic ---")
ocean_ids = {f.id for f in features.features if f and f.type == "ocean"}
print(f"Ocean IDs: {ocean_ids}")

lakes_to_check = [f for f in features.features if f and f.type == "lake"]
print(f"Lakes to check: {len(lakes_to_check)}")

for lake in lakes_to_check:
    print(f"\nChecking lake {lake.id}:")
    lake_cells = [i for i, fid in enumerate(features.feature_ids) if fid == lake.id]
    print(f"  Lake has {len(lake_cells)} cells")
    
    # Check a few lake cells
    for lake_cell in lake_cells[:3]:
        print(f"\n  Lake cell {lake_cell} at {graph.points[lake_cell]}:")
        for coast_candidate in graph.cell_neighbors[lake_cell]:
            is_coast = features.distance_field[coast_candidate] == LAND_COAST
            height = graph.heights[coast_candidate]
            is_low = height <= 22
            
            print(f"    Neighbor {coast_candidate}: height={height}, is_coast={is_coast}, is_low={is_low}")
            
            if is_coast and is_low:
                print(f"      -> This is a potential breach point!")
                # Check its neighbors for ocean
                for ocean_neighbor in graph.cell_neighbors[coast_candidate]:
                    if features.feature_ids[ocean_neighbor] in ocean_ids:
                        print(f"        -> Connected to ocean cell {ocean_neighbor}!")
                        print(f"           BREACH SHOULD HAPPEN HERE!")

# Try the breach
print("\n--- Attempting breach ---")
features.open_near_sea_lakes(breach_limit=22)
print(f"Lake type after breach: {features.features[lake_id].type if lake_id else 'No lake'}")

# Focus on cell 141 - the breach point
breach_cell = 141
print(f"\n--- Detailed analysis of breach cell 141 ---")
print(f"Breach cell {breach_cell} at {graph.points[breach_cell]}:")
print(f"  Height: {graph.heights[breach_cell]}")
print(f"  Distance field: {features.distance_field[breach_cell]}")
print(f"  Feature ID: {features.feature_ids[breach_cell]}")

print(f"\nNeighbors of breach cell {breach_cell}:")
for n in graph.cell_neighbors[breach_cell]:
    print(f"  Neighbor {n} at {graph.points[n]}:")
    print(f"    Height: {graph.heights[n]}")
    print(f"    Distance field: {features.distance_field[n]}")
    print(f"    Feature ID: {features.feature_ids[n]}")
    
    # Check feature type
    fid = features.feature_ids[n]
    if fid < len(features.features) and features.features[fid]:
        print(f"    Feature type: {features.features[fid].type}")

# Check which cells are actually ocean near the breach point
print("\n--- Ocean cells near breach point ---")
ocean_near_breach = []
for i in range(len(graph.points)):
    if features.feature_ids[i] == 1:  # Ocean
        x, y = graph.points[i]
        if 35 <= x <= 55 and 65 <= y <= 85:
            ocean_near_breach.append(i)
            print(f"Ocean cell {i} at ({x:.1f}, {y:.1f})")

print(f"\nTotal ocean cells near breach: {len(ocean_near_breach)}")