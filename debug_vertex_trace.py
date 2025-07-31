from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph

# Create a very simple test case
config = GridConfig(200, 200, 400)
graph = generate_voronoi_graph(config, seed='square_island')
graph.heights[:] = 10  # All water

# Make a square island
for i in range(len(graph.points)):
    x, y = graph.points[i]
    if 75 < x < 125 and 75 < y < 125:
        graph.heights[i] = 40  # Land

features = Features(graph)
features.markup_grid()

graph.distance_field = features.distance_field
graph.feature_ids = features.feature_ids
graph.features = features.features

packed = regraph(graph)

# Find island cells in packed graph
island_cells = []
for i in range(len(packed.points)):
    if packed.heights[i] >= 20:
        island_cells.append(i)

print(f"Island cells in packed graph: {len(island_cells)}")
print(f"First few island cells: {island_cells[:5]}")

# Find a border cell
border_cell = None
for cell_id in island_cells:
    # Check if any neighbor is water
    for neighbor_id in packed.cell_neighbors[cell_id]:
        if packed.heights[neighbor_id] < 20:
            border_cell = cell_id
            break
    if border_cell is not None:
        break

print(f"\nBorder cell: {border_cell}")
print(f"Border cell vertices: {packed.cell_vertices[border_cell]}")

# Check each vertex of the border cell
for v in packed.cell_vertices[border_cell]:
    cells_at_v = packed.vertex_cells[v]
    print(f"\nVertex {v}:")
    print(f"  Cells: {cells_at_v}")
    print(f"  Cell heights: {[packed.heights[c] if c < len(packed.heights) else 'boundary' for c in cells_at_v]}")
    print(f"  Neighboring vertices: {packed.vertex_neighbors[v]}")
    
    # Check if this is a boundary vertex
    has_land = any(c < len(packed.heights) and packed.heights[c] >= 20 for c in cells_at_v)
    has_water = any(c < len(packed.heights) and packed.heights[c] < 20 for c in cells_at_v)
    
    if has_land and has_water:
        print(f"  -> This is a BOUNDARY vertex")
        
        # Let's trace from this vertex
        print(f"\n  Tracing from vertex {v}:")
        for i, neighbor_v in enumerate(packed.vertex_neighbors[v]):
            print(f"    Neighbor {i} (vertex {neighbor_v}):")
            neighbor_cells = packed.vertex_cells[neighbor_v]
            print(f"      Cells: {neighbor_cells}")
            
            # Check if this edge is on the boundary
            shared_cells = [c for c in cells_at_v if c in neighbor_cells]
            if len(shared_cells) == 2:
                c1_height = packed.heights[shared_cells[0]] if shared_cells[0] < len(packed.heights) else 0
                c2_height = packed.heights[shared_cells[1]] if shared_cells[1] < len(packed.heights) else 0
                c1_land = c1_height >= 20
                c2_land = c2_height >= 20
                print(f"      Shared cells: {shared_cells} (heights: {c1_height}, {c2_height})")
                if c1_land != c2_land:
                    print(f"      -> BOUNDARY EDGE!")