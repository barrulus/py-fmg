from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph

# Create a very simple test case - tiny island
config = GridConfig(100, 100, 100)
graph = generate_voronoi_graph(config, seed='simple_test')
graph.heights[:] = 10  # All water

# Find cells in center and make a small island
center_cells = []
for i in range(len(graph.points)):
    x, y = graph.points[i]
    if 45 < x < 55 and 45 < y < 55:
        center_cells.append(i)
        graph.heights[i] = 40  # Land

print(f"Island cells in original: {len(center_cells)}")

features = Features(graph)
features.markup_grid()

graph.distance_field = features.distance_field
graph.feature_ids = features.feature_ids
graph.features = features.features

# Check island feature
island_feature = None
for f in graph.features:
    if f and f.type == 'island':
        island_feature = f
        print(f"Original island: {f.cells} cells")
        break

packed = regraph(graph)
print(f"\nPacked: {len(packed.points)} cells (from {len(graph.points)})")

# Now test markup_pack
features.markup_pack(packed)

# Check packed island
for f in packed.features:
    if f and f.type == 'island':
        print(f"\nPacked island:")
        print(f"  Cells: {f.cells}")
        print(f"  Vertices: {f.vertices}")
        print(f"  Area: {f.area}")
        
        # Let's manually trace the island boundary
        print(f"\nManual boundary trace:")
        
        # Find all vertices that are on the island boundary
        boundary_vertices = set()
        for cell_id in range(len(packed.points)):
            if packed.heights[cell_id] >= 20:  # Land cell
                # Check each vertex of this cell
                for v in packed.cell_vertices[cell_id]:
                    # Check if vertex is on boundary
                    cells_at_v = packed.vertex_cells[v]
                    has_land = False
                    has_water = False
                    for c in cells_at_v:
                        if c < len(packed.points):
                            if packed.heights[c] >= 20:
                                has_land = True
                            else:
                                has_water = True
                    if has_land and has_water:
                        boundary_vertices.add(v)
        
        print(f"  Boundary vertices found: {len(boundary_vertices)}")
        print(f"  Vertices: {sorted(boundary_vertices)}")
        
        # Check connectivity of boundary vertices
        print(f"\nBoundary vertex connectivity:")
        for v in sorted(boundary_vertices)[:5]:  # First 5
            neighbors = packed.vertex_neighbors[v]
            boundary_neighbors = [n for n in neighbors if n in boundary_vertices]
            print(f"  Vertex {v}: neighbors={list(neighbors)}, boundary_neighbors={boundary_neighbors}")