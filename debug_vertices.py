from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph

# Create square island
config = GridConfig(200, 200, 400)
graph = generate_voronoi_graph(config, seed='square_island')
graph.heights[:] = 10

center_x, center_y = 100, 100
half_size = 30

for i in range(len(graph.points)):
    x, y = graph.points[i]
    if (center_x - half_size <= x <= center_x + half_size and
        center_y - half_size <= y <= center_y + half_size):
        graph.heights[i] = 40

features = Features(graph)
features.markup_grid()

graph.distance_field = features.distance_field
graph.feature_ids = features.feature_ids
graph.features = features.features

packed = regraph(graph)

# Find the island feature in original
island_id = None
for f in graph.features:
    if f and f.type == 'island':
        island_id = f.id
        print(f'Island feature ID: {island_id}')
        break

# Run markup_pack
features.markup_pack(packed)

# Find island in packed features
island_packed = None
for f in packed.features:
    if f and f.type == 'island':
        island_packed = f
        print(f'\nPacked island:')
        print(f'  ID: {f.id}')
        print(f'  Cells: {f.cells}')
        print(f'  First cell: {f.first_cell}')
        print(f'  Vertices: {f.vertices}')
        print(f'  Area: {f.area}')
        break

# Debug vertices
if island_packed and island_packed.vertices:
    print(f'\nVertex chain debug:')
    for i, v in enumerate(island_packed.vertices):
        coord = packed.vertex_coordinates[v]
        v_cells = packed.vertex_cells[v]
        v_neighbors = packed.vertex_neighbors[v]
        print(f'  [{i}] Vertex {v} at {coord}')
        print(f'      Cells: {list(v_cells)}')
        print(f'      Neighbors: {list(v_neighbors)}')
        
        # Check if this is a boundary vertex
        if len(v_neighbors) < 3:
            print(f'      *** BOUNDARY VERTEX ***')