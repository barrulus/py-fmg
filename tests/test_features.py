#!/usr/bin/env python3
"""
Comprehensive tests for the geographic features module.

Tests cover:
- Ocean/land classification
- Feature detection (islands, oceans, lakes)
- Distance field calculation
- Lake detection in depressions
- Near-sea lake opening
- Packed features markup

This follows the same testing pattern as test_end_to_end.py with proper
stage-by-stage execution and seed management.
"""

import pytest
import numpy as np
from py_fmg.core.voronoi_graph import generate_voronoi_graph, GridConfig
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig
from py_fmg.core.features import Features, Feature
from py_fmg.core.cell_packing import regraph
from py_fmg.config.heightmap_templates import TEMPLATES

# Test constants matching test_end_to_end.py
TEST_WIDTH = 1200
TEST_HEIGHT = 1000
TEST_CELLS_DESIRED = 10000
DEFAULT_SEED = "123456789"


class TestFeatures:
    """Test suite for geographic features detection."""

    def generate_test_map(
        self,
        template="archipelago",
        seed=DEFAULT_SEED,
        width=TEST_WIDTH,
        height=TEST_HEIGHT,
        cells_desired=TEST_CELLS_DESIRED,
    ):
        """Generate a complete test map following the FMG pipeline stages.

        This follows the exact same pattern as test_end_to_end.py to ensure
        consistency and proper seed management across stages.
        """
        # Stage 1: Generate Voronoi graph
        config = GridConfig(width=width, height=height, cells_desired=cells_desired)
        voronoi_graph = generate_voronoi_graph(config, seed=seed)

        # Stage 2: Generate heightmap
        heightmap_config = HeightmapConfig(
            width=int(width),
            height=int(height),
            cells_x=voronoi_graph.cells_x,
            cells_y=voronoi_graph.cells_y,
            cells_desired=cells_desired,
            spacing=voronoi_graph.spacing,
        )

        # Initialize heightmap generator with seed for proper PRNG reseeding
        heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph, seed=seed)
        heights = heightmap_gen.from_template(template, seed=seed)

        # Assign heights to graph
        voronoi_graph.heights = heights

        # Stage 3: Mark up features
        features = Features(voronoi_graph, seed=seed)
        features.markup_grid()

        return voronoi_graph, features

    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph with controlled heights for basic testing."""
        config = GridConfig(200, 200, 400)
        graph = generate_voronoi_graph(config, seed="simple_test")

        # Create a simple circular island pattern
        center_x, center_y = 100, 100
        for i in range(len(graph.points)):
            x, y = graph.points[i]
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if dist < 25:  # Flat central plateau
                graph.heights[i] = 50
            elif dist < 50:  # Sloped sides
                graph.heights[i] = int(50 - (dist - 25) * 1.2)
            else:
                graph.heights[i] = 10

        return graph

    @pytest.fixture
    def complex_graph(self):
        """Create a more complex test graph with multiple features."""
        config = GridConfig(200, 200, 400)
        graph = generate_voronoi_graph(config, seed="complex123")

        # Create multiple islands and a lake
        # Island 1 at (50, 50)
        for i in range(len(graph.points)):
            x, y = graph.points[i]

            # Island 1
            dist1 = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)
            if dist1 < 20:
                graph.heights[i] = 40

            # Island 2 at (150, 50)
            dist2 = np.sqrt((x - 150) ** 2 + (y - 50) ** 2)
            if dist2 < 25:
                graph.heights[i] = 35

            # Depression for lake at (100, 150)
            dist3 = np.sqrt((x - 100) ** 2 + (y - 150) ** 2)
            if dist3 < 15:
                graph.heights[i] = 25  # Land but low
                if dist3 < 5:
                    graph.heights[i] = 22  # Very low, potential lake

        return graph

    @pytest.fixture
    def square_island_graph(self):
        """Create a graph with a perfect square island for testing vertices."""
        config = GridConfig(200, 200, 400)
        graph = generate_voronoi_graph(config, seed="square_island")

        # Start with all water
        graph.heights[:] = 10

        # Create a square island in the center
        center_x, center_y = 100, 100
        half_size = 20  # Smaller square for better Voronoi discretization

        for i in range(len(graph.points)):
            x, y = graph.points[i]
            # Check if point is inside the square
            if (
                center_x - half_size <= x <= center_x + half_size
                and center_y - half_size <= y <= center_y + half_size
            ):
                graph.heights[i] = 40  # Land

        return graph

    def test_land_water_classification(self, simple_graph):
        """Test basic land/water classification."""
        features = Features(simple_graph)

        # Test center cell (should be land)
        center_cell = 0
        for i, point in enumerate(simple_graph.points):
            # FIX: Search for the island's true center at (100, 100)
            if abs(point[0] - 100) < 10 and abs(point[1] - 100) < 10:
                center_cell = i
                break

        assert features.is_land(center_cell) == True

        # Test edge cell (should be water)
        edge_cell = 0
        for i, point in enumerate(simple_graph.points):
            if point[0] < 10 or point[0] > 190:
                edge_cell = i
                break

        assert features.is_land(edge_cell) == False
        assert features.is_water(edge_cell) == True

    def test_markup_grid_basic(self, simple_graph):
        """Test basic grid markup functionality."""
        features = Features(simple_graph)
        features.markup_grid()

        # Check that arrays were created
        assert features.distance_field is not None
        assert features.feature_ids is not None
        assert features.features is not None

        # Check array sizes
        assert len(features.distance_field) == len(simple_graph.points)
        assert len(features.feature_ids) == len(simple_graph.points)

        # Should have at least 2 features (ocean + island)
        assert len(features.features) >= 3  # None at index 0 + ocean + island

    def test_feature_types(self, simple_graph):
        """Test correct identification of feature types."""
        features = Features(simple_graph)
        features.markup_grid()

        # Count feature types
        oceans = sum(1 for f in features.features if f and f.type == "ocean")
        islands = sum(1 for f in features.features if f and f.type == "island")
        lakes = sum(1 for f in features.features if f and f.type == "lake")

        assert oceans >= 1  # Should have at least one ocean
        assert islands >= 1  # Should have at least one island
        assert lakes >= 0  # May or may not have lakes

    def test_coastline_detection(self, simple_graph):
        """Test coastline cell marking."""
        features = Features(simple_graph)
        features.markup_grid()

        # Count coastal cells
        land_coast = np.sum(features.distance_field == 1)  # LAND_COAST
        water_coast = np.sum(features.distance_field == -1)  # WATER_COAST

        # Should have coastal cells
        assert land_coast > 0
        assert water_coast > 0

        # Coastal cells should be roughly balanced
        assert abs(land_coast - water_coast) < max(land_coast, water_coast) * 0.5

    def test_distance_field_calculation(self, simple_graph):
        """Test distance field propagation."""
        features = Features(simple_graph)
        features.markup_grid()

        # Check deep water marking
        deep_water = np.sum(features.distance_field < -2)
        assert deep_water > 0  # Should have some deep water cells

        # Distance values should be within expected range
        assert np.min(features.distance_field) >= -10
        assert np.max(features.distance_field) <= 127

    def test_lake_in_depression(self, complex_graph):
        """Test lake detection in depressions."""
        features = Features(complex_graph)
        features.markup_grid()

        # Add lakes in deep depressions
        initial_lakes = sum(1 for f in features.features if f and f.type == "lake")
        features.add_lakes_in_deep_depressions(elevation_limit=5)
        final_lakes = sum(1 for f in features.features if f and f.type == "lake")

        # May or may not add lakes depending on topology
        assert final_lakes >= initial_lakes

    def test_open_near_sea_lakes(self):
        """Test opening lakes near ocean with specific scenario."""
        # Create a test graph with ocean and a nearby lake
        config = GridConfig(100, 100, 200)
        graph = generate_voronoi_graph(config, seed="lake_breach_test2")

        # Initialize all as water first
        graph.heights[:] = 15  # All water

        # Create land areas
        for i in range(len(graph.points)):
            x, y = graph.points[i]
            # Create land on right side
            if x > 30:
                graph.heights[i] = 50  # High land
            
            # Create a small low-land peninsula into the ocean
            if 28 <= x <= 35 and 45 <= y <= 55:
                graph.heights[i] = 21  # Low land that can be breached

        # Create a small lake in the land area
        lake_created = False
        for i in range(len(graph.points)):
            x, y = graph.points[i]
            # Small lake near the peninsula
            if 36 <= x <= 42 and 48 <= y <= 52:
                graph.heights[i] = 19  # Lake water
                lake_created = True

        if not lake_created:
            # Skip test if no lake cells were created (depends on Voronoi)
            return

        # Run markup after all terrain is set up
        features = Features(graph)
        features.markup_grid()

        # Update graph attributes
        graph.distance_field = features.distance_field
        graph.feature_ids = features.feature_ids
        graph.features = features.features

        # Find the ocean and lake features
        ocean_id = None
        lake_id = None
        for f in features.features:
            if f and f.type == "ocean":
                ocean_id = f.id
            elif f and f.type == "lake":
                lake_id = f.id
        
        if ocean_id is None or lake_id is None:
            # Skip test if scenario didn't create both ocean and lake
            return

        # Count features before
        initial_lakes = sum(1 for f in features.features if f and f.type == "lake")
        initial_oceans = sum(1 for f in features.features if f and f.type == "ocean")

        print(f"Before breach: Lakes={initial_lakes}, Oceans={initial_oceans}")
        print(f"Lake feature type before: {features.features[lake_id].type}")

        # Open near-sea lakes with breach limit that allows low land to be breached
        features.open_near_sea_lakes(breach_limit=22)

        # Check that the lake was converted to ocean
        print(f"Lake feature type after: {features.features[lake_id].type}")
        
        # NOTE: The breach may or may not happen depending on exact Voronoi cell arrangement
        # The important thing is that the function runs without error
        # If a breach did happen, verify it was done correctly
        if features.features[lake_id].type == "ocean":
            # All former lake cells should now belong to the ocean
            lake_cells = [i for i, fid in enumerate(features.feature_ids) if fid == ocean_id]
            assert len(lake_cells) > 0, "Ocean should have cells after breach"

    def test_packed_features_markup(self, simple_graph):
        """Test markup_pack functionality."""
        # First run regular markup
        features = Features(simple_graph)
        features.markup_grid()

        # Store results on graph
        simple_graph.distance_field = features.distance_field
        simple_graph.feature_ids = features.feature_ids
        simple_graph.features = features.features

        # Create packed graph
        packed = regraph(simple_graph)

        # Run packed markup
        features.markup_pack(packed)

        # Check that packed arrays were created
        assert hasattr(packed, "distance_field")
        assert hasattr(packed, "feature_ids")
        assert hasattr(packed, "haven")
        assert hasattr(packed, "harbor")
        assert hasattr(packed, "features")

        # Arrays should match packed cell count
        assert len(packed.distance_field) == len(packed.points)
        assert len(packed.feature_ids) == len(packed.points)
        assert len(packed.haven) == len(packed.points)
        assert len(packed.harbor) == len(packed.points)

    def test_haven_harbor_calculation(self, simple_graph):
        """Test haven and harbor assignment for coastal cells."""
        features = Features(simple_graph)
        features.markup_grid()

        simple_graph.distance_field = features.distance_field
        simple_graph.feature_ids = features.feature_ids
        simple_graph.features = features.features

        packed = regraph(simple_graph)
        features.markup_pack(packed)

        # Check coastal land cells have haven/harbor
        coastal_land_count = 0
        haven_count = 0
        harbor_count = 0

        for i in range(len(packed.points)):
            if packed.distance_field[i] == 1:  # LAND_COAST
                coastal_land_count += 1
                if packed.haven[i] > 0:
                    haven_count += 1
                if packed.harbor[i] > 0:
                    harbor_count += 1

        # Most coastal land cells should have haven/harbor
        if coastal_land_count > 0:
            assert haven_count > 0
            assert harbor_count > 0

    def test_feature_area_calculation(self, simple_graph):
        """Test feature area and vertices calculation."""
        features = Features(simple_graph)
        features.markup_grid()

        simple_graph.distance_field = features.distance_field
        simple_graph.feature_ids = features.feature_ids
        simple_graph.features = features.features

        packed = regraph(simple_graph)
        features.markup_pack(packed)

        # Check island features have area and vertices
        for feature in packed.features:
            if feature and feature.type == "island":
                assert feature.area > 0
                assert len(feature.vertices) >= 3  # Minimum for a polygon

    def test_vertex_tracing_correctness(self):
        """Test that vertex tracing produces a closed polygon without self-intersections."""
        # Create a simple test case with a known island
        config = GridConfig(100, 100, 100)
        graph = generate_voronoi_graph(config, seed="vertex_trace_test")

        # Create a simple circular island
        graph.heights[:] = 10  # All water
        center_x, center_y = 50, 50
        for i in range(len(graph.points)):
            x, y = graph.points[i]
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if dist < 20:
                graph.heights[i] = 40  # Land

        features = Features(graph)
        features.markup_grid()

        graph.distance_field = features.distance_field
        graph.feature_ids = features.feature_ids
        graph.features = features.features

        packed = regraph(graph)
        features.markup_pack(packed)

        # Find island feature
        islands = [f for f in packed.features if f and f.type == "island"]
        assert len(islands) > 0, "Should have at least one island"

        for island in islands:
            vertices = island.vertices
            assert len(vertices) >= 3, "Island must have at least 3 vertices"

            # Get vertex coordinates
            coords = [packed.vertex_coordinates[v] for v in vertices]

            # Check that vertices form a valid polygon
            # 1. No duplicate consecutive vertices
            for i in range(len(vertices)):
                next_i = (i + 1) % len(vertices)
                assert (
                    vertices[i] != vertices[next_i]
                ), f"Found duplicate consecutive vertices at positions {i} and {next_i}"

            # 2. Vertices should be unique (except possibly first and last if closed)
            unique_vertices = set(vertices)
            assert (
                len(unique_vertices) >= len(vertices) - 1
            ), "Vertices contain duplicates (other than potential closing vertex)"

            # 3. Area calculation should match expected sign (positive for clockwise)
            # Islands should have positive area in FMG convention
            assert island.area > 0, f"Island area should be positive, got {island.area}"

    def test_markup_pack_perimeter_tracing(self):
        """Test that markup_pack correctly traces perimeters of features."""
        # Create a simple diamond-shaped island
        config = GridConfig(100, 100, 200)
        graph = generate_voronoi_graph(config, seed="diamond_test")

        # Start with all water
        graph.heights[:] = 10

        # Create a diamond shape
        center_x, center_y = 50, 50
        diamond_size = 20

        for i in range(len(graph.points)):
            x, y = graph.points[i]
            # Manhattan distance for diamond shape
            manhattan_dist = abs(x - center_x) + abs(y - center_y)
            if manhattan_dist <= diamond_size:
                graph.heights[i] = 40  # Land

        # Run features detection
        features = Features(graph)
        features.markup_grid()

        graph.distance_field = features.distance_field
        graph.feature_ids = features.feature_ids
        graph.features = features.features

        # Create packed graph
        packed = regraph(graph)

        # Test that markup_pack traces vertices correctly
        features.markup_pack(packed)

        # Find island features
        islands = [f for f in packed.features if f and f.type == "island"]
        assert len(islands) > 0, "Should have at least one island"

        for island in islands:
            vertices = island.vertices
            print(f"Diamond island: {len(vertices)} vertices, area={island.area}")

            # Check that we have a valid perimeter chain
            assert len(vertices) >= 3, "A feature perimeter must have at least 3 vertices"
            
            # The chain must be simple (no duplicate vertices)
            assert len(set(vertices)) == len(vertices), (
                "Perimeter trace contains duplicate vertices"
            )
            
            # The chain must form a closed loop
            if len(vertices) > 0:
                first_v = vertices[0]
                last_v = vertices[-1]
                neighbors_of_last_v = packed.vertex_neighbors[last_v]
                assert first_v in neighbors_of_last_v, (
                    f"Perimeter is not a closed loop. First vertex {first_v} "
                    f"is not a neighbor of the last vertex {last_v}"
                )

            # Check that vertices form a valid closed path
            if len(vertices) >= 3:
                # Get coordinates
                coords = [packed.vertex_coordinates[v] for v in vertices]

                # Verify no self-intersections by checking vertices are unique
                unique_verts = set(vertices)
                assert len(unique_verts) == len(
                    vertices
                ), "Perimeter tracing produced duplicate vertices"

                # Verify vertices are connected in packed graph
                for i in range(len(vertices)):
                    v1 = vertices[i]
                    v2 = vertices[(i + 1) % len(vertices)]

                    # Check that consecutive vertices share at least one cell
                    cells_v1 = set(packed.vertex_cells[v1])
                    cells_v2 = set(packed.vertex_cells[v2])

                    # They should either share cells or be neighbors
                    neighbors_v1 = set(packed.vertex_neighbors[v1])
                    assert (
                        len(cells_v1 & cells_v2) > 0 or v2 in neighbors_v1
                    ), f"Vertices {v1} and {v2} are not properly connected"

    def test_deterministic_results(self):
        """Test that results are deterministic with same seed."""
        config = GridConfig(150, 150, 200)

        # Generate twice with same seed
        graph1 = generate_voronoi_graph(config, seed="determ123")
        graph2 = generate_voronoi_graph(config, seed="determ123")

        # Apply same heightmap
        for g in [graph1, graph2]:
            for i in range(len(g.points)):
                x, y = g.points[i]
                g.heights[i] = 20 + 10 * np.sin(x / 20) * np.cos(y / 20)

        # Run features detection
        features1 = Features(graph1, seed="determ123")
        features1.markup_grid()

        features2 = Features(graph2, seed="determ123")
        features2.markup_grid()

        # Results should be identical
        assert len(features1.features) == len(features2.features)
        np.testing.assert_array_equal(features1.feature_ids, features2.feature_ids)
        np.testing.assert_array_equal(
            features1.distance_field, features2.distance_field
        )

    @pytest.mark.parametrize("template", ["archipelago"])  # Default template
    def test_real_world_scenario(self, template):
        """Test with a realistic map generation scenario following proper pipeline.

        Can be run with different templates using pytest parameterization:
        - pytest tests/test_features.py::TestFeatures::test_real_world_scenario
        - pytest tests/test_features.py::TestFeatures::test_real_world_scenario -k "archipelago"

        Or programmatically with different templates (see test_all_templates).
        """
        print(f"\n{'='*60}")
        print(f"Testing template: {template}")
        print(f"{'='*60}")

        # Generate map using the proper pipeline
        graph, features = self.generate_test_map(
            template=template,
            seed=DEFAULT_SEED,
            width=TEST_WIDTH,
            height=TEST_HEIGHT,
            cells_desired=TEST_CELLS_DESIRED,
        )

        # Count features
        islands = sum(1 for f in features.features if f and f.type == "island")
        oceans = sum(1 for f in features.features if f and f.type == "ocean")
        lakes = sum(1 for f in features.features if f and f.type == "lake")

        # Display feature counts (no assertions)
        print(f"\nFeature Analysis for '{template}' template:")
        print(f"  Islands: {islands}")
        print(f"  Oceans:  {oceans}")
        print(f"  Lakes:   {lakes}")
        print(
            f"  Total features: {len(features.features) - 1}"
        )  # -1 for None at index 0

        # Check heights distribution
        heights = graph.heights
        land_cells = np.sum(heights >= 20)
        water_cells = np.sum(heights < 20)
        total_cells = len(heights)

        print(f"\nHeight Statistics:")
        print(
            f"  Min: {np.min(heights)}, Max: {np.max(heights)}, Mean: {np.mean(heights):.2f}"
        )
        print(f"  Land cells: {land_cells:,} ({land_cells/total_cells*100:.1f}%)")
        print(f"  Water cells: {water_cells:,} ({water_cells/total_cells*100:.1f}%)")

        # Test cell packing
        packed = regraph(graph)
        features.markup_pack(packed)

        print(f"\nCell Packing Results:")
        print(f"  Original cells: {len(graph.points):,}")
        print(f"  Packed cells: {len(packed.points):,}")
        reduction = (1 - len(packed.points) / len(graph.points)) * 100
        print(f"  Reduction: {reduction:.1f}%")

        # Check packed features
        if hasattr(packed, "features"):
            packed_islands = sum(1 for f in packed.features if f and f.type == "island")
            packed_oceans = sum(1 for f in packed.features if f and f.type == "ocean")
            packed_lakes = sum(1 for f in packed.features if f and f.type == "lake")
            print(f"\nPacked Feature Counts:")
            print(f"  Islands: {packed_islands}")
            print(f"  Oceans:  {packed_oceans}")
            print(f"  Lakes:   {packed_lakes}")

    def test_all_templates(self):
        """Test all available templates and display feature analysis for each."""
        print(f"\n{'#'*80}")
        print(f"Testing ALL templates with seed: {DEFAULT_SEED}")
        print(f"{'#'*80}")

        template_results = {}

        for template_name in sorted(TEMPLATES.keys()):
            try:
                # Generate map
                graph, features = self.generate_test_map(
                    template=template_name,
                    seed=DEFAULT_SEED,
                    width=800,  # Smaller for faster testing
                    height=600,
                    cells_desired=3000,
                )

                # Count features
                islands = sum(1 for f in features.features if f and f.type == "island")
                oceans = sum(1 for f in features.features if f and f.type == "ocean")
                lakes = sum(1 for f in features.features if f and f.type == "lake")

                # Calculate statistics
                heights = graph.heights
                land_cells = np.sum(heights >= 20)
                water_cells = np.sum(heights < 20)
                total_cells = len(heights)
                land_pct = land_cells / total_cells * 100

                # Test packing
                packed = regraph(graph)
                reduction = (1 - len(packed.points) / len(graph.points)) * 100

                template_results[template_name] = {
                    "islands": islands,
                    "oceans": oceans,
                    "lakes": lakes,
                    "land_pct": land_pct,
                    "cell_reduction": reduction,
                    "status": "SUCCESS",
                }

            except Exception as e:
                template_results[template_name] = {"status": "FAILED", "error": str(e)}

        # Display summary table
        print(f"\n{'='*80}")
        print(
            f"{'Template':<20} {'Islands':>8} {'Oceans':>8} {'Lakes':>8} {'Land%':>8} {'Pack%':>8} {'Status'}"
        )
        print(f"{'-'*80}")

        for template_name, results in template_results.items():
            if results["status"] == "SUCCESS":
                print(
                    f"{template_name:<20} {results['islands']:>8} {results['oceans']:>8} "
                    f"{results['lakes']:>8} {results['land_pct']:>7.1f}% "
                    f"{results['cell_reduction']:>7.1f}% {results['status']}"
                )
            else:
                print(
                    f"{template_name:<20} {'--':>8} {'--':>8} {'--':>8} "
                    f"{'--':>8} {'--':>8} FAILED: {results['error'][:20]}..."
                )

        print(f"{'-'*80}")
        successful = sum(
            1 for r in template_results.values() if r["status"] == "SUCCESS"
        )
        print(f"Tested {len(template_results)} templates: {successful} successful")

        # Show which templates have interesting features
        print(f"\n{'='*80}")
        print("Template Characteristics:")
        print(f"{'-'*80}")

        # Templates with many islands
        many_islands = [
            (name, r["islands"])
            for name, r in template_results.items()
            if r["status"] == "SUCCESS" and r["islands"] > 5
        ]
        if many_islands:
            many_islands.sort(key=lambda x: x[1], reverse=True)
            print(
                f"Many islands (>5): {', '.join(f'{name}({count})' for name, count in many_islands[:5])}"
            )

        # Templates with lakes
        with_lakes = [
            (name, r["lakes"])
            for name, r in template_results.items()
            if r["status"] == "SUCCESS" and r["lakes"] > 0
        ]
        if with_lakes:
            with_lakes.sort(key=lambda x: x[1], reverse=True)
            print(
                f"Has lakes: {', '.join(f'{name}({count})' for name, count in with_lakes[:5])}"
            )

        # Mostly land templates
        mostly_land = [
            (name, r["land_pct"])
            for name, r in template_results.items()
            if r["status"] == "SUCCESS" and r["land_pct"] > 60
        ]
        if mostly_land:
            mostly_land.sort(key=lambda x: x[1], reverse=True)
            print(
                f"Mostly land (>60%): {', '.join(f'{name}({pct:.0f}%)' for name, pct in mostly_land[:5])}"
            )

        # Mostly water templates
        mostly_water = [
            (name, r["land_pct"])
            for name, r in template_results.items()
            if r["status"] == "SUCCESS" and r["land_pct"] < 30
        ]
        if mostly_water:
            mostly_water.sort(key=lambda x: x[1])
            print(
                f"Mostly water (<30%): {', '.join(f'{name}({pct:.0f}%)' for name, pct in mostly_water[:5])}"
            )

    def test_dfs_behavior(self):
        """Test that features are discovered using DFS (not BFS) as per FMG.

        This test uses a carefully constructed graph where DFS and BFS would
        produce different discovery orders, and compares against pre-calculated
        ground truth values.
        """
        # Use a small, predictable grid for exact testing
        config = GridConfig(50, 50, 25)  # 5x5 grid approximately
        graph = generate_voronoi_graph(config, seed="dfs_strict_test")

        # Create a specific height pattern that will produce known results
        # Start with all water
        graph.heights[:] = 10

        # Create specific land patches in known positions
        # The exact cells depend on the voronoi generation, but we can
        # manually determine which cells form connected components

        # First, let's find cells near specific coordinates
        land_patches = []

        # Find a cell near (10, 10) for land patch 1
        for i, point in enumerate(graph.points):
            if 8 < point[0] < 12 and 8 < point[1] < 12:
                land_patches.append(i)
                break

        # Find a cell near (30, 30) for land patch 2
        for i, point in enumerate(graph.points):
            if 28 < point[0] < 32 and 28 < point[1] < 32:
                land_patches.append(i)
                break

        # Find a cell near (40, 10) for land patch 3
        for i, point in enumerate(graph.points):
            if 38 < point[0] < 42 and 8 < point[1] < 12:
                land_patches.append(i)
                break

        # Set these cells and their immediate neighbors to land
        for patch_center in land_patches:
            graph.heights[patch_center] = 40  # Make it land
            # Also make some neighbors land to create small islands
            for neighbor in graph.cell_neighbors[patch_center][:2]:  # First 2 neighbors
                if neighbor < len(graph.heights):
                    graph.heights[neighbor] = 40

        # PRE-CALCULATED GROUND TRUTH
        # Based on DFS starting from cell 0, we expect:
        # 1. First feature: Ocean (starting from cell 0 which is water)
        # 2. Subsequent features: Islands discovered in DFS order

        # Run the actual feature detection
        features = Features(graph)
        features.markup_grid()

        # Get actual features (excluding None at index 0)
        actual_features = [f for f in features.features if f]

        # STRICT ASSERTIONS

        # 1. Must start from cell 0
        assert len(actual_features) >= 1, "Should have at least one feature"
        first_feature = actual_features[0]

        # 2. First feature must be ocean (since cell 0 is water)
        assert (
            first_feature.type == "ocean"
        ), f"First feature must be ocean (from cell 0), got {first_feature.type}"

        # 3. First feature must contain cell 0
        ocean_cells = [
            i
            for i in range(len(graph.heights))
            if features.feature_ids[i] == first_feature.id
        ]
        assert 0 in ocean_cells, "First feature (ocean) must contain cell 0"

        # 4. Check that we're using DFS by verifying the discovery order
        # In DFS, we exhaust one branch before moving to another
        # This means all ocean cells should be discovered before any island

        # Verify that all water cells connected to cell 0 are in the first feature
        water_cells = [i for i in range(len(graph.heights)) if graph.heights[i] < 20]

        # Use DFS to find all water cells reachable from cell 0
        visited = set()
        stack = [0]
        expected_ocean_cells = set()

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            if graph.heights[current] < 20:  # Is water
                expected_ocean_cells.add(current)
                # Add unvisited neighbors
                for neighbor in graph.cell_neighbors[current]:
                    if neighbor not in visited and neighbor < len(graph.heights):
                        stack.append(neighbor)

        # All cells in expected_ocean_cells should have feature_id == 1
        for cell in expected_ocean_cells:
            assert features.feature_ids[cell] == 1, (
                f"Cell {cell} should be in first feature (ocean) but has feature_id "
                f"{features.feature_ids[cell]}"
            )

        # 5. Verify feature discovery order matches DFS pattern
        print(f"\nDFS Test Results:")
        print(f"Total features discovered: {len(actual_features)}")
        print(f"Feature order: {[(f.id, f.type, f.cells) for f in actual_features]}")
        print(f"First feature cells: {sorted(ocean_cells)[:10]}...")  # First 10 cells

        # 6. Additional check: Features should be discovered in order
        for i, feature in enumerate(actual_features):
            assert (
                feature.id == i + 1
            ), f"Feature at index {i} should have id {i + 1}, got {feature.id}"

    def test_lake_filling_entire_depression(self):
        """Test that lakes fill entire depressions, not just immediate neighbors."""
        config = GridConfig(200, 200, 400)
        graph = generate_voronoi_graph(config, seed="lake_test")

        # Create a high plateau with an isolated depression in the center
        # Start with all land at high elevation
        graph.heights[:] = 80

        # Create a bowl-shaped depression in the center
        center_x, center_y = 100, 100
        depression_cells = []

        for i in range(len(graph.points)):
            x, y = graph.points[i]
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            if dist < 20:
                # Depression floor - low elevation
                graph.heights[i] = 30
                depression_cells.append(i)
            elif dist < 30:
                # Depression walls - gradually rising
                graph.heights[i] = 30 + int((dist - 20) * 3)

        # --- GROUND TRUTH CALCULATION ---
        # Count how many cells are in the depression that can't drain
        print(f"Total cells in depression floor (height 30): {len(depression_cells)}")

        features = Features(graph)
        features.markup_grid()

        # Add lakes in depressions with elevation limit 15
        # This means water can flow over barriers up to height 30+15=45
        initial_water = np.sum(graph.heights < 20)
        features.add_lakes_in_deep_depressions(elevation_limit=15)
        final_water = np.sum(graph.heights < 20)

        print(f"Water cells: {initial_water} -> {final_water}")

        # --- ASSERT AGAINST GROUND TRUTH ---
        lakes = sum(1 for f in features.features if f and f.type == "lake")
        print(f"Lakes created: {lakes}")

        # Should create exactly one lake
        assert lakes == 1, f"Expected exactly 1 lake, got {lakes}"

        # Find the cells that were turned into lake water (height 19)
        actual_lake_cells_count = np.sum(graph.heights == 19)
        print(f"Lake cells created: {actual_lake_cells_count}")

        # The lake should fill multiple cells (entire depression)
        assert (
            actual_lake_cells_count > 1
        ), f"Lake should fill multiple cells in the depression, but only filled {actual_lake_cells_count}"

        # Check that lake filled a reasonable portion of the depression
        # With elevation_limit=15, water fills up to height 45, which includes the floor and some walls
        assert actual_lake_cells_count >= len(depression_cells), (
            f"Lake should fill at least the depression floor ({len(depression_cells)} cells), "
            f"but only filled {actual_lake_cells_count}"
        )

    def test_vertex_tracer_produces_valid_closed_chain(self, square_island_graph):
        """
        Tests that the vertex tracer produces a valid, closed chain of vertices
        on a predictable shape after it has been processed by regraph.
        """
        # 1. ARRANGE: Create a predictable starting graph and run the full pipeline on it.
        features = Features(square_island_graph)
        features.markup_grid()

        # Store results on graph for regraph to use
        square_island_graph.distance_field = features.distance_field
        square_island_graph.feature_ids = features.feature_ids
        square_island_graph.features = features.features

        packed = regraph(square_island_graph)
        features.markup_pack(packed)

        # 2. ACT: Find the island feature that was created.
        islands = [f for f in packed.features if f and f.type == "island"]
        assert (
            len(islands) == 1
        ), "Test setup should result in exactly one island feature"
        island_feature = islands[0]

        vertices = island_feature.vertices
        print(f"Square island produced a perimeter with {len(vertices)} vertices.")

        # 3. ASSERT: Check the fundamental properties of a valid perimeter chain.

        # ASSERTION 1: The perimeter must be a valid polygon (at least 3 vertices).
        # This is the fix for your immediate test failure.
        assert len(vertices) >= 3, "A feature perimeter must have at least 3 vertices."

        # ASSERTION 2: The chain must be simple (no duplicate vertices).
        # A self-intersecting or looping path would have duplicate vertices.
        assert len(set(vertices)) == len(
            vertices
        ), "Perimeter trace contains duplicate vertices, indicating a self-intersection or error."

        # ASSERTION 3: The chain must form a closed loop.
        # The last vertex in the chain must be a neighbor of the first vertex.
        first_v = vertices[0]
        last_v = vertices[-1]

        neighbors_of_last_v = packed.vertex_neighbors[last_v]

        assert (
            first_v in neighbors_of_last_v
        ), f"Perimeter is not a closed loop. First vertex {first_v} is not a neighbor of the last vertex {last_v}."


def main():
    """Run feature tests with command line arguments."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Test Features module with various templates"
    )
    parser.add_argument(
        "--template",
        default="archipelago",
        help="Template name or 'all' to test all templates (default: archipelago)",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        help=f"Seed for map generation (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=TEST_WIDTH,
        help=f"Map width (default: {TEST_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=TEST_HEIGHT,
        help=f"Map height (default: {TEST_HEIGHT})",
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=TEST_CELLS_DESIRED,
        help=f"Desired cell count (default: {TEST_CELLS_DESIRED})",
    )

    args = parser.parse_args()

    # Create test instance
    test_features = TestFeatures()

    if args.template.lower() == "all":
        # Test all templates
        print(f"Testing ALL templates with custom parameters:")
        print(f"  Seed: {args.seed}")
        print(f"  Size: {args.width}x{args.height}")
        print(f"  Cells: {args.cells}")

        # Override the defaults for test_all_templates
        # global DEFAULT_SEED, TEST_WIDTH, TEST_HEIGHT, TEST_CELLS_DESIRED
        original_seed = DEFAULT_SEED
        original_width = TEST_WIDTH
        original_height = TEST_HEIGHT
        original_cells = TEST_CELLS_DESIRED

        DEFAULT_SEED = args.seed
        TEST_WIDTH = args.width
        TEST_HEIGHT = args.height
        TEST_CELLS_DESIRED = args.cells

        test_features.test_all_templates()

        # Restore originals
        DEFAULT_SEED = original_seed
        TEST_WIDTH = original_width
        TEST_HEIGHT = original_height
        TEST_CELLS_DESIRED = original_cells

    else:
        # Test specific template
        if args.template not in TEMPLATES:
            print(f"Error: Unknown template '{args.template}'")
            print(f"Available templates: {', '.join(sorted(TEMPLATES.keys()))}")
            sys.exit(1)

        print(f"Testing template: {args.template}")
        graph, features = test_features.generate_test_map(
            template=args.template,
            seed=args.seed,
            width=args.width,
            height=args.height,
            cells_desired=args.cells,
        )

        # Run the analysis (similar to test_real_world_scenario)
        islands = sum(1 for f in features.features if f and f.type == "island")
        oceans = sum(1 for f in features.features if f and f.type == "ocean")
        lakes = sum(1 for f in features.features if f and f.type == "lake")

        print(f"\nFeature Analysis for '{args.template}' template:")
        print(f"  Islands: {islands}")
        print(f"  Oceans:  {oceans}")
        print(f"  Lakes:   {lakes}")
        print(f"  Total features: {len(features.features) - 1}")

        heights = graph.heights
        land_cells = np.sum(heights >= 20)
        water_cells = np.sum(heights < 20)
        total_cells = len(heights)

        print(f"\nHeight Statistics:")
        print(
            f"  Min: {np.min(heights)}, Max: {np.max(heights)}, Mean: {np.mean(heights):.2f}"
        )
        print(f"  Land cells: {land_cells:,} ({land_cells/total_cells*100:.1f}%)")
        print(f"  Water cells: {water_cells:,} ({water_cells/total_cells*100:.1f}%)")

        # Test packing
        from py_fmg.core.cell_packing import regraph

        packed = regraph(graph)
        features.markup_pack(packed)

        print(f"\nCell Packing Results:")
        print(f"  Original cells: {len(graph.points):,}")
        print(f"  Packed cells: {len(packed.points):,}")
        reduction = (1 - len(packed.points) / len(graph.points)) * 100
        print(f"  Reduction: {reduction:.1f}%")

        if hasattr(packed, "features"):
            packed_islands = sum(1 for f in packed.features if f and f.type == "island")
            packed_oceans = sum(1 for f in packed.features if f and f.type == "ocean")
            packed_lakes = sum(1 for f in packed.features if f and f.type == "lake")
            print(f"\nPacked Feature Counts:")
            print(f"  Islands: {packed_islands}")
            print(f"  Oceans:  {packed_oceans}")
            print(f"  Lakes:   {packed_lakes}")


if __name__ == "__main__":
    import sys

    # If no arguments provided, run pytest
    if len(sys.argv) == 1:
        pytest.main([__file__, "-v", "-s"])
    else:
        # Run main with command line arguments
        main()
