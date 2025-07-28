#!/usr/bin/env python3
"""
Test script for the improved heightmap generation system.

Tests the complete pipeline:
1. Dual seed system (grid_seed vs map_seed)
2. Voronoi graph generation
3. Heightmap generation with fractious template
4. Cell packing (pack_graph)
5. Compatibility improvements
"""

import sys
import time
import numpy as np
from py_fmg.core.voronoi_graph import GridConfig, generate_voronoi_graph, pack_graph
from py_fmg.core.heightmap_generator import HeightmapGenerator, HeightmapConfig


def test_heightmap_generation():
    """Test the complete heightmap generation pipeline."""
    print("🧪 Testing Heightmap Generation Pipeline")
    print("=" * 50)
    
    # Test parameters matching FMG behavior: SAME seed for both Voronoi and heightmap
    main_seed = "651658815"  # Main seed from FMG analysis
    grid_seed = main_seed    # Voronoi uses main seed
    map_seed = main_seed     # Heightmap uses SAME seed (FMG resets PRNG)
    template_name = "lowIsland"  # Template used in FMG test data
    
    print(f"Grid Seed: {grid_seed}")
    print(f"Map Seed: {map_seed}")
    print(f"Template: {template_name}")
    print()
    
    # Step 1: Generate Voronoi Graph
    print("📊 Step 1: Generating Voronoi Graph...")
    start_time = time.time()
    
    # Match FMG parameters: 300x300 canvas, 10000 points -> 4499 cells
    config = GridConfig(
        width=300,
        height=300,
        cells_desired=10000
    )
    
    voronoi_graph = generate_voronoi_graph(config, grid_seed)
    
    voronoi_time = time.time() - start_time
    print(f"✅ Voronoi generation complete: {voronoi_time:.2f}s")
    print(f"   Original cells: {len(voronoi_graph.points)}")
    print()
    
    # Step 2: Generate Heightmap
    print("🏔️  Step 2: Generating Heightmap...")
    start_time = time.time()
    
    heightmap_config = HeightmapConfig(
        width=int(config.width),
        height=int(config.height),
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=config.cells_desired
    )
    
    heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph)
    heights = heightmap_gen.from_template(template_name, map_seed)
    
    heightmap_time = time.time() - start_time
    print(f"✅ Heightmap generation complete: {heightmap_time:.2f}s")
    print(f"   Height range: {np.min(heights)}-{np.max(heights)}")
    print(f"   Mean height: {np.mean(heights):.1f}")
    print()
    
    # Step 3: Pack Graph
    print("📦 Step 3: Packing Graph...")
    start_time = time.time()
    
    packed_graph = pack_graph(voronoi_graph, heights)
    
    pack_time = time.time() - start_time
    print(f"✅ Graph packing complete: {pack_time:.2f}s")
    print(f"   Packed cells: {len(packed_graph.points)}")
    print(f"   Reduction: {len(voronoi_graph.points)} → {len(packed_graph.points)} ({((len(voronoi_graph.points) - len(packed_graph.points)) / len(voronoi_graph.points) * 100):.1f}% removed)")
    print()
    
    # Step 4: Analysis
    print("📈 Step 4: Compatibility Analysis...")
    
    # Height distribution analysis
    land_cells = np.sum(heights >= 20)
    water_cells = np.sum(heights < 20)
    
    print(f"   Land cells (≥20): {land_cells} ({land_cells/len(heights)*100:.1f}%)")
    print(f"   Water cells (<20): {water_cells} ({water_cells/len(heights)*100:.1f}%)")
    print(f"   Packed cells match land cells: {'✅' if len(packed_graph.points) == land_cells else '❌'}")
    
    # Compare with expected FMG values
    expected_fmg_range = (4, 100)  # From our analysis
    actual_range = (int(np.min(heights)), int(np.max(heights)))
    
    print(f"   Expected FMG range: {expected_fmg_range}")
    print(f"   Actual range: {actual_range}")
    print(f"   Range compatibility: {'✅' if actual_range[0] <= expected_fmg_range[0] and actual_range[1] >= expected_fmg_range[1] else '⚠️'}")
    
    # Packed cell count analysis
    expected_packed_cells = 4500  # Approximate from analysis
    actual_packed_cells = len(packed_graph.points)
    
    print(f"   Expected packed cells: ~{expected_packed_cells}")
    print(f"   Actual packed cells: {actual_packed_cells}")
    print(f"   Cell count compatibility: {'✅' if abs(actual_packed_cells - expected_packed_cells) < 1000 else '⚠️'}")
    
    print()
    
    # Step 5: Summary
    total_time = voronoi_time + heightmap_time + pack_time
    print("🎯 Summary:")
    print(f"   Total generation time: {total_time:.2f}s")
    print(f"   Final cell count: {len(packed_graph.points)} (vs original {len(voronoi_graph.points)})")
    print(f"   Memory optimization: {((len(voronoi_graph.points) - len(packed_graph.points)) / len(voronoi_graph.points) * 100):.1f}% reduction")
    
    return {
        'voronoi_graph': voronoi_graph,
        'packed_graph': packed_graph,
        'heights': heights,
        'times': {
            'voronoi': voronoi_time,
            'heightmap': heightmap_time,
            'packing': pack_time,
            'total': total_time
        }
    }


def test_template_variations():
    """Test different templates to verify template parsing works."""
    print("\n🎨 Testing Template Variations")
    print("=" * 50)
    
    templates = ['highIsland', 'lowIsland', 'archipelago', 'fractious']
    
    # Use smaller config for speed
    config = GridConfig(width=400, height=300, cells_desired=2000)
    voronoi_graph = generate_voronoi_graph(config, "test123")
    
    heightmap_config = HeightmapConfig(
        width=int(config.width),
        height=int(config.height),
        cells_x=voronoi_graph.cells_x,
        cells_y=voronoi_graph.cells_y,
        cells_desired=config.cells_desired
    )
    
    for template in templates:
        print(f"Testing template: {template}")
        start_time = time.time()
        
        heightmap_gen = HeightmapGenerator(heightmap_config, voronoi_graph)
        heights = heightmap_gen.from_template(template, "template_test")
        
        generation_time = time.time() - start_time
        
        print(f"   ✅ Generated in {generation_time:.2f}s")
        print(f"   Range: {np.min(heights)}-{np.max(heights)}, Mean: {np.mean(heights):.1f}")
        
        # Test packing
        packed_graph = pack_graph(voronoi_graph, heights)
        reduction = (len(voronoi_graph.points) - len(packed_graph.points)) / len(voronoi_graph.points) * 100
        print(f"   Packed: {len(voronoi_graph.points)} → {len(packed_graph.points)} ({reduction:.1f}% reduction)")
        print()


if __name__ == "__main__":
    try:
        # Test main pipeline
        results = test_heightmap_generation()
        
        # Test template variations
        test_template_variations()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)