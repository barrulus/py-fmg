"""
Example demonstrating climate calculation system.
"""

import numpy as np
import matplotlib.pyplot as plt
from py_fmg.core import (
    GridConfig, generate_voronoi_graph, 
    HeightmapGenerator, HeightmapConfig,
    Climate, ClimateOptions, MapCoordinates
)
from py_fmg.core.features import Features
from py_fmg.core.cell_packing import regraph


def main():
    # Configuration
    seed = "climate_demo"
    template = "archipelago"  # Creates islands with varied terrain
    
    # Create map
    config = GridConfig(width=400, height=300, cells_desired=2000)
    graph = generate_voronoi_graph(config, seed=seed)
    
    # Generate heightmap
    heightmap_config = HeightmapConfig(
        width=int(config.width),
        height=int(config.height),
        cells_x=graph.cells_x,
        cells_y=graph.cells_y,
        cells_desired=config.cells_desired,
        spacing=graph.spacing
    )
    
    heightmap_gen = HeightmapGenerator(heightmap_config, graph, seed=seed)
    heights = heightmap_gen.from_template(template, seed=seed)
    graph.heights = heights
    
    # Detect features (needed for proper coastline detection)
    features = Features(graph)
    features.markup_grid()
    
    # Calculate climate
    print("Calculating climate...")
    
    # Set up climate with custom options
    climate_options = ClimateOptions(
        temperature_equator=28,  # Warm equator
        temperature_north_pole=-35,
        temperature_south_pole=-30,
        precipitation_modifier=1.2  # Slightly wetter world
    )
    
    # Map spans from 60°N to 30°S
    map_coords = MapCoordinates(lat_n=60, lat_s=-30)
    
    climate = Climate(graph, options=climate_options, map_coords=map_coords)
    
    # Calculate temperature and precipitation
    climate.calculate_temperatures()
    climate.generate_precipitation()
    
    print(f"Temperature range: {np.min(graph.temperatures)}°C to {np.max(graph.temperatures)}°C")
    print(f"Average temperature: {np.mean(graph.temperatures):.1f}°C")
    print(f"Precipitation range: {np.min(graph.precipitation)} to {np.max(graph.precipitation)} mm")
    print(f"Average precipitation: {np.mean(graph.precipitation):.1f} mm")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Height map
    ax = axes[0, 0]
    scatter = ax.scatter(graph.points[:, 0], graph.points[:, 1], 
                        c=graph.heights, cmap='terrain', s=5)
    ax.set_title('Elevation')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Height')
    
    # Temperature map
    ax = axes[0, 1]
    scatter = ax.scatter(graph.points[:, 0], graph.points[:, 1], 
                        c=graph.temperatures, cmap='RdBu_r', s=5,
                        vmin=-30, vmax=30)
    ax.set_title('Temperature')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='°C')
    
    # Precipitation map
    ax = axes[1, 0]
    scatter = ax.scatter(graph.points[:, 0], graph.points[:, 1], 
                        c=graph.precipitation, cmap='YlGnBu', s=5)
    ax.set_title('Precipitation')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='mm')
    
    # Combined climate zones (simplified biomes)
    ax = axes[1, 1]
    
    # Simple biome classification based on temperature and precipitation
    biomes = np.zeros(len(graph.points))
    for i in range(len(graph.points)):
        temp = graph.temperatures[i]
        prec = graph.precipitation[i]
        
        if graph.heights[i] < 20:  # Water
            biomes[i] = 0
        elif temp < -5:  # Tundra
            biomes[i] = 1
        elif temp < 20 and prec < 60:  # Steppe
            biomes[i] = 2
        elif temp < 20 and prec >= 60:  # Temperate forest
            biomes[i] = 3
        elif temp >= 20 and prec < 60:  # Desert
            biomes[i] = 4
        elif temp >= 20 and prec < 150:  # Savanna
            biomes[i] = 5
        else:  # Tropical forest
            biomes[i] = 6
    
    colors = ['blue', 'white', 'tan', 'green', 'yellow', 'orange', 'darkgreen']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    scatter = ax.scatter(graph.points[:, 0], graph.points[:, 1], 
                        c=biomes, cmap=cmap, s=5, vmin=0, vmax=6)
    ax.set_title('Climate Zones')
    ax.set_aspect('equal')
    
    # Add legend
    labels = ['Ocean', 'Tundra', 'Steppe', 'Temperate', 'Desert', 'Savanna', 'Tropical']
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(7))
    cbar.ax.set_yticklabels(labels)
    
    plt.tight_layout()
    plt.savefig('climate_demo.png', dpi=150)
    print("\nClimate visualization saved to climate_demo.png")
    
    # Print some statistics
    print("\nBiome distribution:")
    for i, label in enumerate(labels):
        count = np.sum(biomes == i)
        pct = count / len(biomes) * 100
        print(f"  {label}: {count} cells ({pct:.1f}%)")


if __name__ == "__main__":
    main()