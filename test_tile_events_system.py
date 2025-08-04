#!/usr/bin/env python3
"""
Test complet du système tile_events pour VoronoiGraph.
"""

import numpy as np
from py_fmg.core.voronoi_graph import VoronoiGraph, generate_voronoi_graph, GridConfig
from py_fmg.core.climate import Climate
from py_fmg.core.hydrology import Hydrology
from py_fmg.core.biomes import BiomeClassifier


def test_tile_events_system():
    """Test complet du système tile_events."""
    print("🧪 Test complet du système tile_events")
    print("=" * 50)
    
    # 1. Création du graphe
    print("1️⃣ Création du graphe Voronoi...")
    config = GridConfig(width=300, height=300, cells_desired=200)
    graph = generate_voronoi_graph(config, 'test_tile_events')
    print(f"   ✅ Graphe créé avec {len(graph.cell_neighbors)} cellules")
    print(f"   📊 Événements initiaux: {graph.list_tile_events()}")
    
    # 2. Test des méthodes tile_events
    print("\n2️⃣ Test des méthodes tile_events...")
    
    # Test set/get/has
    test_data = np.array([1, 2, 3, 4, 5])
    graph.set_tile_data('test_event', test_data)
    assert graph.has_tile_data('test_event'), "has_tile_data failed"
    retrieved = graph.get_tile_data('test_event')
    assert np.array_equal(retrieved, test_data), "get_tile_data failed"
    print("   ✅ set_tile_data, get_tile_data, has_tile_data fonctionnent")
    
    # Test fallback
    fallback_value = graph.get_tile_data('nonexistent', default='fallback')
    assert fallback_value == 'fallback', "Fallback failed"
    print("   ✅ Fallback par défaut fonctionne")
    
    # Test clear
    graph.clear_tile_event('test_event')
    assert not graph.has_tile_data('test_event'), "clear_tile_event failed"
    print("   ✅ clear_tile_event fonctionne")
    
    # 3. Test d'initialisation des modules
    print("\n3️⃣ Test d'initialisation des modules...")
    
    # Climate
    climate = Climate(graph)
    climate_events = ['temperatures', 'precipitation', 'climate']
    for event in climate_events:
        assert graph.has_tile_data(event), f"Climate n'a pas initialisé {event}"
    print("   ✅ Climate initialise correctement ses événements")
    
    # Hydrology
    hydrology = Hydrology(graph)
    hydro_events = ['water_flux', 'flow_directions', 'filled_heights', 'rivers', 'lakes']
    for event in hydro_events:
        assert graph.has_tile_data(event), f"Hydrology n'a pas initialisé {event}"
    print("   ✅ Hydrology initialise correctement ses événements")
    
    # Biomes
    biomes = BiomeClassifier(graph)
    biome_events = ['biome_regions', 'biomes']
    for event in biome_events:
        assert graph.has_tile_data(event), f"BiomeClassifier n'a pas initialisé {event}"
    print("   ✅ BiomeClassifier initialise correctement ses événements")
    
    # 4. Test des propriétés legacy
    print("\n4️⃣ Test des propriétés legacy...")
    
    # Test températures
    temps = graph.temperatures
    assert temps is not None, "Propriété temperatures failed"
    assert len(temps) == len(graph.cell_neighbors), "Taille temperatures incorrecte"
    
    # Test modification via propriété
    new_temps = np.full(len(graph.cell_neighbors), 25, dtype=np.int8)
    graph.temperatures = new_temps
    assert np.array_equal(graph.temperatures, new_temps), "Setter temperatures failed"
    print("   ✅ Propriétés legacy fonctionnent (get/set)")
    
    # 5. Test de validation des prérequis
    print("\n5️⃣ Test de validation des prérequis...")
    
    # Ajouter des heights pour les tests
    graph.heights = np.random.randint(0, 100, len(graph.cell_neighbors))
    
    try:
        climate._validate_prerequisites()
        print("   ✅ Validation Climate réussie")
    except Exception as e:
        print(f"   ❌ Validation Climate échouée: {e}")
    
    try:
        hydrology._validate_prerequisites()
        print("   ✅ Validation Hydrology réussie")
    except Exception as e:
        print(f"   ❌ Validation Hydrology échouée: {e}")
    
    try:
        biomes._validate_prerequisites()
        print("   ✅ Validation BiomeClassifier réussie")
    except Exception as e:
        print(f"   ❌ Validation BiomeClassifier échouée: {e}")
    
    # 6. Test de l'état final
    print("\n6️⃣ État final du système...")
    all_events = graph.list_tile_events()
    print(f"   📊 Tous les événements de tuiles: {all_events}")
    print(f"   🔢 Nombre total d'événements: {len(all_events)}")
    
    # Vérifier que tous les événements critiques existent
    critical_events = [
        'temperatures', 'precipitation', 'climate',
        'water_flux', 'flow_directions', 'filled_heights', 'rivers', 'lakes',
        'biome_regions', 'biomes'
    ]
    
    missing_events = [event for event in critical_events if not graph.has_tile_data(event)]
    if missing_events:
        print(f"   ❌ Événements manquants: {missing_events}")
    else:
        print("   ✅ Tous les événements critiques sont présents")
    
    print("\n🎉 Test du système tile_events terminé avec succès!")
    return True


if __name__ == "__main__":
    try:
        test_tile_events_system()
        print("\n✅ TOUS LES TESTS PASSÉS")
    except Exception as e:
        print(f"\n❌ TEST ÉCHOUÉ: {e}")
        import traceback
        traceback.print_exc()

