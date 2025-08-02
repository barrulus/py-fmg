#!/usr/bin/env python3
"""
Test runner for py-fmg system validation.

This script runs a comprehensive test suite to validate all system components.
"""

import sys
import time
import requests
import subprocess
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
API_BASE_URL = "http://localhost:9888"
VIEWER_BASE_URL = "http://localhost:8081"


def check_service_health(url: str, service_name: str, timeout: int = 10) -> bool:
    """Check if a service is healthy."""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        if response.status_code == 200:
            logger.info(f"✅ {service_name} is healthy")
            return True
        else:
            logger.error(f"❌ {service_name} returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ {service_name} is not responding: {e}")
        return False


def test_api_endpoints():
    """Test basic API endpoints."""
    logger.info("🧪 Testing API endpoints...")
    
    endpoints = [
        ("/maps", "GET"),
        ("/openapi.json", "GET"),
        ("/docs", "GET")
    ]
    
    for endpoint, method in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
            else:
                response = requests.post(f"{API_BASE_URL}{endpoint}", timeout=10)
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ {method} {endpoint} - OK")
            else:
                logger.warning(f"⚠️  {method} {endpoint} - Status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ {method} {endpoint} - Error: {e}")


def test_map_generation():
    """Test map generation functionality."""
    logger.info("🗺️  Testing map generation...")
    
    try:
        # Generate a small test map
        response = requests.post(
            f"{API_BASE_URL}/maps/generate",
            json={
                "map_name": "Test_Validation_Map",
                "width": 200,
                "height": 150,
                "seed": 12345
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Map generation started - Job ID: {data.get('job_id', 'N/A')}")
            return data.get('job_id')
        else:
            logger.error(f"❌ Map generation failed - Status {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"❌ Map generation error: {e}")
        return None


def test_database_connectivity():
    """Test database connectivity through API."""
    logger.info("🗄️  Testing database connectivity...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/maps", timeout=10)
        if response.status_code == 200:
            maps = response.json()
            logger.info(f"✅ Database connectivity OK - Found {len(maps)} maps")
            return True
        else:
            logger.error(f"❌ Database connectivity failed - Status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Database connectivity error: {e}")
        return False


def test_3d_visualization():
    """Test 3D visualization setup."""
    logger.info("🎮 Testing 3D visualization...")
    
    try:
        # Get maps to test with
        response = requests.get(f"{API_BASE_URL}/maps", timeout=10)
        if response.status_code != 200:
            logger.warning("⚠️  No maps available for 3D testing")
            return False
        
        maps = response.json()
        if not maps:
            logger.warning("⚠️  No maps found for 3D testing")
            return False
        
        map_id = maps[0]["id"]
        
        # Test visualization info
        response = requests.get(f"{API_BASE_URL}/maps/{map_id}/visualize/info", timeout=10)
        if response.status_code == 200:
            viz_info = response.json()
            logger.info(f"✅ 3D visualization info OK - Map: {viz_info.get('map_name', 'Unknown')}")
            return True
        else:
            logger.error(f"❌ 3D visualization info failed - Status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ 3D visualization error: {e}")
        return False


def test_editing_endpoints():
    """Test editing endpoints."""
    logger.info("✏️  Testing editing endpoints...")
    
    try:
        # Get maps to test with
        response = requests.get(f"{API_BASE_URL}/maps", timeout=10)
        if response.status_code != 200 or not response.json():
            logger.warning("⚠️  No maps available for editing tests")
            return False
        
        maps = response.json()
        map_id = maps[0]["id"]
        
        # Test terrain editing
        response = requests.post(
            f"{API_BASE_URL}/maps/{map_id}/edit/terrain",
            json={
                "cell_indices": [1, 2],
                "operation": "set_height",
                "value": 100.0
            },
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("✅ Terrain editing OK")
            return True
        else:
            logger.error(f"❌ Terrain editing failed - Status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Editing endpoints error: {e}")
        return False


def test_file_permissions():
    """Test file system permissions."""
    logger.info("📁 Testing file permissions...")
    
    try:
        # Test tiles directory
        tiles_dir = Path("tiles")
        tiles_dir.mkdir(exist_ok=True)
        
        # Test write permissions
        test_file = tiles_dir / "test_permissions.txt"
        test_file.write_text("test")
        
        # Test read permissions
        content = test_file.read_text()
        assert content == "test"
        
        # Cleanup
        test_file.unlink()
        
        logger.info("✅ File permissions OK")
        return True
    except Exception as e:
        logger.error(f"❌ File permissions error: {e}")
        return False


def test_docker_services():
    """Test Docker services status."""
    logger.info("🐳 Testing Docker services...")
    
    try:
        # Check if Docker is running
        result = subprocess.run(
            ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            output = result.stdout
            logger.info("✅ Docker services status:")
            for line in output.split('\n')[1:]:  # Skip header
                if line.strip():
                    logger.info(f"   {line}")
            return True
        else:
            logger.error(f"❌ Docker command failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Docker services error: {e}")
        return False


def run_comprehensive_tests():
    """Run all tests and return summary."""
    logger.info("🚀 Starting comprehensive system validation...")
    
    tests = [
        ("Service Health", lambda: check_service_health(API_BASE_URL, "API Service")),
        ("Database Connectivity", test_database_connectivity),
        ("API Endpoints", test_api_endpoints),
        ("File Permissions", test_file_permissions),
        ("Docker Services", test_docker_services),
        ("Map Generation", lambda: test_map_generation() is not None),
        ("3D Visualization", test_3d_visualization),
        ("Editing Endpoints", test_editing_endpoints),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for production.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please review the issues above.")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick health check only
        logger.info("🏃 Running quick health check...")
        api_healthy = check_service_health(API_BASE_URL, "API Service")
        db_healthy = test_database_connectivity()
        
        if api_healthy and db_healthy:
            logger.info("✅ Quick health check passed")
            sys.exit(0)
        else:
            logger.error("❌ Quick health check failed")
            sys.exit(1)
    else:
        # Full test suite
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

