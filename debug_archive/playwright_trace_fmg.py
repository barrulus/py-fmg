#!/usr/bin/env python3
"""
Use Playwright to trace FMG heightmap generation step by step.
This will help us understand what's actually happening in the browser.
"""

import json
import asyncio
from playwright.async_api import async_playwright
import os


async def trace_fmg_heightmap():
    async with async_playwright() as p:
        # Launch browser with devtools open
        browser = await p.chromium.launch(
            headless=False,  # We want to see what's happening
            devtools=True
        )
        
        context = await browser.new_context()
        page = await context.new_page()
        
        # Enable console logging
        page.on("console", lambda msg: print(f"[Console] {msg.type()}: {msg.text()}"))
        
        # Navigate to local FMG instance
        fmg_path = os.path.abspath("fmg/index.html")
        await page.goto(f"file://{fmg_path}")
        
        # Wait for FMG to load
        await page.wait_for_timeout(3000)
        
        # Inject our debugging code
        await page.evaluate("""
            // Override the heightmap generator to add logging
            window.originalHeightmapGenerate = HeightmapGenerator.generate;
            window.heightmapLog = [];
            
            // Patch the blob power calculation
            const originalGetBlobPower = HeightmapGenerator.getBlobPower || function(cells) {
                const blobPowerMap = {
                    1000: 0.93,
                    2000: 0.95,
                    5000: 0.97,
                    10000: 0.98,
                    20000: 0.99,
                    30000: 0.991,
                    40000: 0.993,
                    50000: 0.994,
                    60000: 0.995,
                    70000: 0.9955,
                    80000: 0.996,
                    90000: 0.9964,
                    100000: 0.9973
                };
                return blobPowerMap[cells] || 0.98;
            };
            
            // Track what happens during generation
            window.trackHeightmap = true;
            
            // Override Math.random to track random values
            const originalRandom = Math.random;
            window.randomCalls = [];
            Math.random = function() {
                const value = originalRandom.call(this);
                if (window.trackHeightmap) {
                    window.randomCalls.push(value);
                }
                return value;
            };
            
            console.log("Debugging hooks installed");
        """)
        
        # Set specific generation parameters
        await page.evaluate("""
            // Set the seed
            document.getElementById('optionsSeed').value = '651658815';
            
            // Set map size
            document.getElementById('mapSizeInput').value = 'cropped';
            graphWidth = 300;
            graphHeight = 300;
            
            // Set heightmap template
            document.getElementById('templateInput').value = 'lowIsland';
            
            // Set cells count
            document.getElementById('cellsNumberInput').value = '10000';
            
            console.log("Generation parameters set");
        """)
        
        # Add more detailed tracking
        await page.evaluate("""
            // Track the first hill operation in detail
            window.hillTracking = [];
            
            // Patch the addHill function if accessible
            if (window.HeightmapGenerator && typeof HeightmapGenerator.addHill === 'function') {
                const originalAddHill = HeightmapGenerator.addHill;
                HeightmapGenerator.addHill = function(count, height, rangeX, rangeY) {
                    console.log(`addHill called: count=${count}, height=${height}, rangeX=${rangeX}, rangeY=${rangeY}`);
                    window.hillTracking.push({
                        count, height, rangeX, rangeY,
                        randomCalls: window.randomCalls.length
                    });
                    return originalAddHill.apply(this, arguments);
                };
            }
        """)
        
        # Generate the map
        print("Generating map...")
        await page.evaluate("regenerateMap('menu');")
        
        # Wait for generation to complete
        await page.wait_for_timeout(5000)
        
        # Extract debugging data
        debug_data = await page.evaluate("""
            // Get the generated data
            const data = {
                seed: seed,
                graphWidth: graphWidth,
                graphHeight: graphHeight,
                cellsDesired: grid.cellsDesired,
                cellsActual: grid.cells.h.length,
                heightsMin: Math.min(...grid.cells.h),
                heightsMax: Math.max(...grid.cells.h),
                heightsMean: grid.cells.h.reduce((a, b) => a + b) / grid.cells.h.length,
                landCells: grid.cells.h.filter(h => h >= 20).length,
                waterCells: grid.cells.h.filter(h => h < 20).length,
                first20Heights: Array.from(grid.cells.h.slice(0, 20)),
                randomCallsCount: window.randomCalls.length,
                first10Randoms: window.randomCalls.slice(0, 10),
                hillTracking: window.hillTracking,
                blobPower: originalGetBlobPower(grid.cellsDesired),
                linePower: 0.81  // Check what it actually is
            };
            
            // Try to get line power if function exists
            try {
                if (typeof getLinePower === 'function') {
                    data.linePowerActual = getLinePower();
                }
            } catch (e) {
                data.linePowerError = e.toString();
            }
            
            return data;
        """)
        
        print("\nDebug Data:")
        print(json.dumps(debug_data, indent=2))
        
        # Export the full heightmap
        heights = await page.evaluate("Array.from(grid.cells.h)")
        
        # Save to file
        with open("fmg_browser_heights.json", "w") as f:
            json.dump({
                "debug": debug_data,
                "heights": heights
            }, f, indent=2)
        
        print(f"\nSaved {len(heights)} height values to fmg_browser_heights.json")
        
        # Check for any errors
        errors = await page.evaluate("window.fmgErrors || []")
        if errors:
            print("\nErrors detected:")
            for error in errors:
                print(f"  - {error}")
        
        # Keep browser open for manual inspection
        print("\nBrowser will stay open for manual inspection. Press Ctrl+C to close.")
        try:
            await asyncio.Future()  # Wait forever
        except KeyboardInterrupt:
            pass
        
        await browser.close()


async def main():
    """Main entry point."""
    print("Starting FMG heightmap tracing with Playwright...")
    print("Make sure you have FMG files in the 'fmg' directory")
    print("-" * 60)
    
    await trace_fmg_heightmap()


if __name__ == "__main__":
    asyncio.run(main())