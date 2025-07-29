// JavaScript to run in FMG browser console to export heightmap data
// Usage: Paste this into the browser console after FMG has generated a map

function exportHeightmapData() {
    // Check if pack.cells exists
    if (typeof pack === 'undefined' || !pack.cells || !pack.cells.h) {
        console.error("No heightmap data found. Generate a map first.");
        return;
    }
    
    const heights = pack.cells.h;
    const seed = seed || "unknown";
    const template = templateInput ? templateInput.value : "unknown";
    
    // Calculate statistics
    const totalCells = heights.length;
    const landCells = heights.filter(h => h >= 20).length;
    const waterCells = heights.filter(h => h < 20).length;
    const landPct = (landCells / totalCells * 100).toFixed(1);
    const waterPct = (waterCells / totalCells * 100).toFixed(1);
    
    // Height distribution
    const distribution = {};
    for (let threshold = 0; threshold <= 100; threshold += 10) {
        const count = heights.filter(h => h >= threshold).length;
        distribution[threshold] = {
            count: count,
            percentage: (count / totalCells * 100).toFixed(1)
        };
    }
    
    // Create export data
    const exportData = {
        seed: seed,
        template: template,
        totalCells: totalCells,
        landCells: landCells,
        waterCells: waterCells,
        landPercentage: parseFloat(landPct),
        waterPercentage: parseFloat(waterPct),
        statistics: {
            min: Math.min(...heights),
            max: Math.max(...heights),
            mean: heights.reduce((a, b) => a + b, 0) / totalCells,
            median: heights.slice().sort((a, b) => a - b)[Math.floor(totalCells / 2)]
        },
        distribution: distribution,
        // Sample of first 100 heights for detailed comparison
        sampleHeights: heights.slice(0, 100),
        // Grid dimensions
        grid: {
            width: graphWidth,
            height: graphHeight,
            cellsX: grid.cellsX,
            cellsY: grid.cellsY,
            cellsDesired: grid.cellsDesired || 10000,
            spacing: grid.spacing
        }
    };
    
    // Create downloadable JSON
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `fmg_heightmap_${template}_${seed}.json`;
    link.click();
    
    // Also log summary to console
    console.log("=== FMG Heightmap Export ===");
    console.log(`Template: ${template}`);
    console.log(`Seed: ${seed}`);
    console.log(`Total Cells: ${totalCells}`);
    console.log(`Land: ${landCells} (${landPct}%)`);
    console.log(`Water: ${waterCells} (${waterPct}%)`);
    console.log("\nHeight Distribution:");
    Object.entries(distribution).forEach(([threshold, data]) => {
        console.log(`  h >= ${threshold}: ${data.count} cells (${data.percentage}%)`);
    });
    
    console.log("\nExported to:", link.download);
}

// Run the export
exportHeightmapData();