// Debug patch for FMG heightmap generation
// Add this to index.html after the heightmap-generator.js script

(function() {
    console.log("Installing FMG debug patches...");
    
    // Store original functions
    const origGenerate = HeightmapGenerator.generate;
    const origFromTemplate = HeightmapGenerator.fromTemplate;
    const origAddHill = HeightmapGenerator.addHill;
    
    // Track generation state
    window.fmgDebug = {
        heightmapLog: [],
        randomLog: [],
        hillLog: []
    };
    
    // Patch Math.random to log values during heightmap generation
    const origRandom = Math.random;
    let trackingEnabled = false;
    
    Math.random = function() {
        const value = origRandom.call(this);
        if (trackingEnabled) {
            window.fmgDebug.randomLog.push(value);
        }
        return value;
    };
    
    // Patch generate function
    HeightmapGenerator.generate = async function(graph) {
        console.log("=== HEIGHTMAP GENERATION START ===");
        console.log("Graph dimensions:", graph.cellsX, "x", graph.cellsY);
        console.log("Cells desired:", graph.cellsDesired);
        console.log("Seed:", seed);
        
        trackingEnabled = true;
        window.fmgDebug.randomLog = [];
        
        const result = await origGenerate.call(this, graph);
        
        trackingEnabled = false;
        
        console.log("=== HEIGHTMAP GENERATION END ===");
        console.log("Random calls made:", window.fmgDebug.randomLog.length);
        console.log("First 10 randoms:", window.fmgDebug.randomLog.slice(0, 10));
        
        // Analyze result
        try {
            const heights = HeightmapGenerator.getHeights();
            if (heights && heights.length > 0) {
                console.log("Heights analysis:");
                console.log("  Min:", Math.min(...heights));
                console.log("  Max:", Math.max(...heights));
                console.log("  Mean:", heights.reduce((a,b) => a+b) / heights.length);
                console.log("  Land cells:", heights.filter(h => h >= 20).length);
                console.log("  First 20:", Array.from(heights.slice(0, 20)));
            } else {
                console.log("Heights not available yet");
            }
        } catch (e) {
            console.log("Error analyzing heights:", e.message);
        }
        
        return result;
    };
    
    // Patch fromTemplate to log template execution
    HeightmapGenerator.fromTemplate = function(graph, templateId) {
        console.log("Executing template:", templateId);
        const template = heightmapTemplates[templateId];
        console.log("Template string:", template.template);
        
        window.fmgDebug.heightmapLog = [];
        
        // Store heights before
        const heightsBefore = HeightmapGenerator.getHeights();
        console.log("Heights before template:", heightsBefore ? Array.from(heightsBefore.slice(0, 10)) : "null");
        
        const result = origFromTemplate.call(this, graph, templateId);
        
        // Log final state
        console.log("Template execution complete");
        
        return result;
    };
    
    // Patch addHill to log parameters and results
    HeightmapGenerator.addHill = function(count, height, rangeX, rangeY) {
        const heights = HeightmapGenerator.getHeights();
        const beforeNonZero = heights.filter(h => h > 0).length;
        
        console.log(`addHill(${count}, ${height}, ${rangeX}, ${rangeY})`);
        console.log("  Random log size before:", window.fmgDebug.randomLog.length);
        
        // Log the actual random values used
        const randomsBefore = window.fmgDebug.randomLog.length;
        
        const result = origAddHill.call(this, count, height, rangeX, rangeY);
        
        const randomsUsed = window.fmgDebug.randomLog.length - randomsBefore;
        const afterNonZero = heights.filter(h => h > 0).length;
        
        console.log("  Randoms used:", randomsUsed);
        console.log("  Non-zero cells: before =", beforeNonZero, ", after =", afterNonZero);
        console.log("  Cells affected:", afterNonZero - beforeNonZero);
        
        // Log the specific random values used for this hill
        if (randomsUsed > 0) {
            console.log("  First few randoms:", window.fmgDebug.randomLog.slice(randomsBefore, randomsBefore + 5));
        }
        
        window.fmgDebug.hillLog.push({
            count, height, rangeX, rangeY,
            cellsAffected: afterNonZero - beforeNonZero,
            randomsUsed: randomsUsed
        });
        
        return result;
    };
    
    // Also try to check getLinePower
    if (typeof getLinePower !== 'undefined') {
        try {
            console.log("Testing getLinePower():", getLinePower());
        } catch (e) {
            console.log("getLinePower() error:", e.message);
        }
    }
    
    // Check blob power
    if (typeof getBlobPower !== 'undefined') {
        try {
            console.log("getBlobPower(10000):", getBlobPower(10000));
        } catch (e) {
            console.log("getBlobPower() might be internal");
        }
    }
    
    console.log("Debug patches installed successfully");
    
    // Export debug data function
    window.exportFmgDebugData = function() {
        try {
            // Try multiple ways to get heights
            let heights = null;
            
            // Method 1: From grid.cells.h
            if (typeof grid !== 'undefined' && grid.cells && grid.cells.h) {
                heights = grid.cells.h;
                console.log("Got heights from grid.cells.h");
            }
            // Method 2: From HeightmapGenerator
            else if (typeof HeightmapGenerator !== 'undefined' && HeightmapGenerator.getHeights) {
                heights = HeightmapGenerator.getHeights();
                console.log("Got heights from HeightmapGenerator");
            }
            // Method 3: From pack.cells.h
            else if (typeof pack !== 'undefined' && pack.cells && pack.cells.h) {
                heights = pack.cells.h;
                console.log("Got heights from pack.cells.h");
            }
            
            if (!heights) {
                console.error("No heights data found!");
                return { error: "No heights data available" };
            }
            
            const heightsArray = Array.from(heights);
            
            return {
                seed: typeof seed !== 'undefined' ? seed : 'unknown',
                dimensions: { 
                    width: typeof graphWidth !== 'undefined' ? graphWidth : 'unknown',
                    height: typeof graphHeight !== 'undefined' ? graphHeight : 'unknown'
                },
                cells: { 
                    desired: (typeof grid !== 'undefined' && grid.cellsDesired) ? grid.cellsDesired : 10000,
                    actual: heightsArray.length,
                    x: (typeof grid !== 'undefined' && grid.cellsX) ? grid.cellsX : 'unknown',
                    y: (typeof grid !== 'undefined' && grid.cellsY) ? grid.cellsY : 'unknown'
                },
                heights: {
                    data: heightsArray,
                    min: Math.min(...heightsArray),
                    max: Math.max(...heightsArray),
                    mean: heightsArray.reduce((a,b) => a+b) / heightsArray.length,
                    landCells: heightsArray.filter(h => h >= 20).length,
                    waterCells: heightsArray.filter(h => h < 20).length,
                    first20: heightsArray.slice(0, 20)
                },
                debug: window.fmgDebug || {}
            };
        } catch (e) {
            console.error("Error in exportFmgDebugData:", e);
            return { error: e.message, stack: e.stack };
        }
    };
})();