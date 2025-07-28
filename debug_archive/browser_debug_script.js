// JavaScript debugging script to inject into FMG browser console
// This will help us track the exact generation process

console.log("🔍 FMG Debugging Script Loaded");

// Override Math.random to track consumption
let randomCallCount = 0;
const originalRandom = Math.random;
Math.random = function() {
    const value = originalRandom();
    randomCallCount++;
    if (randomCallCount <= 20) {
        console.log(`Random ${randomCallCount}: ${value.toFixed(10)}`);
    }
    return value;
};

// Track heightmap generation
if (window.HeightmapGenerator) {
    const originalGenerate = window.HeightmapGenerator.generate;
    window.HeightmapGenerator.generate = async function(graph) {
        console.log("🏔️ HeightmapGenerator.generate() called");
        console.log("Graph cells:", graph.points.length);
        console.log("Seed:", seed);
        
        const result = await originalGenerate.call(this, graph);
        
        console.log("Heights generated:");
        console.log("  Range:", Math.min(...result), "-", Math.max(...result));
        console.log("  Mean:", result.reduce((a,b) => a+b) / result.length);
        console.log("  Land cells:", result.filter(h => h >= 20).length);
        
        return result;
    };
}

// Track template execution
if (window.HeightmapGenerator && window.HeightmapGenerator.fromTemplate) {
    const originalFromTemplate = window.HeightmapGenerator.fromTemplate;
    window.HeightmapGenerator.fromTemplate = function(graph, templateId) {
        console.log("📜 Executing template:", templateId);
        console.log("Template content:", heightmapTemplates[templateId]?.template);
        
        const result = originalFromTemplate.call(this, graph, templateId);
        return result;
    };
}

console.log("✅ FMG debugging hooks installed");