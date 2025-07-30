// FINAL CORRECTED FMG JavaScript instrumentation for PRNG tracking
// This wraps the actual random functions that FMG uses: rand() and P()

// Add this code BEFORE any heightmap generation (e.g., at the top of heightmap-generator.js):

// Initialize counter
let random_call_count = 0;

// Wrap the rand() function that FMG actually uses
const original_rand = rand;
window.rand = function(...args) {
    random_call_count++;
    return original_rand(...args);
};

// Wrap the P() probability function as well
const original_P = P;
window.P = function(probability) {
    random_call_count++;
    return original_P(probability);
};

// Also wrap Math.random just in case it's used directly
const original_random = Math.random;
Math.random = function() {
    random_call_count++;
    return original_random.call(this);
};

// Then modify the fromTemplate function to add logging:
const fromTemplate = (graph, id) => {
    const templateString = heightmapTemplates[id]?.template || "";
    const steps = templateString.split("\n");

    if (!steps.length) throw new Error(`Heightmap template: no steps. Template: ${id}. Steps: ${steps}`);
    setGraph(graph);

    // Reset counter at start to get clean counts
    random_call_count = 0;
    
    // Add PRNG tracking initialization
    console.log("=== PRNG Call Tracking for Template ===");
    console.log(`Template: ${id}`);
    console.log(`Total steps: ${steps.length}`);
    console.log(`Initial PRNG calls: ${random_call_count}`);
    console.log("=====================================");

    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        // Track PRNG calls BEFORE executing the command
        const command_start_count = random_call_count;
        
        const elements = step.trim().split(" ");
        if (elements.length < 2) throw new Error(`Heightmap template: steps < 2. Template: ${id}. Step: ${elements}`);
        
        addStep(...elements);
        
        // Report PRNG usage AFTER executing the command
        const calls_for_command = random_call_count - command_start_count;
        console.log(`Step ${i + 1}: ${step}`);
        console.log(`  PRNG calls: ${calls_for_command}`);
        console.log(`  Total calls so far: ${random_call_count}`);
        
        // Calculate land percentage
        const land = heights.filter(h => h > 20).length;
        const landPct = (land / heights.length * 100).toFixed(1);
        console.log(`  Land: ${landPct}%`);
    }
    
    console.log("=====================================");
    console.log(`Total PRNG calls: ${random_call_count}`);
    
    return heights;  // Don't forget to return heights!
}