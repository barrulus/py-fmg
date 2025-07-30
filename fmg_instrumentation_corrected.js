// CORRECTED FMG JavaScript instrumentation for PRNG tracking
// This accounts for FMG's reassignment of Math.random to use Alea PRNG

// Method 1: Modify the generate() function in heightmap-generator.js
// Find this line (around line 68):
//   Math.random = aleaPRNG(seed);
// 
// Add immediately AFTER that line:
let random_call_count = 0;
const fmg_random = Math.random;  // This is now the Alea PRNG
Math.random = function() { 
    random_call_count++; 
    return fmg_random.call(this);
}

// Method 2: If you can't modify heightmap-generator.js directly,
// you can intercept aleaPRNG instead:
// At the TOP of the file (before any functions), add:
const original_aleaPRNG = aleaPRNG;
let random_call_count = 0;

window.aleaPRNG = function(seed) {
    const prng = original_aleaPRNG(seed);
    // Return a wrapped version that counts calls
    return function() {
        random_call_count++;
        return prng();
    };
};

// Then modify the fromTemplate function to add logging:
const fromTemplate = (graph, id) => {
    const templateString = heightmapTemplates[id]?.template || "";
    const steps = templateString.split("\n");

    if (!steps.length) throw new Error(`Heightmap template: no steps. Template: ${id}. Steps: ${steps}`);
    setGraph(graph);

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