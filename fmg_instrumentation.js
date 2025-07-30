// FMG JavaScript instrumentation for PRNG tracking
// Add this to your FMG heightmap-generator.js file

// 1. Add at the top of the file:
let random_call_count = 0;
let command_start_count = 0;
const original_random = Math.random;
Math.random = function() { 
    random_call_count++; 
    return original_random.call(this);
}

// 2. In the fromTemplate function, replace the command execution loop with:
function fromTemplate(template) {
    const commands = template.trim().split("\n");
    console.log("=== PRNG Call Tracking for Template ===");
    console.log(`Total commands: ${commands.length}`);
    console.log(`Initial PRNG calls: ${random_call_count}`);
    console.log("=====================================");
    
    commands.forEach((command, i) => {
        command_start_count = random_call_count;
        
        // Parse command
        const [cmd, ...args] = command.trim().split(/\s+/);
        
        // Execute command (existing code)
        if (cmd === "Hill") addHill(...args);
        else if (cmd === "Pit") addPit(...args);
        else if (cmd === "Range") addRange(...args);
        else if (cmd === "Trough") addTrough(...args);
        else if (cmd === "Strait") addStrait(...args);
        else if (cmd === "Smooth") smooth(...args);
        else if (cmd === "Mask") mask(...args);
        else if (cmd === "Add") add(...args);
        else if (cmd === "Multiply") multiply(...args);
        else if (cmd === "Invert") invert(...args);
        
        // Report PRNG usage
        const calls_for_command = random_call_count - command_start_count;
        console.log(`Step ${i + 1}: ${command}`);
        console.log(`  PRNG calls: ${calls_for_command}`);
        console.log(`  Total calls so far: ${random_call_count}`);
        
        // Calculate land percentage
        const land = heights.filter(h => h > 20).length;
        const landPct = (land / heights.length * 100).toFixed(1);
        console.log(`  Land: ${landPct}%`);
    });
    
    console.log("=====================================");
    console.log(`Total PRNG calls: ${random_call_count}`);
}

// 3. Test with isthmus template:
// Run in browser console:
// - Set seed to "654321"
// - Generate map with isthmus template
// - Check console for PRNG call counts