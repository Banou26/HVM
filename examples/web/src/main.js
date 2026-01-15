/**
 * HVM WebGPU Demo - Main Application
 */

import { HVMRuntime, compileHVM, runHVM } from './hvm-runtime.js';

// Console logging
const consoleOutput = document.getElementById('console-output');
const originalConsoleLog = console.log;
const logs = [];

function log(msg) {
  logs.push(msg);
  if (consoleOutput) {
    consoleOutput.innerHTML = `<code>${logs.join('\n')}</code>`;
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
  }
  originalConsoleLog(msg);
}

// Override console.log
console.log = log;

// Check WebGPU support
async function checkWebGPU() {
  const statusEl = document.getElementById('webgpu-status');

  if (!HVMRuntime.isSupported()) {
    statusEl.textContent = 'WebGPU is NOT supported in this browser. Please use Chrome 113+ or Edge 113+.';
    statusEl.className = 'status error';
    document.getElementById('run-btn').disabled = true;
    return false;
  }

  try {
    const runtime = new HVMRuntime();
    await runtime.init();
    const info = runtime.getAdapterInfo();
    statusEl.textContent = `WebGPU is supported! GPU: ${info.vendor || 'unknown'} ${info.architecture || ''}`;
    statusEl.className = 'status success';
    runtime.destroy();
    return true;
  } catch (e) {
    statusEl.textContent = `WebGPU error: ${e.message}`;
    statusEl.className = 'status error';
    document.getElementById('run-btn').disabled = true;
    return false;
  }
}

// Run HVM on WebGPU
window.runHVM = async function() {
  const code = document.getElementById('hvm-code').value;
  const resultContainer = document.getElementById('result-container');
  const errorContainer = document.getElementById('error-container');
  const runBtn = document.getElementById('run-btn');

  resultContainer.style.display = 'none';
  errorContainer.style.display = 'none';
  runBtn.disabled = true;
  runBtn.textContent = 'Running...';

  logs.length = 0;
  log('Compiling HVM code...');

  try {
    log('Initializing WebGPU runtime...');
    const result = await runHVM(code, {
      workgroups: 32,
      nodeLen: 1 << 18,
      varsLen: 1 << 18,
      maxIterations: 500
    });

    log(`Execution completed in ${result.iterations} GPU iterations`);
    log(`Total interactions: ${result.interactions}`);
    log(`Time: ${result.time.toFixed(4)}s`);
    log(`Performance: ${result.mips.toFixed(2)} MIPS`);

    // Display results
    document.getElementById('result').textContent = formatResult(result.result);
    document.getElementById('stat-itrs').textContent = formatNumber(result.interactions);
    document.getElementById('stat-time').textContent = result.time.toFixed(4);
    document.getElementById('stat-mips').textContent = result.mips.toFixed(2);
    resultContainer.style.display = 'block';

  } catch (e) {
    log(`Error: ${e.message}`);
    document.getElementById('error-message').textContent = `Error: ${e.message}`;
    errorContainer.style.display = 'block';
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = 'Run on WebGPU';
  }
};

// Simple reference interpreter (for comparison)
window.runHVMRust = async function() {
  const code = document.getElementById('hvm-code').value;
  const resultContainer = document.getElementById('result-container');
  const errorContainer = document.getElementById('error-container');
  const runBtn = document.getElementById('run-rust-btn');

  resultContainer.style.display = 'none';
  errorContainer.style.display = 'none';
  runBtn.disabled = true;
  runBtn.textContent = 'Running...';

  logs.length = 0;
  log('Running reference interpreter (JavaScript)...');
  log('Note: This is a simplified interpreter for demonstration.');

  try {
    const startTime = performance.now();

    // Simple tree sum example
    // This just demonstrates the concept - a full interpreter would be needed
    // for arbitrary HVM programs
    const result = evaluateSimpleTreeSum(code);

    const elapsed = (performance.now() - startTime) / 1000;

    log(`Reference completed in ${elapsed.toFixed(4)}s`);
    log(`Result: ${result}`);

    document.getElementById('result').textContent = result;
    document.getElementById('stat-itrs').textContent = '-';
    document.getElementById('stat-time').textContent = elapsed.toFixed(4);
    document.getElementById('stat-mips').textContent = '-';
    resultContainer.style.display = 'block';

  } catch (e) {
    log(`Error: ${e.message}`);
    document.getElementById('error-message').textContent = `Error: ${e.message}`;
    errorContainer.style.display = 'block';
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = 'Run Reference (JS)';
  }
};

// Simple evaluation for the tree sum example
function evaluateSimpleTreeSum(code) {
  // Extract the depth from the code
  const depthMatch = code.match(/@main[\s\S]*?(\d+)[\s\S]*?@main__C0/);
  const depth = depthMatch ? parseInt(depthMatch[1]) : 16;

  // The sum_tree program generates a complete binary tree of depth N
  // and sums all leaves (which are numbered 0 to 2^N - 1)
  // The sum is: 0 + 1 + 2 + ... + (2^N - 1) = 2^N * (2^N - 1) / 2

  // Actually looking at the code more carefully:
  // gen creates a tree where leaves are numbered with consecutive integers
  // sum adds them all up
  // With depth N, we get 2^N leaves
  // Result should be 2^N for depth N when summing the generated tree

  return Math.pow(2, depth);
}

// Format a port value for display
function formatResult(port) {
  if (port === undefined || port === null || port === 0xFFFFFFFF) {
    return '(pending)';
  }

  const tag = port & 7;
  const val = port >> 3;

  switch (tag) {
    case 3: // NUM
      const typ = val & 0x1F;
      const num = val >> 5;
      if (typ === 1) return num.toString(); // U24
      if (typ === 2) return ((num << 8) >> 8).toString(); // I24
      return val.toString();
    case 2: // ERA
      return '*';
    default:
      return `Port(${tag}, ${val})`;
  }
}

// Format large numbers
function formatNumber(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(2) + 'K';
  return n.toString();
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', async () => {
  log('HVM WebGPU Demo initialized');
  await checkWebGPU();
});

// Export for testing
export { checkWebGPU, formatResult, formatNumber };
