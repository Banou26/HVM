# HVM WebGPU Web Example

This example demonstrates running HVM programs on WebGPU in a web browser.

## Features

- **WebGPU Execution**: Run HVM programs on the GPU using WebGPU
- **Live Editor**: Edit and run HVM code directly in the browser
- **Performance Metrics**: See interactions, time, and MIPS
- **E2E Tests**: Full browser tests using Vitest + Playwright

## Requirements

- Node.js 18+
- A browser with WebGPU support:
  - Chrome 113+
  - Edge 113+
  - Firefox Nightly (with `dom.webgpu.enabled` flag)

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:3000 in your browser
```

## Running Tests

```bash
# Run browser tests (requires Playwright browsers)
npx playwright install chromium
npm run test:browser

# Run tests with UI
npm run test:ui
```

## Project Structure

```
examples/web/
├── index.html          # Main demo page
├── src/
│   ├── main.js         # Demo application
│   └── hvm-runtime.js  # HVM WebGPU runtime
├── tests/
│   └── hvm-webgpu.test.js  # E2E browser tests
├── package.json
├── vite.config.js
└── vitest.config.ts
```

## HVM Runtime API

### Basic Usage

```javascript
import { runHVM } from './src/hvm-runtime.js';

const code = `
@main = a
  & @sum ~ (16 a)

@sum = (?(((* 1) @sum__C0) a) a)
@sum__C0 = ({a c} ((b d) f))
  &! @sum ~ (a (b $([+] $(e f))))
  &! @sum ~ (c (d e))
`;

const result = await runHVM(code, {
  workgroups: 32,
  nodeLen: 1 << 18,
  varsLen: 1 << 18,
  maxIterations: 500
});

console.log('Interactions:', result.interactions);
console.log('Time:', result.time, 's');
console.log('MIPS:', result.mips);
```

### Advanced Usage

```javascript
import { HVMRuntime, compileHVM } from './src/hvm-runtime.js';

// Compile HVM code to a book buffer
const bookData = compileHVM(hvmCode);

// Create and initialize runtime
const runtime = new HVMRuntime();
await runtime.init();

// Run the compiled program
const result = await runtime.run(bookData, config);

// Clean up
runtime.destroy();
```

## Using with Bend

[Bend](https://github.com/HigherOrderCO/Bend) is a high-level functional programming language that compiles to HVM. You can compile Bend programs and run them in the browser:

```bash
# Compile Bend to HVM
bend compile my_program.bend -o my_program.hvm

# Generate WebGPU code
hvm gen-wgpu my_program.hvm > my_program.js
```

Then load the generated code in your web application.

## Limitations

- **Memory**: WebGPU has limited GPU memory compared to CUDA
- **Atomics**: WebGPU atomic operations are more limited than CUDA
- **I/O**: No direct I/O support in the WebGPU backend (computation only)
- **Debugging**: Limited debugging capabilities compared to native execution

## Troubleshooting

### "WebGPU is not supported"

Make sure you're using a supported browser:
- Chrome 113+ (recommended)
- Edge 113+
- Firefox Nightly with `dom.webgpu.enabled` flag

### "Failed to get GPU adapter"

This can happen if:
- Your system doesn't have a compatible GPU
- GPU drivers are outdated
- Running in a virtual machine without GPU passthrough

### Tests failing in CI

WebGPU tests may fail in headless CI environments. The tests are designed to gracefully skip if WebGPU is not available.

## License

MIT - See the main HVM repository for full license details.
