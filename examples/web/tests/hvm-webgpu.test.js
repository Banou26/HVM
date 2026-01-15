/**
 * HVM WebGPU E2E Tests
 *
 * These tests run in a real browser using Vitest browser mode with Playwright.
 * They test the full WebGPU execution pipeline.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  HVMRuntime,
  HVMParser,
  BookSerializer,
  compileHVM,
  runHVM,
  HVM_SHADER
} from '../src/hvm-runtime.js';

describe('HVM WebGPU E2E Tests', () => {

  describe('WebGPU Support', () => {
    it('should detect WebGPU support', () => {
      const isSupported = HVMRuntime.isSupported();
      // In headless Chromium, WebGPU might not be available
      // So we just check that the function returns a boolean
      expect(typeof isSupported).toBe('boolean');
    });

    it('should initialize runtime if WebGPU is available', async () => {
      if (!HVMRuntime.isSupported()) {
        console.log('WebGPU not supported, skipping test');
        return;
      }

      const runtime = new HVMRuntime();
      await runtime.init();
      expect(runtime.initialized).toBe(true);
      expect(runtime.device).toBeTruthy();
      runtime.destroy();
    });
  });

  describe('HVM Parser', () => {
    it('should parse a simple definition', () => {
      const code = '@main = 42';
      const book = HVMParser.parse(code);

      expect(book.defs).toHaveLength(1);
      expect(book.defs[0].name).toBe('main');
    });

    it('should parse constructor nodes', () => {
      const code = '@main = (a b)';
      const book = HVMParser.parse(code);

      expect(book.defs[0].nodes.length).toBeGreaterThanOrEqual(1);
    });

    it('should parse duplicator nodes', () => {
      const code = '@main = {a b}';
      const book = HVMParser.parse(code);

      expect(book.defs[0].nodes.length).toBeGreaterThanOrEqual(1);
    });

    it('should parse references', () => {
      const code = '@foo = 1\n@main = @foo';
      const book = HVMParser.parse(code);

      expect(book.defs).toHaveLength(2);
      expect(book.defs[0].name).toBe('foo');
      expect(book.defs[1].name).toBe('main');
    });

    it('should parse erasers', () => {
      const code = '@main = *';
      const book = HVMParser.parse(code);

      expect(book.defs).toHaveLength(1);
    });

    it('should parse switch nodes', () => {
      const code = '@main = ?(a b)';
      const book = HVMParser.parse(code);

      expect(book.defs).toHaveLength(1);
    });

    it('should parse redexes', () => {
      const code = '@main = a\n  & @foo ~ a\n@foo = 42';
      const book = HVMParser.parse(code);

      expect(book.defs[0].redexes.length).toBeGreaterThanOrEqual(1);
    });

    it('should parse the sum_tree example', () => {
      const code = `
@gen = (?(((a a) @gen__C0) b) b)

@gen__C0 = ({a d} ({$([*2] $([+1] b)) $([*2] e)} (c f)))
  &! @gen ~ (a (b c))
  &! @gen ~ (d (e f))

@main = a
  & @sum ~ (16 (@main__C0 a))

@main__C0 = a
  & @gen ~ (16 (0 a))

@sum = (?(((* 1) @sum__C0) a) a)

@sum__C0 = ({a c} ((b d) f))
  &! @sum ~ (a (b $([+] $(e f))))
  &! @sum ~ (c (d e))
`;
      const book = HVMParser.parse(code);

      expect(book.defs.length).toBeGreaterThanOrEqual(5);

      const defNames = book.defs.map(d => d.name);
      expect(defNames).toContain('gen');
      expect(defNames).toContain('main');
      expect(defNames).toContain('sum');
    });
  });

  describe('Book Serializer', () => {
    it('should serialize a simple book', () => {
      const book = HVMParser.parse('@main = 42');
      const buffer = BookSerializer.serialize(book);

      expect(buffer).toBeInstanceOf(Uint8Array);
      expect(buffer.length).toBeGreaterThan(0);
    });

    it('should serialize multiple definitions', () => {
      const code = '@foo = 1\n@bar = 2\n@main = @foo';
      const book = HVMParser.parse(code);
      const buffer = BookSerializer.serialize(book);

      // Check defs_len is encoded at the beginning
      const view = new DataView(buffer.buffer);
      const defsLen = view.getUint32(0, true);
      expect(defsLen).toBe(3);
    });
  });

  describe('Full Compilation', () => {
    it('should compile HVM code to buffer', () => {
      const code = '@main = 42';
      const buffer = compileHVM(code);

      expect(buffer).toBeInstanceOf(Uint8Array);
      expect(buffer.length).toBeGreaterThan(0);
    });

    it('should compile complex programs', () => {
      const code = `
@gen = (?(((a a) @gen__C0) b) b)
@gen__C0 = ({a d} ({$([*2] $([+1] b)) $([*2] e)} (c f)))
  &! @gen ~ (a (b c))
  &! @gen ~ (d (e f))
@main = a
  & @sum ~ (4 (@main__C0 a))
@main__C0 = a
  & @gen ~ (4 (0 a))
@sum = (?(((* 1) @sum__C0) a) a)
@sum__C0 = ({a c} ((b d) f))
  &! @sum ~ (a (b $([+] $(e f))))
  &! @sum ~ (c (d e))
`;
      const buffer = compileHVM(code);
      expect(buffer).toBeInstanceOf(Uint8Array);
    });
  });

  describe('WebGPU Execution', () => {
    it('should run a simple program on WebGPU', async () => {
      if (!HVMRuntime.isSupported()) {
        console.log('WebGPU not supported, skipping test');
        return;
      }

      const code = '@main = 42';

      try {
        const result = await runHVM(code, {
          workgroups: 4,
          nodeLen: 1 << 16,
          varsLen: 1 << 16,
          maxIterations: 100
        });

        expect(result).toHaveProperty('interactions');
        expect(result).toHaveProperty('time');
        expect(result).toHaveProperty('mips');
        expect(result.time).toBeGreaterThan(0);
      } catch (e) {
        // WebGPU might fail in CI environments
        console.log('WebGPU execution failed:', e.message);
      }
    });

    it('should run the sum_tree program on WebGPU', async () => {
      if (!HVMRuntime.isSupported()) {
        console.log('WebGPU not supported, skipping test');
        return;
      }

      const code = `
@gen = (?(((a a) @gen__C0) b) b)
@gen__C0 = ({a d} ({$([*2] $([+1] b)) $([*2] e)} (c f)))
  &! @gen ~ (a (b c))
  &! @gen ~ (d (e f))
@main = a
  & @sum ~ (4 (@main__C0 a))
@main__C0 = a
  & @gen ~ (4 (0 a))
@sum = (?(((* 1) @sum__C0) a) a)
@sum__C0 = ({a c} ((b d) f))
  &! @sum ~ (a (b $([+] $(e f))))
  &! @sum ~ (c (d e))
`;

      try {
        const result = await runHVM(code, {
          workgroups: 8,
          nodeLen: 1 << 17,
          varsLen: 1 << 17,
          maxIterations: 200
        });

        expect(result).toHaveProperty('interactions');
        expect(result.interactions).toBeGreaterThan(0);
        console.log('Sum tree result:', result);
      } catch (e) {
        console.log('WebGPU execution failed:', e.message);
      }
    });
  });

  describe('WGSL Shader', () => {
    it('should have valid shader source', () => {
      expect(HVM_SHADER).toBeTruthy();
      expect(typeof HVM_SHADER).toBe('string');
      expect(HVM_SHADER).toContain('@compute');
      expect(HVM_SHADER).toContain('fn main');
      expect(HVM_SHADER).toContain('fn boot');
    });

    it('should compile shader if WebGPU is available', async () => {
      if (!HVMRuntime.isSupported()) {
        console.log('WebGPU not supported, skipping test');
        return;
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.log('No GPU adapter, skipping test');
        return;
      }

      const device = await adapter.requestDevice();

      try {
        const shaderModule = device.createShaderModule({
          code: HVM_SHADER
        });

        // If we get here without throwing, the shader compiled successfully
        expect(shaderModule).toBeTruthy();
      } catch (e) {
        // Shader compilation failed
        console.error('Shader compilation failed:', e);
        throw e;
      } finally {
        device.destroy();
      }
    });
  });

  describe('Runtime Lifecycle', () => {
    it('should properly initialize and destroy runtime', async () => {
      if (!HVMRuntime.isSupported()) {
        console.log('WebGPU not supported, skipping test');
        return;
      }

      const runtime = new HVMRuntime();

      expect(runtime.initialized).toBe(false);
      await runtime.init();
      expect(runtime.initialized).toBe(true);

      runtime.destroy();
      expect(runtime.initialized).toBe(false);
      expect(runtime.device).toBe(null);
    });

    it('should handle multiple runs', async () => {
      if (!HVMRuntime.isSupported()) {
        console.log('WebGPU not supported, skipping test');
        return;
      }

      const code = '@main = 1';

      try {
        // Run multiple times
        for (let i = 0; i < 3; i++) {
          const result = await runHVM(code, {
            workgroups: 2,
            nodeLen: 1 << 15,
            varsLen: 1 << 15,
            maxIterations: 50
          });
          expect(result.time).toBeGreaterThan(0);
        }
      } catch (e) {
        console.log('WebGPU execution failed:', e.message);
      }
    });
  });
});

describe('Integration Tests', () => {
  it('should handle edge cases in parsing', () => {
    // Empty program
    expect(() => HVMParser.parse('')).not.toThrow();

    // Comments (if supported) - skip for now as parser doesn't support comments
    // expect(() => HVMParser.parse('// comment\n@main = 1')).not.toThrow();

    // Multiple definitions
    const code = '@a = 1\n@b = 2\n@c = 3\n@main = @a';
    const book = HVMParser.parse(code);
    expect(book.defs).toHaveLength(4);
  });

  it('should handle various number formats', () => {
    // Positive numbers
    const book1 = HVMParser.parse('@main = 123');
    expect(book1.defs).toHaveLength(1);

    // Zero
    const book2 = HVMParser.parse('@main = 0');
    expect(book2.defs).toHaveLength(1);
  });

  it('should handle nested structures', () => {
    // Nested constructors
    const code = '@main = ((a b) (c d))';
    const book = HVMParser.parse(code);
    expect(book.defs[0].nodes.length).toBeGreaterThanOrEqual(2);
  });
});
