import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import {
  javaMetrics,
  kernelMetrics,
  percent,
  requireThreshold,
  selectPackages,
} from '../../../scripts/coverage/metrics.mjs';
import { instrumentNative } from '../../../scripts/coverage/native-source.mjs';

test('coverage gates reject low, missing, invalid and empty measurements', () => {
  assert.equal(percent(90, 100), 90);
  assert.match(requireThreshold('lines', 90, 85), /90.00%/);
  assert.throws(() => requireThreshold('lines', 84.99, 85));
  assert.throws(() => requireThreshold('lines', Number.NaN, 85));
  assert.throws(() => percent(0, 0));
  assert.throws(() => percent(101, 100));
  assert.throws(() => percent(-1, 100));
});

test('Java metrics aggregate actual counts rather than averaging package percentages', () => {
  const xml =
    '<report><package name="a"><counter type="LINE" covered="9" missed="1"/>' +
    '<counter type="BRANCH" covered="8" missed="2"/></package>' +
    '<package name="b"><counter type="LINE" covered="1" missed="89"/>' +
    '<counter type="BRANCH" covered="2" missed="8"/></package></report>';
  const document = new JSDOM(xml, { contentType: 'text/xml' }).window.document;
  assert.deepEqual(javaMetrics(selectPackages(document)), { LINE: 10, BRANCH: 50 });
  assert.deepEqual(javaMetrics(selectPackages(document, ['b'])), { LINE: 90, BRANCH: 80 });
});

test('kernel selection coverage reports missing shaders and ignores duplicate selections', () => {
  const source = 'kernel void kernel_a() {} kernel void kernel_b() {}';
  assert.deepEqual(kernelMetrics(source, 'kernel_a\nkernel_a\n'), {
    total: 2,
    covered: 1,
    missing: ['kernel_b'],
  });
  assert.deepEqual(kernelMetrics(source, 'kernel_b\nkernel_a\n'), {
    total: 2,
    covered: 2,
    missing: [],
  });
});

test('native instrumentation touches only a generated copy and instruments every pipeline selection', () => {
  const source = readFileSync('native/deepj_metal_jni.mm', 'utf8');
  const hooks = readFileSync('src/test/native/coverage-hooks.h', 'utf8');
  const result = instrumentNative(source, hooks);
  const selections = [...source.matchAll(/\[encoder setComputePipelineState:/g)].length;
  assert.equal([...result.matchAll(/selectCoveragePipeline\(encoder,/g)].length, selections);
  assert.ok(result.includes('registerCoveragePipeline(pso, name);'));
  assert.ok(source.includes('    return pso;'));
  assert.throws(() => instrumentNative('', hooks));
  assert.throws(() => instrumentNative('    return pso;', hooks));
  assert.throws(() => instrumentNative('static MetalContext* gCtx;\n    return pso;', hooks));
});
