import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { clangAst } from '../../../scripts/quality/clang-ast.mjs';
import { nativeMetrics, nativeViolations } from '../../../scripts/quality/native-metrics.mjs';
import { analysisSource, splitNativeSource } from '../../../scripts/quality/native-source.mjs';
import { fixtures } from './complexity-fixtures.mjs';

const limits = JSON.parse(readFileSync('config/native-quality.json', 'utf8'));
const ast = clangAst(
  fixtures.map((item) => item.source).join('\n'),
  'fixtures.cpp',
  'deepj_quality_fixtures',
);
const measured = nativeMetrics(ast);

for (const expected of fixtures) {
  test('native control-flow metric: ' + expected.name, () => {
    const actual = measured.find((item) => item.name === expected.name);
    assert.ok(actual, 'Clang must analyse every fixture');
    assert.equal(actual.cognitive, expected.cognitive);
    assert.equal(actual.nesting, expected.nesting);
    assert.ok(Number.isInteger(actual.line));
  });
}

test('native limits reject 16 complexity and four levels, but accept their boundaries', () => {
  const names = nativeViolations(measured, limits).map((message) => message.split(':')[0]);
  assert.deepEqual(names.toSorted(), ['boundary16', 'fourLevels', 'unbraced'].toSorted());
});

test('native parse failures are errors, not successful empty reports', () => {
  assert.throws(
    () => clangAst('int broken() { if ( }', 'bad.cpp', 'deepj_invalid_quality'),
    /Native source analysis failed/,
  );
});

test('recursive C++ methods are scored, not only free functions', () => {
  const source =
    'struct Sample { int recurse(int x) { if (x) { return recurse(x - 1); } return 0; } };';
  const metrics = nativeMetrics(clangAst(source, 'method.cpp', 'deepj_method_quality'));
  assert.equal(metrics.find((fn) => fn.name === 'recurse').cognitive, 2);
});

test('shader extraction preserves original line numbers and empties only the host string', () => {
  const source = 'prefix\nstatic auto code = @R"(\nkernel void example() {}\n)";\nsuffix';
  const { host, shaders } = splitNativeSource(source, 'example.mm');
  assert.equal(host.split('\n').length, source.split('\n').length);
  assert.ok(host.includes('static auto code = @""'));
  assert.ok(!host.includes('kernel void'));
  assert.equal(shaders[0].source.split('\n')[2], 'kernel void example() {}');
});

test('multiple embedded shader blocks are all extracted', () => {
  const block = '@R"(#include <metal_stdlib>\nkernel void sample() {})"';
  assert.equal(splitNativeSource(block + ';\n' + block, 'multiple.mm').shaders.length, 2);
});

test('missing shader blocks fail closed', () => {
  assert.throws(() => splitNativeSource('no shader', 'empty.mm'), /No embedded Metal/);
});

test('invalid limits or missing native metrics cannot pass the gate', () => {
  assert.throws(() => nativeViolations([{ name: 'broken', cognitive: NaN, nesting: 0 }], limits));
  assert.throws(() => nativeViolations(measured, { cognitive: -1, nesting: 3 }));
});

test('analysis namespaces leave the control-flow body unchanged', () => {
  const body = 'int sample() { if (true) { return 1; } return 0; }';
  assert.ok(analysisSource('#include <cmath>\n' + body, 'sample.mm', 'sample').includes(body));
});
