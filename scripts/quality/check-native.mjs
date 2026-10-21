import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import process from 'node:process';
import { clangAst } from './clang-ast.mjs';
import { nativeMetrics, nativeViolations } from './native-metrics.mjs';
import { splitNativeSource } from './native-source.mjs';

const file = 'native/deepj_metal_jni.mm';
const limits = JSON.parse(readFileSync('config/native-quality.json', 'utf8'));

function hostFlags() {
  const java = process.argv[2];
  assert.ok(java, 'Pass the JDK home to native code-quality checks');
  return ['-I' + resolve(java, 'include'), '-I' + resolve(java, 'include/darwin')];
}

function shaderMetrics(shader, index) {
  const shim = resolve('src/test/native/metal-analysis-shim.h');
  assert.equal(shader.source.match(/#include <metal_stdlib>/g)?.length, 1);
  const source = shader.source.replace('#include <metal_stdlib>', `#include "${shim}"`);
  const metrics = nativeMetrics(clangAst(source, shader.file, 'deepj_shader_quality_' + index));
  const declared = [...shader.source.matchAll(/\bkernel\s+void\s+(\w+)\s*\(/g)];
  assert.ok(declared.length > 0, 'No Metal kernel declarations found');
  for (const [, name] of declared) {
    assert.ok(
      metrics.some((fn) => fn.name === name),
      'Metal kernel was not analysed: ' + name,
    );
  }
  return metrics;
}

function check(scope, metrics) {
  assert.ok(metrics.length > 0, scope + ': no native function definitions analysed');
  const violations = nativeViolations(metrics, limits);
  process.stdout.write(
    `${scope}: ${metrics.length} functions analysed, ${violations.length} violations\n`,
  );
  return violations.map((message) => scope + ': ' + message);
}

function testHelperMetrics() {
  const file = 'src/test/native/coverage-hooks.h';
  const headers = [
    '#include <Foundation/Foundation.h>',
    '#include <Metal/Metal.h>',
    '#include <stdexcept>',
    '#line 1 ' + JSON.stringify(file),
  ].join('\n');
  const source = headers + '\n' + readFileSync(file, 'utf8');
  return nativeMetrics(clangAst(source, file, 'deepj_native_test_quality'));
}

function run() {
  const { host, shaders } = splitNativeSource(readFileSync(file, 'utf8'), file);
  const hostMetrics = nativeMetrics(clangAst(host, file, 'deepj_host_quality', hostFlags()));
  const shaderResults = shaders.flatMap(shaderMetrics);
  const helpers = testHelperMetrics();
  const violations = [
    ...check('Native host', hostMetrics),
    ...check('Metal shaders', shaderResults),
    ...check('Native test helpers', helpers),
  ];
  const report = { limits, host: hostMetrics, shaders: shaderResults, helpers, violations };
  writeFileSync('target/quality/native/report.json', JSON.stringify(report, null, 2) + '\n');
  assert.deepEqual(violations, [], violations.join('\n'));
}

assert.equal(process.platform, 'darwin', 'Native source analysis requires macOS SDK headers');
run();
