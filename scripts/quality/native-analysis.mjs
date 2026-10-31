import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { extname, resolve } from 'node:path';
import { clangAst } from './clang-ast.mjs';
import { nativeMetrics } from './native-metrics.mjs';
import { splitNativeSource } from './native-source.mjs';

function hostMetrics(source, file, name, flags) {
  const headers = [
    '#include <Foundation/Foundation.h>',
    '#include <Metal/Metal.h>',
    '#include <stdexcept>',
    '#line 1 ' + JSON.stringify(file),
  ].join('\n');
  return nativeMetrics(clangAst(headers + '\n' + source, file, name, flags));
}

function shaderMetrics(shader, name) {
  const shim = resolve('src/test/native/metal-analysis-shim.h');
  assert.equal(shader.source.match(/#include <metal_stdlib>/g)?.length, 1);
  const source = shader.source
    .replace('#include <metal_stdlib>', `#include "${shim}"`)
    .replace('#include <metal_simdgroup_matrix>', '');
  const metrics = nativeMetrics(clangAst(source, shader.file, name));
  for (const [, kernel] of shader.source.matchAll(/\bkernel\s+void\s+(\w+)\s*\(/g)) {
    assert.ok(
      metrics.some((fn) => fn.name === kernel),
      'Metal kernel was not analysed: ' + kernel,
    );
  }
  return metrics;
}

export function analyseNativeFile(file, index, flags) {
  const source = readFileSync(file, 'utf8');
  const name = 'deepj_source_quality_' + index;
  if (extname(file) === '.metal') {
    return [{ file, kind: 'shaders', metrics: shaderMetrics({ source, file }, name) }];
  }
  const parts = source.includes('@R"(')
    ? splitNativeSource(source, file)
    : { host: source, shaders: [] };
  const host = { file, kind: 'host', metrics: hostMetrics(parts.host, file, name, flags) };
  const shaders = parts.shaders.flatMap(shaderVariants).map((shader, number) => ({
    file,
    kind: 'shaders',
    metrics: shaderMetrics(shader, name + '_shader_' + number),
  }));
  return [host, ...shaders];
}

function shaderVariants(shader) {
  if (!shader.source.includes('#if COMPACT_KIND == 2')) return [shader];
  return [1, 2, 3].map((kind) => ({
    ...shader,
    source: '#define COMPACT_KIND ' + kind + '\n' + shader.source,
  }));
}
