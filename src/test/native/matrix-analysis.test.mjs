import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { analyseNativeFile } from '../../../scripts/quality/native-analysis.mjs';

const source = `#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
template<uint Size>
inline void matrix_copy(device float* values) {
    for (uint offset = 0; offset < Size; offset += 8) {
        simdgroup_float8x8 block;
        simdgroup_load(block, values + offset, 64);
        simdgroup_store(block, values + offset, 64);
    }
}
kernel void matrix_kernel(device float* values [[buffer(0)]]) {
    matrix_copy<64>(values);
}`;

test('SIMD matrix shader helpers remain covered by native complexity analysis', () => {
  const directory = mkdtempSync(join(tmpdir(), 'deepj-matrix-analysis-'));
  try {
    const file = join(directory, 'fixture.metal');
    writeFileSync(file, source);
    const [{ metrics }] = analyseNativeFile(file, 0, []);
    assert.ok(metrics.some((fn) => fn.name === 'matrix_kernel'));
    assert.equal(metrics.find((fn) => fn.name === 'matrix_copy').cognitive, 1);
    assert.equal(metrics.find((fn) => fn.name === 'matrix_copy').nesting, 1);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
