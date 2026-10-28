import assert from 'node:assert/strict';
import process from 'node:process';
import test from 'node:test';
import { analyseNativeFile } from '../../../scripts/quality/native-analysis.mjs';

test('all three compact matrix shader variants undergo native code-quality analysis', () => {
  const java = process.env.JAVA_HOME;
  assert.ok(java, 'Pass JAVA_HOME for the host JNI declarations');
  const scopes = analyseNativeFile('native/deepj_metal_jni.mm', 980, [
    '-I' + java + '/include',
    '-I' + java + '/include/darwin',
  ]);
  const compact = scopes.filter(({ metrics }) =>
    metrics.some(({ name }) => name === 'kernel_compact_matmul'),
  );
  assert.equal(compact.length, 3);
  for (const { metrics } of compact) {
    assert.ok(metrics.some(({ name }) => name === 'compact_accumulate'));
    assert.ok(metrics.some(({ name }) => name === 'compact_read'));
  }
  assertPreciseNormalization(scopes);
});

function assertPreciseNormalization(scopes) {
  const probability = scopes
    .flatMap(({ metrics }) => metrics)
    .find(({ name }) => name === 'layout_probability');
  assert.ok(probability, 'Precise compact normalization remains subject to native analysis');
  assert.equal(probability.cognitive, 1);
}
