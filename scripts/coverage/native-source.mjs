import assert from 'node:assert/strict';

export function instrumentNative(source, hooks) {
  assert.ok(source.includes('    return pso;'), 'Cannot locate Metal pipeline creation');
  assert.ok(source.includes('static MetalContext* gCtx'), 'Cannot locate Metal context');
  const selections = [...source.matchAll(/\[encoder setComputePipelineState:([^\]]+)\];/g)];
  assert.ok(selections.length > 0, 'Cannot locate Metal pipeline selections');
  const instrumented = source
    .replaceAll(
      /\[encoder setComputePipelineState:([^\]]+)\];/g,
      'selectCoveragePipeline(encoder, $1);',
    )
    .replace('    return pso;', '    registerCoveragePipeline(pso, name);\n    return pso;');
  return instrumented.replace('static MetalContext* gCtx', hooks + '\nstatic MetalContext* gCtx');
}
