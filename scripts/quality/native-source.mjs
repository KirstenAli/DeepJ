import assert from 'node:assert/strict';

export function splitNativeSource(source, file) {
  const pattern = /@R"\(([^]*?)\)"/g;
  const shaders = [];
  const host = source.replace(pattern, (match, text, offset) => {
    const line = source.slice(0, offset).split('\n').length;
    shaders.push({ source: '\n'.repeat(line - 1) + text, file });
    return '@""' + '\n'.repeat(match.split('\n').length - 1);
  });
  assert.ok(shaders.length > 0, 'No embedded Metal shader source found');
  assert.ok(!host.includes('kernel void'), 'A Metal shader block was not extracted');
  return { host, shaders };
}

export function analysisSource(source, file, namespace) {
  const headers = [];
  const body = source.replace(/^#include[^\n]*$/gm, (line) => {
    headers.push(line);
    return '';
  });
  return [
    ...headers,
    `namespace ${namespace} {`,
    `#line 1 ${JSON.stringify(file)}`,
    body,
    '}',
  ].join('\n');
}
