import assert from 'node:assert/strict';

export function nativeFunctionLines(fn, source) {
  const begin = fn.range.begin.expansionLoc ?? fn.range.begin;
  const end = fn.range.end.expansionLoc ?? fn.range.end;
  const finish = end.offset + end.tokLen;
  assert.ok(
    Number.isSafeInteger(begin.offset) && Number.isSafeInteger(finish),
    'Missing Clang source range',
  );
  assert.ok(
    begin.offset >= 0 && finish > begin.offset && finish <= source.length,
    'Invalid Clang source range',
  );
  return source
    .subarray(begin.offset, finish)
    .toString('utf8')
    .split('\n')
    .filter((line) => line.trim()).length;
}
