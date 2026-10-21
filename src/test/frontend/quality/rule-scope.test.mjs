import assert from 'node:assert/strict';
import test from 'node:test';
import { ESLint } from 'eslint';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { nativeFiles } from '../../../../scripts/quality/native-files.mjs';

const checkedRules = [
  'complexity',
  'sonarjs/cognitive-complexity',
  'max-depth',
  'max-lines-per-function',
  'max-statements',
];

for (const path of [
  'src/test/frontend/quality/rule-scope.test.mjs',
  'src/test/native/complexity.test.mjs',
]) {
  test('frontend quality rules also apply to ' + path, async () => {
    const eslint = new ESLint();
    const production = await eslint.calculateConfigForFile('src/main/resources/chat.js');
    const tests = await eslint.calculateConfigForFile(path);
    assert.equal(await eslint.isPathIgnored(path), false);
    for (const rule of checkedRules) assert.deepEqual(tests.rules[rule], production.rules[rule]);
  });
}

test('native analysis discovers both test headers, not just production Metal', () => {
  const files = nativeFiles().map((file) => file.replaceAll('\\', '/'));
  assert.ok(files.includes('native/deepj_metal_jni.mm'));
  assert.ok(files.includes('src/test/native/coverage-hooks.h'));
  assert.ok(files.includes('src/test/native/metal-analysis-shim.h'));
});

test('native analysis discovers nested test sources and ignores binary artifacts', (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'deepj-native-scope-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const nested = join(directory, 'tests', 'helpers');
  mkdirSync(nested, { recursive: true });
  const files = ['a.cpp', 'b.h', 'c.metal'].map((name) => join(nested, name));
  for (const file of [...files, join(nested, 'library.dylib')]) writeFileSync(file, '');
  assert.deepEqual(nativeFiles([directory]), files);
});
