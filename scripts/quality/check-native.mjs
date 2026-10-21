import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import process from 'node:process';
import { analyseNativeFile } from './native-analysis.mjs';
import { nativeFiles } from './native-files.mjs';
import { nativeViolations } from './native-metrics.mjs';

const limits = JSON.parse(readFileSync('config/native-quality.json', 'utf8'));

function hostFlags() {
  const java = process.argv[2];
  assert.ok(java, 'Pass the JDK home to native code-quality checks');
  return ['-I' + resolve(java, 'include'), '-I' + resolve(java, 'include/darwin')];
}

function check({ file, kind, metrics }) {
  const violations = nativeViolations(metrics, limits);
  const scope = file + ' (' + kind + ')';
  process.stdout.write(
    `${scope}: ${metrics.length} functions analysed, ${violations.length} violations\n`,
  );
  return violations.map((message) => scope + ': ' + message);
}

function run() {
  const flags = hostFlags();
  const files = nativeFiles();
  assert.ok(files.length > 0, 'No native source files discovered');
  const scopes = files.flatMap((file, index) => analyseNativeFile(file, index, flags));
  const functions = scopes.flatMap((scope) => scope.metrics);
  assert.ok(functions.length > 100, 'Native analysis found too few function definitions');
  const violations = scopes.flatMap(check);
  const report = { limits, files: scopes, violations };
  writeFileSync('target/quality/native/report.json', JSON.stringify(report, null, 2) + '\n');
  assert.deepEqual(violations, [], violations.join('\n'));
}

assert.equal(process.platform, 'darwin', 'Native source analysis requires macOS SDK headers');
run();
