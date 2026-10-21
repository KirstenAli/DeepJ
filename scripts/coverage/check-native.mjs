import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import process from 'node:process';
import { kernelMetrics, percent, requireThreshold } from './metrics.mjs';

const directory = resolve('target/coverage/native');
const library = resolve('target/coverage/test-classes/native/macos/libdeepj_metal_jni.dylib');
const config = JSON.parse(readFileSync('config/coverage.json', 'utf8'));
const profile = resolve(directory, 'coverage.profdata');

function mergeProfiles() {
  const files = readdirSync(directory).filter((file) => file.endsWith('.profraw'));
  assert.ok(files.length > 0, 'Missing native execution profiles; Metal tests must actually run');
  execFileSync('xcrun', [
    'llvm-profdata',
    'merge',
    '-sparse',
    ...files.map((file) => resolve(directory, file)),
    '-o',
    profile,
  ]);
}

function report() {
  const source = resolve(directory, 'deepj_metal_coverage.mm');
  const args = ['-instr-profile=' + profile, library, source];
  const text = execFileSync('xcrun', ['llvm-cov', 'export', ...args], { encoding: 'utf8' });
  writeFileSync(resolve(directory, 'coverage.json'), text);
  execFileSync('xcrun', [
    'llvm-cov',
    'show',
    ...args,
    '-format=html',
    '-output-dir=' + resolve(directory, 'html'),
  ]);
  const measured = JSON.parse(text).data[0].files.find((file) => file.filename === source);
  assert.ok(measured, 'Native source coverage is missing');
  return measured.summary;
}

function checkTotals(totals) {
  for (const name of ['lines', 'branches', 'functions']) {
    const metric = totals[name];
    const value = percent(metric.covered, metric.count);
    process.stdout.write(requireThreshold(`native ${name}`, value, config.native[name]) + '\n');
  }
}

function checkKernels() {
  const source = readFileSync('native/deepj_metal_jni.mm', 'utf8');
  const selected = readFileSync(resolve(directory, 'kernels.txt'), 'utf8');
  const metrics = kernelMetrics(source, selected);
  writeFileSync(resolve(directory, 'kernel-summary.json'), JSON.stringify(metrics, null, 2) + '\n');
  assert.deepEqual(
    metrics.missing,
    [],
    'Metal shader pipelines not exercised: ' + metrics.missing.join(', '),
  );
  process.stdout.write(
    requireThreshold(
      'Metal kernel selections',
      percent(metrics.covered, metrics.total),
      config.metalKernelSelections,
    ) + '\n',
  );
}

mergeProfiles();
checkTotals(report());
checkKernels();
