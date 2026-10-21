import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, readdirSync, unlinkSync } from 'node:fs';
import { resolve } from 'node:path';
import process from 'node:process';
import { instrumentNative } from './native-source.mjs';

const directory = resolve('target/coverage/native');
const libraryDirectory = resolve('target/coverage/test-classes/native/macos');
const compilerFlags = [
  'clang++',
  '-std=c++17',
  '-O0',
  '-g',
  '-fobjc-arc',
  '-dynamiclib',
  '-fprofile-instr-generate',
  '-fcoverage-mapping',
  '-framework',
  'Foundation',
  '-framework',
  'Metal',
  '-framework',
  'MetalPerformanceShaders',
];

function prepareSources() {
  mkdirSync(directory, { recursive: true });
  mkdirSync(libraryDirectory, { recursive: true });
  for (const file of readdirSync(directory)) {
    if (file.endsWith('.profraw') || file === 'kernels.txt') unlinkSync(resolve(directory, file));
  }
  const source = readFileSync('native/deepj_metal_jni.mm', 'utf8');
  const hooks = '#include "' + resolve('src/test/native/coverage-hooks.h') + '"';
  writeFileSync(resolve(directory, 'deepj_metal_coverage.mm'), instrumentNative(source, hooks));
}

function compile() {
  const java = process.argv[2];
  assert.ok(java, 'Pass the JDK home to the native coverage build');
  const args = [
    ...compilerFlags,
    '-I' + resolve(java, 'include'),
    '-I' + resolve(java, 'include/darwin'),
    resolve(directory, 'deepj_metal_coverage.mm'),
    '-o',
    resolve(libraryDirectory, 'libdeepj_metal_jni.dylib'),
  ];
  execFileSync('xcrun', args, { stdio: 'inherit' });
}

assert.equal(process.platform, 'darwin', 'Native coverage requires macOS and a Metal device');
prepareSources();
compile();
