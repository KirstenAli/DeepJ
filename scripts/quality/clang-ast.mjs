import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { analysisSource } from './native-source.mjs';

const compilerFlags = [
  'clang++',
  '-std=c++17',
  '-fobjc-arc',
  '-fsyntax-only',
  '-Wno-unknown-attributes',
  '-Xclang',
  '-ast-dump=json',
  '-Xclang',
];

export function clangAst(source, file, namespace, flags = []) {
  const directory = resolve('target/quality/native');
  mkdirSync(directory, { recursive: true });
  const generated = resolve(directory, namespace + '.mm');
  const prepared = analysisSource(source, file, namespace);
  writeFileSync(generated, prepared);
  const args = [...compilerFlags, '-ast-dump-filter=' + namespace, ...flags, generated];
  const ast = JSON.parse(dumpAst(args));
  assert.equal(ast.kind, 'NamespaceDecl', 'Clang did not return the complete analysis namespace');
  assert.equal(ast.name, namespace, 'Unexpected native analysis scope');
  ast.analysisSource = prepared;
  return ast;
}

function dumpAst(args) {
  try {
    return execFileSync('xcrun', args, {
      encoding: 'utf8',
      stdio: 'pipe',
      maxBuffer: 32 * 1024 * 1024,
    });
  } catch (error) {
    const diagnostic = error.stderr?.toString().trim() || error.message;
    error.stdout = '';
    error.output = [];
    throw new Error('Native source analysis failed:\n' + diagnostic, { cause: error });
  }
}
