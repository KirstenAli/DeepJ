import { readdirSync, statSync } from 'node:fs';
import { extname, join } from 'node:path';

const extensions = new Set(['.h', '.hpp', '.c', '.cc', '.cpp', '.m', '.mm', '.metal']);

export function nativeFiles(roots = ['native', 'src/test/native']) {
  return roots
    .flatMap((root) => readdirSync(root, { recursive: true }).map((file) => join(root, file)))
    .filter((file) => extensions.has(extname(file)) && statSync(file).isFile())
    .sort();
}
