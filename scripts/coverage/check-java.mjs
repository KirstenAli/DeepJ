import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import process from 'node:process';
import { JSDOM } from 'jsdom';
import { javaMetrics, requireThreshold, selectPackages } from './metrics.mjs';

const config = JSON.parse(readFileSync('config/coverage.json', 'utf8'));
const xml = readFileSync('target/site/jacoco/jacoco.xml', 'utf8');
const document = new JSDOM(xml, { contentType: 'text/xml' }).window.document;
const summary = {};

function checkScope(name, packages) {
  const metrics = javaMetrics(packages);
  summary[name] = metrics;
  process.stdout.write(requireThreshold(`${name} lines`, metrics.LINE, config[name].lines) + '\n');
  process.stdout.write(
    requireThreshold(`${name} branches`, metrics.BRANCH, config[name].branches) + '\n',
  );
}

function checkHardware() {
  checkScope('fullJava', selectPackages(document));
  const metal = selectPackages(document).filter(
    (node) => node.getAttribute('name') === 'io/github/kirstenali/deepj/tensor/metal',
  );
  checkScope('metalJava', metal);
}

for (const name of ['portableJava', 'coreJava']) {
  checkScope(name, selectPackages(document, config[name].excludePackages));
}
if (process.argv.includes('--full')) checkHardware();
mkdirSync('target/coverage', { recursive: true });
writeFileSync('target/coverage/java-summary.json', JSON.stringify(summary, null, 2) + '\n');
