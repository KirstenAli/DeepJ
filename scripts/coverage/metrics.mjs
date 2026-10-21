import assert from 'node:assert/strict';

export function percent(covered, total) {
  assert.ok(total > 0, 'Coverage report contains no executable code');
  assert.ok(covered >= 0 && covered <= total, 'Invalid coverage counts');
  return (covered / total) * 100;
}

export function requireThreshold(label, value, minimum) {
  assert.ok(Number.isFinite(value), `${label}: missing or invalid coverage`);
  assert.ok(value >= minimum, `${label}: ${value.toFixed(2)}% is below ${minimum}%`);
  return `${label}: ${value.toFixed(2)}% (minimum ${minimum}%)`;
}

export function javaMetrics(packages) {
  return Object.fromEntries(['LINE', 'BRANCH'].map((name) => [name, sumCounter(packages, name)]));
}

function sumCounter(packages, name) {
  const counters = packages.flatMap((node) =>
    [...node.children].filter((item) => item.tagName === 'counter'),
  );
  const selected = counters.filter((node) => node.getAttribute('type') === name);
  const covered = selected.reduce((sum, node) => sum + Number(node.getAttribute('covered')), 0);
  const missed = selected.reduce((sum, node) => sum + Number(node.getAttribute('missed')), 0);
  return percent(covered, covered + missed);
}

export function selectPackages(document, excluded = []) {
  return [...document.querySelectorAll('report > package')].filter(
    (node) => !excluded.includes(node.getAttribute('name').replaceAll('/', '.')),
  );
}

export function kernelMetrics(source, selected) {
  const expected = new Set(
    [...source.matchAll(/kernel\s+void\s+(\w+)\s*\(/g)].map((match) => match[1]),
  );
  const exercised = new Set(selected.trim().split(/\r?\n/).filter(Boolean));
  const missing = [...expected].filter((name) => !exercised.has(name));
  return { total: expected.size, covered: expected.size - missing.length, missing };
}
