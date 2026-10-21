import assert from 'node:assert/strict';

const functions = new Set([
  'FunctionDecl',
  'CXXMethodDecl',
  'CXXConstructorDecl',
  'CXXDestructorDecl',
  'CXXConversionDecl',
  'ObjCMethodDecl',
]);
const tries = new Set(['CXXTryStmt', 'ObjCAtTryStmt']);
const bodies = new Set(['CompoundStmt', ...tries]);
const loops = new Set([
  'ForStmt',
  'CXXForRangeStmt',
  'ObjCForCollectionStmt',
  'WhileStmt',
  'DoStmt',
  'SwitchStmt',
]);
const choices = new Set([
  'ConditionalOperator',
  'BinaryConditionalOperator',
  'CXXCatchStmt',
  'ObjCAtCatchStmt',
]);
const transparent = new Set(['ParenExpr', 'ImplicitCastExpr', 'ExprWithCleanups']);
const calls = new Set(['CallExpr', 'CXXMemberCallExpr', 'CXXOperatorCallExpr']);

function children(node) {
  return (node.inner ?? []).filter((child) => child.kind);
}

export function nativeMetrics(ast) {
  const found = new Map();
  collectFunctions(ast, found);
  return [...found.values()];
}

function collectFunctions(node, found) {
  const body = children(node).find((child) => bodies.has(child.kind));
  if (functions.has(node.kind) && body && !node.isImplicit) found.set(node.id, measure(node, body));
  for (const child of children(node)) collectFunctions(child, found);
}

function measure(fn, body) {
  const state = { cognitive: 0, nesting: 0, functionId: fn.id, recursive: false, seen: new Set() };
  visit(body, state, 0, 0);
  const location = fn.loc ?? fn.range?.begin ?? {};
  return {
    name: fn.name,
    line: location.presumedLine ?? location.line,
    cognitive: state.cognitive,
    nesting: state.nesting,
  };
}

function visit(node, state, nesting, depth, logical = null) {
  if (state.seen.has(node.id)) return;
  state.seen.add(node.id);
  if (functions.has(node.kind) || node.kind === 'CXXRecordDecl') return;
  if (node.kind === 'IfStmt') return visitIf(node, state, nesting, depth);
  if (node.kind === 'LambdaExpr') return visitLambda(node, state, nesting);
  recordIncrement(node, state, nesting, logical);
  visitChildren(node, state, nesting, depth, logical);
}

function visitChildren(node, state, nesting, depth, logical) {
  const control = loops.has(node.kind) || tries.has(node.kind);
  const nextDepth = depth + Number(control);
  const nextNesting = nesting + Number(loops.has(node.kind) || choices.has(node.kind));
  state.nesting = Math.max(state.nesting, nextDepth);
  for (const child of children(node)) {
    visit(child, state, nextNesting, nextDepth, logicalContext(node, logical));
  }
}

function visitIf(node, state, nesting, depth, alternative = false) {
  const parts = children(node);
  const bodyIndex = parts.length - (node.hasElse ? 2 : 1);
  const nextDepth = depth + Number(!alternative);
  state.cognitive += alternative ? 1 : 1 + nesting;
  state.nesting = Math.max(state.nesting, nextDepth);
  for (const condition of parts.slice(0, bodyIndex)) visit(condition, state, nesting, depth);
  visit(parts[bodyIndex], state, nesting + 1, nextDepth);
  if (node.hasElse) visitElse(parts.at(-1), state, nesting, nextDepth);
}

function visitElse(node, state, nesting, depth) {
  if (node.kind === 'IfStmt') return visitIf(node, state, nesting, depth, true);
  state.cognitive++;
  visit(node, state, nesting + 1, depth);
}

function visitLambda(node, state, nesting) {
  const body = children(node).findLast((child) => child.kind === 'CompoundStmt');
  if (body) visit(body, state, nesting + 1, 0);
}

function isLogical(node) {
  return node.kind === 'BinaryOperator' && ['&&', '||'].includes(node.opcode);
}

function logicalContext(node, previous) {
  if (isLogical(node)) return node.opcode;
  return transparent.has(node.kind) ? previous : null;
}

function recordIncrement(node, state, nesting, logical) {
  if (loops.has(node.kind) || choices.has(node.kind)) state.cognitive += 1 + nesting;
  if (isLogical(node) && node.opcode !== logical) state.cognitive++;
  if (['GotoStmt', 'IndirectGotoStmt'].includes(node.kind)) state.cognitive++;
  if (calls.has(node.kind)) recordRecursion(node, state);
}

function recordRecursion(node, state) {
  const callee = children(node)[0];
  if (!state.recursive && references(callee, state.functionId)) {
    state.cognitive++;
    state.recursive = true;
  }
}

function references(node, id) {
  return (
    node?.referencedDecl?.id === id ||
    node?.referencedMemberDecl === id ||
    children(node ?? {}).some((child) => references(child, id))
  );
}

export function nativeViolations(metrics, limits) {
  validateMetrics(metrics, limits);
  return metrics.flatMap((fn) =>
    ['cognitive', 'nesting']
      .filter((metric) => fn[metric] > limits[metric])
      .map((metric) => `${fn.name}:${fn.line} ${metric} ${fn[metric]} exceeds ${limits[metric]}`),
  );
}

function validateMetrics(metrics, limits) {
  for (const name of ['cognitive', 'nesting']) {
    assert.ok(
      Number.isSafeInteger(limits[name]) && limits[name] >= 0,
      'Invalid native limit: ' + name,
    );
    for (const fn of metrics) {
      assert.ok(
        Number.isSafeInteger(fn[name]) && fn[name] >= 0,
        'Missing native metric: ' + fn.name,
      );
    }
  }
}
