import assert from 'node:assert/strict';
import test from 'node:test';
import { ESLint } from 'eslint';

async function violations(body, rule) {
  const source = `function fixture(x) { ${body} }`;
  const [result] = await new ESLint().lintText(source, {
    filePath: 'src/main/resources/chat.js',
  });
  assert.equal(result.fatalErrorCount, 0);
  return result.messages.filter((message) => message.ruleId === rule);
}

for (const score of [15, 16]) {
  test(`cognitive complexity boundary: ${score}`, async () => {
    const messages = await violations(
      'if (x > 0) { x--; } '.repeat(score),
      'sonarjs/cognitive-complexity',
    );
    assert.equal(messages.length, score > 15 ? 1 : 0);
  });
}

test('mixed JavaScript nesting rejects a third level', async () => {
  const body = 'if (x) { for (const item of x) { while (item) { x--; } } }';
  assert.equal((await violations(body, 'max-depth')).length, 1);
});

test('two JavaScript control-flow levels remain allowed', async () => {
  const body = 'if (x) { for (const item of x) { x += item; } }';
  assert.equal((await violations(body, 'max-depth')).length, 0);
});

test('JavaScript else-if chains stay flat', async () => {
  const body = 'if (x) { x--; } ' + 'else if (x) { x++; } '.repeat(4);
  assert.equal((await violations(body, 'max-depth')).length, 0);
});
