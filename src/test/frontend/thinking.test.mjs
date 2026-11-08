import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture, finish } from './fixture.mjs';

test('thinking starts compact and keeps the same panel while its opening tag streams', (context) => {
  const ui = fixture(context);
  ui.element('thinkingMode').checked = true;
  ui.element('agentMode').checked = true;
  ui.window.startResponse('Check my files', 1);
  const content = ui.element('response-1').querySelector('.message-content');
  const panel = content.querySelector('.thinking-panel');
  assert.equal(panel.querySelector('summary').textContent, 'Thinking…');
  for (const text of [' ', '<', '<thi', '<think>Check first']) {
    ui.window.updateResponse(1, text);
    assert.equal(content.querySelector('.thinking-panel'), panel);
  }
  assert.equal(panel.querySelector('.thinking-content').textContent, 'Check first');
});

test('ordinary chat uses a compact preparation status without claiming to reason', (context) => {
  const ui = fixture(context);
  ui.window.startResponse('Hello', 1);
  const content = ui.element('response-1').querySelector('.message-content');
  assert.equal(content.querySelector('.generation-status').textContent, 'Preparing response…');
  assert.equal(content.querySelector('.thinking-panel'), null);
  ui.window.updateResponse(1, 'Hello!');
  assert.equal(content.textContent, 'Hello!');
  assert.equal(content.querySelector('.generation-status'), null);
});

test('alternative responses use the same compact thinking placeholder', (context) => {
  const ui = fixture(context);
  finish(ui).querySelector('.alternatives-button').click();
  ui.element('thinkingMode').checked = true;
  ui.window.startAlternative(1, 0);
  const panel = ui.element('response-1-alternative-0').querySelector('.thinking-panel');
  assert.equal(panel.querySelector('summary').textContent, 'Thinking…');
  ui.window.updateAlternative(1, 0, '<thi');
  assert.equal(ui.element('response-1-alternative-0').querySelector('.thinking-panel'), panel);
});

test('thinking appears beside agent mode only for a supported model and defaults off', (context) => {
  const ui = fixture(context);
  ui.window.setThinkingAvailable(true);
  assert.equal(ui.element('agentControls').hidden, false);
  assert.equal(ui.element('agentModeRow').hidden, true);
  assert.equal(ui.window.document.querySelector('.directory-row').hidden, true);
  assert.equal(ui.element('thinkingMode').checked, false);
  ui.window.setAgentWorkspace('/tmp');
  assert.equal(ui.element('thinkingRow').hidden, false);
  ui.window.setThinkingAvailable(false);
  assert.equal(ui.element('agentControls').hidden, false);
  ui.window.setAgentWorkspace('');
  assert.equal(ui.element('agentControls').hidden, true);
});

test('thinking is sent per request and resets when changing model', (context) => {
  const ui = fixture(context);
  ui.window.setModel('Qwen3');
  ui.window.setThinkingAvailable(true);
  ui.element('thinkingMode').checked = true;
  ui.element('rollingCache').checked = true;
  ui.element('prompt').value = 'Solve this';
  ui.window.submitPrompt();
  assert.equal(ui.calls[0][0], 'sendWithThinkingSettings');
  assert.deepEqual(ui.calls[0].slice(-2), [true, true]);
  ui.window.setModel('Frisky');
  assert.equal(ui.element('thinkingMode').checked, false);
  assert.equal(ui.element('thinkingRow').hidden, true);
});

test('streamed reasoning is separated safely from the answer and preserves user expansion', (context) => {
  const ui = fixture(context);
  ui.window.startResponse('Test', 1);
  ui.window.updateResponse(1, '<thi');
  assert.equal(ui.element('response-1').textContent, 'Thinking…');
  ui.window.updateResponse(1, '<think>Consider <img src=x>');
  const panel = ui.element('response-1').querySelector('details');
  panel.open = true;
  ui.window.updateResponse(1, '<think>Consider <img src=x></think>Answer');
  assert.equal(panel.open, false);
  panel.open = true;
  ui.window.finishResponse(1, '<think>Consider <img src=x></think>Answer!');
  assert.equal(panel.open, true);
  assert.equal(panel.querySelector('img'), null);
  assert.equal(ui.element('response-1').querySelector('.thinking-answer').textContent, 'Answer!');
});

test('unfinished thinking is distinguished from a completed answer', (context) => {
  const ui = fixture(context);
  const response = finish(ui, 1, '<think>Still working');
  assert.equal(response.querySelector('summary').textContent, 'Thinking incomplete');
  assert.match(response.querySelector('.thinking-answer').textContent, /Increase maximum tokens/);
  finish(ui, 2, 'Plain response');
  ui.window.renderThinking(response.querySelector('.message-content'), 'Ordinary text');
  assert.equal(response.querySelector('details'), null);
});

test('thinking and agent switches share a clear active label', (context) => {
  const ui = fixture(context);
  ui.window.setAgentWorkspace('/tmp');
  ui.window.setThinkingAvailable(true);
  ui.click('thinkingMode');
  assert.equal(ui.element('agentLabel').textContent, 'Thinking on');
  ui.click('agentMode');
  assert.equal(ui.element('agentLabel').textContent, 'Agent + thinking');
  ui.click('thinkingMode');
  assert.equal(ui.element('agentLabel').textContent, 'Agent on');
  ui.click('agentMode');
  assert.equal(ui.element('agentLabel').textContent, 'Chat options');
  ui.click('agentPicker');
  assert.equal(ui.element('agentTitle').textContent, 'Chat options');
});

test('agent reasoning stays in the same panel across tool rounds and the final answer', (context) => {
  const ui = fixture(context);
  ui.window.startResponse('Check my files', 1);
  ui.window.updateResponse(1, '<think>Check first');
  const panel = ui.element('response-1').querySelector('details');
  panel.open = true;
  ui.window.updateResponse(1, '<think>Check first</think>');
  assert.equal(panel.open, false);
  ui.window.updateResponse(1, '<think>Check first\n\nUse the result');
  assert.equal(panel.querySelector('summary').textContent, 'Thinking…');
  assert.equal(panel.dataset.complete, undefined);
  panel.open = true;
  ui.window.finishResponse(1, '<think>Check first\n\nUse the result</think>Done');
  assertCompletedAgentThoughts(ui, panel);
});

function assertCompletedAgentThoughts(ui, panel) {
  assert.equal(ui.element('response-1').querySelector('details'), panel);
  assert.equal(panel.open, false);
  assert.equal(panel.querySelector('summary').textContent, 'Thought process');
  assert.equal(
    panel.querySelector('.thinking-content').textContent,
    'Check first\n\nUse the result',
  );
  assert.equal(ui.element('response-1').querySelector('.thinking-answer').textContent, 'Done');
}
