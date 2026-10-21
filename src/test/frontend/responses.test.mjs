import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture, finish, scrollAway } from './fixture.mjs';

test('responses stream as literal text with one copy action', (context) => {
  const ui = fixture(context);
  ui.element('prompt').value = 'Hello';
  ui.window.startResponse('A question', 1);
  ui.window.updateResponse(1, '<script>not executable</script>');
  ui.window.updateResponse(1, 'Complete response');
  const response = ui.element('response-1');
  assert.equal(response.querySelector('.message-content').textContent, 'Complete response');
  assert.equal(response.querySelectorAll('.copy-button').length, 1);
  assert.equal(ui.element('prompt').value, '');
  assert.equal(response.classList.contains('pending'), false);
  ui.window.updateResponse(999, 'A stale update');
});

test('copy sends only the response text and restores its icon', (context) => {
  const ui = fixture(context);
  const response = finish(ui, 1, 'Copy exactly this');
  const button = response.querySelector('.copy-button');
  button.click();
  assert.deepEqual(ui.calls, [['copy', 'Copy exactly this']]);
  assert.equal(button.getAttribute('aria-label'), 'Copied');
  ui.timers();
  assert.equal(button.getAttribute('aria-label'), 'Copy response');
  delete ui.window.chatBridge;
  button.click();
  assert.equal(ui.calls.length, 1);
});

for (const text of ['', '   ', null]) {
  test(`empty response ${JSON.stringify(text)} removes the placeholder`, (context) => {
    const ui = fixture(context);
    finish(ui, 1, text);
    assert.equal(ui.element('response-1'), null);
    assert.match(ui.element('toast').textContent, /No text generated/);
    ui.timers();
    assert.equal(ui.element('toast').classList.contains('visible'), false);
  });
}

test('failure removes copy actions and displays an error safely', (context) => {
  const ui = fixture(context);
  const response = finish(ui);
  ui.window.failResponse(1, 'Generation failed');
  assert.ok(response.classList.contains('error'));
  assert.equal(response.querySelector('.message-content').textContent, 'Generation failed');
  assert.equal(response.querySelector('.message-actions'), null);
  ui.window.failResponse(999, 'Stale failure');
});

test('stopping removes placeholders but keeps generated text', (context) => {
  const ui = fixture(context);
  ui.window.startResponse('Question', 1);
  ui.window.stopResponse(1);
  assert.equal(ui.element('response-1'), null);
  const response = finish(ui, 2);
  ui.window.stopResponse(2);
  assert.equal(ui.element('response-2'), response);
  ui.window.stopResponse(999);
});

test('stream updates do not scroll readers away from older messages', (context) => {
  const ui = fixture(context);
  ui.window.startResponse('Question', 1);
  scrollAway(ui);
  ui.window.updateResponse(1, 'A partial response');
  assert.equal(ui.element('conversation').scrollTop, 0);
  ui.element('conversation').scrollTop = 900;
  ui.window.updateResponse(1, 'A longer response');
  assert.equal(ui.element('conversation').scrollTop, 1000);
});
