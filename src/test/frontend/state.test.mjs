import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture, finish, keyboard } from './fixture.mjs';

test('model state and sidebar controls stay synchronized', (context) => {
  const ui = fixture(context);
  ui.window.bridgeReady();
  assert.match(ui.element('status').textContent, /Choose a model/);
  ui.window.setModel('nova.dj');
  ui.window.setStatus('Ready');
  assert.equal(ui.element('modelName').textContent, 'nova.dj');
  assert.equal(ui.element('composerHint').textContent, 'nova.dj');
  assert.ok(ui.element('app').classList.contains('ready'));
  ui.click('collapseSidebar');
  assert.ok(ui.element('app').classList.contains('collapsed'));
  ui.click('reopenSidebar');
  assert.ok(!ui.element('app').classList.contains('collapsed'));
});

test('busy state disables actions while the stop button remains available', (context) => {
  const ui = fixture(context);
  const response = finish(ui);
  ui.window.setBusy(true);
  assert.equal(ui.element('prompt').disabled, true);
  assert.equal(response.querySelector('.alternatives-button').disabled, true);
  assert.equal(ui.element('composerModel').disabled, true);
  ui.window.setGenerating(true);
  assert.equal(ui.element('send').disabled, false);
  assert.equal(ui.element('send').getAttribute('aria-label'), 'Stop generating');
  ui.click('send');
  assert.deepEqual(ui.calls, [['stop']]);
  ui.window.setGenerating(false);
  ui.window.setBusy(false);
  assert.equal(ui.element('prompt').disabled, false);
});

test('blank prompts and busy submissions do not reach Java', (context) => {
  const ui = fixture(context);
  ui.window.submitPrompt();
  ui.element('prompt').value = '  ';
  ui.window.submitPrompt();
  ui.element('prompt').value = 'Hello';
  ui.window.setBusy(true);
  ui.window.submitPrompt();
  assert.deepEqual(ui.calls, []);
});

test('submission without a model opens the desktop picker', (context) => {
  const ui = fixture(context);
  ui.element('prompt').value = 'Hello';
  ui.window.submitPrompt();
  assert.deepEqual(ui.calls, [['chooseModel']]);
  assert.equal(ui.element('modelMenu').hidden, true);
});

test('enter sends the prompt with exactly the selected generation settings', (context) => {
  const ui = fixture(context);
  ui.window.setModel('nova.dj');
  ui.element('prompt').value = '  Hello  ';
  ui.element('repetitionPenalty').value = '1.1';
  const enter = keyboard(ui, ui.element('prompt'), 'Enter');
  assert.equal(enter.defaultPrevented, true);
  assert.deepEqual(ui.calls, [['sendWithSettings', 'Hello', 80, 0.8, 40, 1234, 1.1]]);
});

test('shift-enter and ordinary keys do not submit', (context) => {
  const ui = fixture(context);
  ui.element('prompt').value = 'Hello';
  assert.equal(keyboard(ui, ui.element('prompt'), 'Enter', true).defaultPrevented, false);
  assert.equal(keyboard(ui, ui.element('prompt'), 'a').defaultPrevented, false);
  assert.deepEqual(ui.calls, []);
});

test('rolling cache is opt-in and reaches Java with unchanged sampling settings', (context) => {
  const ui = fixture(context);
  ui.window.setModel('frisky.dj');
  assert.equal(ui.element('rollingCache').checked, false);
  ui.element('rollingCache').checked = true;
  ui.element('prompt').value = 'Hello';
  ui.window.submitPrompt();
  assert.deepEqual(ui.calls, [['sendWithWindowSettings', 'Hello', 80, 0.8, 40, 1234, 1, true]]);
});

test('the send button uses the normal submission path when not generating', (context) => {
  const ui = fixture(context);
  ui.window.setModel('nova.dj');
  ui.element('prompt').value = 'A question';
  ui.window.resizePrompt();
  ui.click('send');
  assert.equal(ui.calls[0][0], 'sendWithSettings');
  assert.equal(ui.calls[0][1], 'A question');
});

test('composer resizing handles input and window resize events', (context) => {
  const ui = fixture(context);
  Object.defineProperty(ui.element('prompt'), 'scrollHeight', { value: 200 });
  ui.element('prompt').value = 'A multiline message';
  ui.element('prompt').dispatchEvent(new ui.window.Event('input'));
  assert.equal(ui.element('prompt').style.height, '138px');
  assert.equal(ui.element('send').disabled, false);
  ui.window.dispatchEvent(new ui.window.Event('resize'));
  assert.equal(ui.element('prompt').style.height, '138px');
});

test('new chat clears the conversation and notifies Java', (context) => {
  const ui = fixture(context);
  finish(ui);
  ui.click('newChat');
  assert.equal(ui.element('messages').children.length, 0);
  assert.ok(ui.element('app').classList.contains('empty'));
  assert.equal(ui.window.document.activeElement, ui.element('prompt'));
  assert.deepEqual(ui.calls, [['newChat']]);
  delete ui.window.chatBridge;
  ui.window.newChat();
  assert.equal(ui.element('messages').children.length, 0);
});
