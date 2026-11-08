import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture } from './fixture.mjs';

test('loading releases stale input focus and restores the draft caret when ready', (context) => {
  const ui = fixture(context);
  const prompt = ui.element('prompt');
  prompt.value = 'A draft';
  prompt.focus();
  prompt.setSelectionRange(3, 3);
  ui.window.setBusy(true);
  assert.notEqual(ui.window.document.activeElement, prompt);
  prompt.focus();
  ui.window.setModel('Qwen3');
  ui.window.setBusy(false);
  assert.equal(ui.window.document.activeElement, prompt);
  assert.equal(prompt.value, 'A draft');
  assert.equal(prompt.selectionStart, 3);
});

test('chat, image and video loading leave the composer ready to type', (context) => {
  const ui = fixture(context);
  ['setModel', 'setImageModel', 'setVideoModel'].forEach((mode) => {
    ui.window.setBusy(true);
    ui.element('prompt').focus();
    ui.window[mode]('Model');
    ui.window.setBusy(false);
    assert.equal(ui.element('prompt').disabled, false);
    assert.equal(ui.window.document.activeElement, ui.element('prompt'));
  });
});

test('failed loading restores an editable focused composer', (context) => {
  const ui = fixture(context);
  ui.window.setBusy(true);
  ui.window.setStatus('Could not load model');
  ui.window.setBusy(false);
  assert.equal(ui.element('prompt').disabled, false);
  assert.equal(ui.window.document.activeElement, ui.element('prompt'));
});

test('finishing loading preserves focus chosen on another control', (context) => {
  const ui = fixture(context);
  ui.window.setBusy(true);
  ui.element('openSettings').focus();
  ui.window.setBusy(false);
  assert.equal(ui.window.document.activeElement, ui.element('openSettings'));
});

test('finishing loading never moves focus behind an open dialog', (context) => {
  const ui = fixture(context);
  ['settingsBackdrop', 'approvalBackdrop'].forEach((id) => {
    ui.window.setBusy(true);
    ui.element(id).hidden = false;
    ui.window.document.activeElement.blur();
    ui.window.setBusy(false);
    assert.notEqual(ui.window.document.activeElement, ui.element('prompt'));
    ui.element(id).hidden = true;
  });
});
