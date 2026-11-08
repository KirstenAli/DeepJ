import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture } from './fixture.mjs';

test('settings stay usable across model modes without an execution preference', (context) => {
  const ui = fixture(context);
  ['setModel', 'setImageModel', 'setVideoModel'].forEach((mode) => {
    ui.window[mode]('Model');
    ui.click('openSettings');
    assert.equal(ui.element('executionBackend'), null);
    assert.equal(ui.element('settingsBackdrop').hidden, false);
    ui.click('doneSettings');
  });
});
