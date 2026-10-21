import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture, keyboard } from './fixture.mjs';

test('model menu and preview picker behave clearly', (context) => {
  const ui = fixture(context);
  ui.click('modelPicker');
  assert.equal(ui.element('modelMenu').hidden, false);
  assert.equal(ui.element('modelPicker').getAttribute('aria-expanded'), 'true');
  ui.click('modelPicker');
  assert.equal(ui.element('modelMenu').hidden, true);
  ui.click('menuLoad');
  ui.click('composerModel');
  assert.deepEqual(ui.calls, [['chooseModel'], ['chooseModel']]);
  delete ui.window.chatBridge;
  ui.click('composerModel');
  assert.match(ui.element('toast').textContent, /not available in this preview/);
});

test('settings open, close, escape and outside clicks restore focus', (context) => {
  const ui = fixture(context);
  ui.click('openSettings');
  assert.equal(ui.element('settingsBackdrop').hidden, false);
  assert.equal(ui.window.document.activeElement, ui.element('closeSettings'));
  ui.click('doneSettings');
  assert.equal(ui.window.document.activeElement, ui.element('openSettings'));
  ui.window.closeSettings();
  ui.click('openSettings');
  ui.element('settingsTitle').click();
  assert.equal(ui.element('settingsBackdrop').hidden, false);
  ui.click('settingsBackdrop');
  assert.equal(ui.element('settingsBackdrop').hidden, true);
});

test('escape closes both the settings dialog and model menu', (context) => {
  const ui = fixture(context);
  ui.click('openSettings');
  ui.click('modelPicker');
  keyboard(ui, ui.window.document, 'Escape');
  assert.equal(ui.element('settingsBackdrop').hidden, true);
  assert.equal(ui.element('modelMenu').hidden, true);
});

test('theme buttons select exactly one theme', (context) => {
  const ui = fixture(context);
  ui.window.document.querySelector('[data-theme="dark"]').click();
  assert.equal(ui.window.document.body.className, 'theme-dark');
  assert.equal(ui.window.document.querySelectorAll('.theme-choice.selected').length, 1);
  assert.equal(ui.window.document.querySelector('.theme-choice.selected').dataset.theme, 'dark');
});

for (const value of ['', '0', '-1']) {
  test(`invalid token limit ${JSON.stringify(value)} blocks submission and opens settings`, (context) => {
    const ui = fixture(context);
    ui.window.setModel('nova.dj');
    ui.element('prompt').value = 'Hello';
    ui.element('maxTokens').value = value;
    ui.window.submitPrompt();
    assert.equal(ui.calls.length, 0);
    assert.equal(ui.element('settingsBackdrop').hidden, false);
    assert.equal(ui.element('generationOptions').open, true);
    assert.equal(ui.window.document.activeElement, ui.element('maxTokens'));
  });
}

test('tab is trapped at both ends of the settings dialog', (context) => {
  const ui = fixture(context);
  const controls = [...ui.element('settingsBackdrop').querySelectorAll('button, input, summary')];
  controls.forEach((control) => Object.defineProperty(control, 'offsetParent', { value: control }));
  controls.at(-1).focus();
  assert.equal(keyboard(ui, controls.at(-1), 'Tab').defaultPrevented, true);
  assert.equal(ui.window.document.activeElement, controls[0]);
  assert.equal(keyboard(ui, controls[0], 'Tab', true).defaultPrevented, true);
  assert.equal(ui.window.document.activeElement, controls.at(-1));
  controls[1].focus();
  assert.equal(keyboard(ui, controls[1], 'Tab').defaultPrevented, false);
  assert.equal(keyboard(ui, controls[1], 'a').defaultPrevented, false);
});
