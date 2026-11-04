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

test('composer tools open beside the prompt and report agent and directory actions', (context) => {
  const ui = fixture(context);
  assert.equal(ui.element('agentControls').hidden, true);
  ui.window.setAgentWorkspace('/tmp/project');
  ui.click('agentPicker');
  assert.equal(ui.element('agentPicker').getAttribute('aria-expanded'), 'true');
  ui.click('agentMode');
  assert.equal(ui.element('agentLabel').textContent, 'Agent on');
  ui.click('chooseAgentDirectory');
  assert.equal(ui.element('agentDirectory').textContent, 'other project');
  assert.equal(ui.element('agentDirectory').title, '/tmp/other project');
  assert.equal(ui.element('agentMode').checked, true);
  ui.click('agentMode');
  assert.equal(ui.element('agentLabel').textContent, 'Tools');
  assert.deepEqual(ui.calls, [['setEnabled', true], ['chooseDirectory'], ['setEnabled', false]]);
});

test('composer tools dismiss with Escape and outside click', (context) => {
  const ui = fixture(context);
  ui.window.setAgentWorkspace('/tmp/project');
  ui.click('agentPicker');
  keyboard(ui, ui.element('agentMode'), 'Escape');
  assert.equal(ui.element('agentPicker').getAttribute('aria-expanded'), 'false');
  assert.equal(ui.window.document.activeElement, ui.element('agentPicker'));
  ui.click('agentPicker');
  ui.click('prompt');
  assert.equal(ui.element('agentMenu').hidden, true);
});

test('composer tools close during generation and reset when the model changes', (context) => {
  const ui = fixture(context);
  ui.window.setAgentWorkspace('/tmp/project');
  assert.equal(ui.element('agentControls').hidden, false);
  ui.click('agentPicker');
  ui.click('agentMode');
  ui.window.setBusy(true);
  assert.equal(ui.element('agentPicker').disabled, true);
  assert.equal(ui.element('agentMenu').hidden, true);
  ui.window.setBusy(false);
  assert.equal(ui.element('agentMode').checked, true);
  ui.window.setAgentWorkspace('');
  assert.equal(ui.element('agentControls').hidden, true);
  assert.equal(ui.element('agentMode').checked, false);
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

test('terminal approval shows literal command details and restores focus after a decision', (context) => {
  const ui = fixture(context);
  ui.window.approvalBridge = {
    decide: (allow) => {
      ui.calls.push(allow);
      ui.window.closeApproval();
    },
  };
  ui.element('prompt').focus();
  ui.window.openApproval('["/bin/echo", "<tag>"]', '"/tmp/project"');
  assert.equal(ui.element('approvalCommand').textContent, '["/bin/echo", "<tag>"]');
  assert.equal(ui.element('approvalDirectory').textContent, '"/tmp/project"');
  assert.equal(ui.window.document.activeElement, ui.element('denyCommand'));
  ui.click('allowCommand');
  assert.equal(ui.element('approvalBackdrop').hidden, true);
  assert.equal(ui.window.document.activeElement, ui.element('prompt'));
  assert.deepEqual(ui.calls, [true]);
});

test('Escape denies a pending terminal command', (context) => {
  const ui = fixture(context);
  ui.window.approvalBridge = { decide: (allow) => ui.calls.push(allow) };
  ui.window.openApproval('["/bin/pwd"]', '"/tmp"');
  keyboard(ui, ui.element('denyCommand'), 'Escape');
  assert.deepEqual(ui.calls, [false]);
});

test('reviewing an edit preserves untouched argument contents and the directory', (context) => {
  const ui = fixture(context);
  ui.window.approvalBridge = {
    reviewEdit: (argv) => {
      ui.calls.push(JSON.parse(argv));
      return JSON.stringify({ preview: argv });
    },
  };
  ui.window.openApproval(JSON.stringify(['/bin/echo', 'first\rsecond', 'remove me']), '"/tmp"');
  ui.click('editCommand');
  assert.equal(ui.element('approvalActions').hidden, true);
  ui.element('commandArguments').lastElementChild.querySelector('button').click();
  ui.click('addCommandArgument');
  ui.click('reviewCommandEdit');
  assert.deepEqual(ui.calls, [['/bin/echo', 'first\rsecond', '']]);
  assert.equal(ui.element('approvalDirectory').textContent, '"/tmp"');
  assert.equal(ui.element('approvalActions').hidden, false);
});

test('an invalid edit stays in the editor and can be cancelled', (context) => {
  const ui = fixture(context);
  ui.window.approvalBridge = {
    reviewEdit: () => JSON.stringify({ error: 'The executable could not be found.' }),
  };
  ui.window.openApproval('["/bin/pwd"]', '"/tmp"');
  ui.click('editCommand');
  ui.element('commandExecutable').value = 'missing';
  ui.click('reviewCommandEdit');
  assert.equal(ui.element('commandEditError').hidden, false);
  assert.equal(ui.element('approvalActions').hidden, true);
  ui.click('cancelCommandEdit');
  assert.equal(ui.element('approvalCommand').textContent, '["/bin/pwd"]');
  assert.equal(ui.element('approvalActions').hidden, false);
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
