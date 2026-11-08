import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture } from './fixture.mjs';

function execution(context, metal = true) {
  const ui = fixture(context);
  let selected = 'auto';
  ui.window.executionBridge = {
    selected: () => selected,
    startup: () => 'auto',
    current: () => (metal ? 'Metal' : 'CPU'),
    metalAvailable: () => metal,
    save: (value) => {
      selected = value;
      ui.calls.push(['backend', value]);
      return true;
    },
  };
  ui.window.executionReady();
  return ui;
}

function select(ui, value) {
  ui.element('executionBackend').value = value;
  ui.element('executionBackend').dispatchEvent(new ui.window.Event('change'));
}

test('execution setting records the next-launch choice without loading or changing a model', (context) => {
  const ui = execution(context);
  assert.match(ui.element('executionHelp').textContent, /Using Metal/);
  select(ui, 'cpu');
  assert.equal(ui.element('executionHelp').textContent, 'Restart DeepJ to use CPU.');
  select(ui, 'auto');
  assert.match(ui.element('executionHelp').textContent, /Using Metal/);
  assert.deepEqual(ui.calls, [
    ['backend', 'cpu'],
    ['backend', 'auto'],
  ]);
});

test('unavailable Metal is disabled and execution stays accessible in each model mode', (context) => {
  const ui = execution(context, false);
  assert.equal(ui.element('executionBackend').querySelector('[value="metal"]').disabled, true);
  assert.match(ui.element('executionHelp').textContent, /Using CPU/);
  ['setModel', 'setImageModel', 'setVideoModel'].forEach((mode) => {
    ui.window[mode]('Model');
    ui.click('openSettings');
    assert.equal(ui.element('executionBackend').closest('[hidden]'), null);
  });
});

test('failed preference writes restore the saved selection and report the failure', (context) => {
  const ui = execution(context);
  ui.window.executionBridge.save = () => false;
  select(ui, 'cpu');
  assert.equal(ui.element('executionBackend').value, 'auto');
  assert.match(ui.element('toast').textContent, /Could not save/);
  assert.match(ui.element('executionHelp').textContent, /Using Metal/);
});
