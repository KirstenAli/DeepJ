import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture, finish, scrollAway } from './fixture.mjs';

function openAlternatives(ui) {
  const response = finish(ui);
  response.querySelector('.alternatives-button').click();
  return response;
}

test('alternatives are requested once and can be hidden without removing the original', (context) => {
  const ui = fixture(context);
  const response = openAlternatives(ui);
  const button = response.querySelector('.alternatives-button');
  button.click();
  assert.equal(response.querySelector('.alternatives').hidden, true);
  assert.equal(button.textContent, 'See alternative responses');
  button.click();
  assert.equal(response.querySelector('.alternatives').hidden, false);
  assert.equal(button.textContent, 'Hide alternative responses');
  assert.deepEqual(ui.calls, [['alternatives', 1]]);
  assert.equal(response.querySelector('.message-content').textContent, 'A useful response');
});

test('both alternative responses stream and have independent copy actions', (context) => {
  const ui = fixture(context);
  openAlternatives(ui);
  for (const option of [0, 1]) {
    ui.window.startAlternative(1, option);
    ui.window.updateAlternative(1, option, 'Partial');
    ui.window.finishAlternative(1, option, 'Alternative ' + option);
    const response = ui.element('response-1-alternative-' + option);
    assert.equal(response.querySelector('.message-content').textContent, 'Alternative ' + option);
    assert.equal(response.querySelectorAll('.copy-button').length, 1);
  }
  ui.window.clearAlternativeActions();
  assert.equal(ui.element('messages').querySelector('.alternatives-button'), null);
});

test('alternative creation respects a reader who has scrolled up', (context) => {
  const ui = fixture(context);
  openAlternatives(ui);
  scrollAway(ui);
  ui.window.startAlternative(1, 0);
  assert.equal(ui.element('conversation').scrollTop, 0);
});

test('empty and failed alternatives show errors rather than copy buttons', (context) => {
  const ui = fixture(context);
  openAlternatives(ui);
  ui.window.startAlternative(1, 0);
  ui.window.finishAlternative(1, 0, '');
  ui.window.startAlternative(1, 1);
  ui.window.finishAlternative(1, 1, '   ');
  assert.ok(ui.element('response-1-alternative-0').classList.contains('error'));
  assert.ok(ui.element('response-1-alternative-1').classList.contains('error'));
  assert.equal(ui.element('response-1-alternative-0').querySelector('.copy-button'), null);
});

test('stopping alternatives removes pending placeholders and preserves partial text', (context) => {
  const ui = fixture(context);
  openAlternatives(ui);
  ui.window.startAlternative(1, 0);
  ui.window.stopAlternative(1, 0);
  assert.equal(ui.element('response-1-alternative-0'), null);
  ui.window.startAlternative(1, 1);
  ui.window.updateAlternative(1, 1, 'Keep this');
  ui.window.stopAlternative(1, 1);
  assert.equal(
    ui.element('response-1-alternative-1').querySelector('.message-content').textContent,
    'Keep this',
  );
  ui.window.stopAlternative(999, 0);
});
