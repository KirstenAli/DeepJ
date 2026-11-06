import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture } from './fixture.mjs';

function player(context) {
  const ui = fixture(context);
  const node = ui.window.videoPlayer(1, ['/0.png', '/1.png', '/2.png'], 320, 240, 16);
  ui.element('messages').append(node);
  return { ...ui, button: node.querySelector('button'), seek: node.querySelector('input'), node };
}

test('preview plays every frame, stops at the end and can replay', (context) => {
  const ui = player(context);
  ui.button.click();
  assert.equal(ui.button.getAttribute('aria-label'), 'Pause video');
  ui.timers();
  assert.equal(ui.seek.value, '1');
  ui.timers();
  assert.equal(ui.seek.value, '2');
  ui.timers();
  assert.equal(ui.button.getAttribute('aria-label'), 'Play video');
  ui.button.click();
  assert.equal(ui.seek.value, '0');
  ui.button.click();
  ui.timers();
  assert.equal(ui.seek.value, '0');
});

test('seeking pauses playback and updates the displayed frame and accessible time', (context) => {
  const ui = player(context);
  ui.button.click();
  ui.seek.value = '2';
  ui.seek.dispatchEvent(new ui.window.Event('input'));
  ui.timers();
  assert.equal(ui.seek.value, '2');
  assert.equal(ui.button.getAttribute('aria-label'), 'Play video');
  assert.match(ui.node.querySelector('img').src, /2.png$/);
  assert.equal(ui.seek.getAttribute('aria-valuetext'), '0.13 seconds');
  ui.button.click();
  ui.window.setModel('Other model');
  ui.timers();
  assert.equal(ui.seek.value, '0');
});
