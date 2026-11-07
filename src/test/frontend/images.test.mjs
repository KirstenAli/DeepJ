import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture, finish } from './fixture.mjs';

function images(context) {
  const ui = fixture(context);
  ui.window.imageBridge = Object.fromEntries(
    ['chooseModel', 'generateWithPrecision', 'stop', 'save'].map((name) => [
      name,
      (...args) => ui.calls.push([name, ...args]),
    ]),
  );
  ui.window.setImageModel('SANA-Sprint 0.6B');
  return ui;
}

test('image mode adapts the existing composer and settings, and switching back restores chat', (context) => {
  const ui = images(context);
  assert.equal(ui.element('imageSettings').hidden, false);
  assert.equal(ui.element('chatSettings').hidden, true);
  assert.equal(ui.element('newChatLabel').textContent, 'New image');
  assert.equal(ui.element('prompt').getAttribute('aria-label'), 'Image description');
  ui.click('imageSettingsButton');
  assert.equal(ui.window.document.activeElement, ui.element('imageSize'));
  ui.window.setModel('Frisky');
  assert.equal(ui.element('imageSettings').hidden, true);
  assert.equal(ui.element('chatSettings').hidden, false);
  assert.equal(ui.element('newChatLabel').textContent, 'New chat');
  assert.equal(ui.element('prompt').placeholder, 'Ask DeepJ');
});

test('image generation uses image settings and ignores text sampling controls', (context) => {
  const ui = images(context);
  ui.element('prompt').value = 'A fox';
  ui.element('imageSize').value = '1344x768';
  ui.element('imageSize').dispatchEvent(new ui.window.Event('change'));
  ui.element('temperature').value = '';
  ui.window.submitPrompt();
  assert.deepEqual(ui.calls, [
    ['generateWithPrecision', 'A fox', 1344, 768, 2, 42, 4.5, 'FLOAT32'],
  ]);
  assert.equal(ui.element('imageSizeHint').textContent, '1344 × 768');
});

test('compact image precision reaches the model bridge', (context) => {
  const ui = images(context);
  ui.element('prompt').value = 'A fox';
  ui.element('imageDtype').value = 'BFLOAT16';
  ui.window.submitPrompt();
  assert.equal(ui.calls[0].at(-1), 'BFLOAT16');
});

test('invalid image settings focus the field before dispatch', (context) => {
  const ui = images(context);
  ui.element('prompt').value = 'A fox';
  ui.element('imageSeed').value = '';
  ui.window.submitPrompt();
  assert.equal(ui.window.document.activeElement, ui.element('imageSeed'));
  assert.equal(ui.element('settingsBackdrop').hidden, false);
  ui.element('imageSeed').value = '42';
  ui.element('imageGuidance').value = '21';
  ui.window.submitPrompt();
  assert.equal(ui.window.document.activeElement, ui.element('imageGuidance'));
  assert.equal(ui.calls.length, 0);
});

test('image folder picker uses the desktop bridge and reports preview unavailability', (context) => {
  const ui = images(context);
  ui.click('menuLoadImage');
  assert.deepEqual(ui.calls, [['chooseModel']]);
  delete ui.window.imageBridge;
  ui.click('menuLoadImage');
  assert.match(ui.element('toast').textContent, /not available/);
});

test('real image stages produce readable status without exposing implementation names', (context) => {
  const ui = images(context);
  ui.window.startImage('A fox', 3);
  assert.ok(ui.element('response-image-3').querySelector('.image-placeholder'));
  const labels = ['Preparing', 'Preparing', 'Creating', 'Finishing', 'Other'];
  [
    'Encoding prompt with Gemma 2',
    'Loading SANA-Sprint transformer',
    'Denoising step 1/2',
    'Decoding image with DC-AE',
    'Other',
  ].forEach((stage, index) => {
    ui.window.imageProgress(3, stage);
    assert.ok(ui.element('status').textContent.startsWith(labels[index]));
  });
  ui.window.imageProgress(99, 'Ignored');
});

test('finished image offers a single consistent save action for its own response', (context) => {
  const ui = images(context);
  ui.window.startImage('A fox <script>', 5);
  ui.window.finishImage(5, 'data:image/png;base64,aGVsbG8=', 1024, 1024, 2, 42);
  const response = ui.element('response-image-5');
  const image = response.querySelector('img');
  assert.equal(image.width, 1024);
  assert.equal(response.querySelectorAll('.image-save').length, 1);
  assert.equal(response.querySelector('.copy-button'), null);
  assert.equal(response.querySelector('.image-meta').textContent, '1024 × 1024 · 2 steps');
  response.querySelector('.image-save').click();
  image.dispatchEvent(new ui.window.Event('load'));
  assert.deepEqual(ui.calls, [['save', 5]]);
});

test('stopping, cancelling and failures keep the image card and composer coherent', (context) => {
  const ui = images(context);
  ui.window.startImage('A fox', 4);
  ui.click('send');
  assert.deepEqual(ui.calls, [['stop']]);
  ui.window.stoppingImage(4);
  assert.equal(ui.element('send').disabled, true);
  ui.window.cancelImage(4);
  assert.match(ui.element('response-image-4').textContent, /stopped/);
  ui.window.failImage(4, 'Insufficient memory');
  assert.match(ui.element('response-image-4').textContent, /Couldn’t generate an image/);
  ui.window.setGenerating(false);
  ui.window.setBusy(false);
  assert.equal(ui.element('send').title, 'Generate image');
});

test('cleared image callbacks are harmless and new image clears previous output', (context) => {
  const ui = images(context);
  finish(ui);
  ui.window.finishImage(99, 'ignored', 32, 32, 1, 1);
  ui.window.cancelImage(99);
  ui.window.failImage(99, 'ignored');
  ui.window.startImage('A fox', 1);
  ui.window.finishImage(1, 'data:image/png;base64,aGVsbG8=', 32, 32, 1, 1);
  ui.click('newChat');
  assert.equal(ui.element('messages').children.length, 0);
  ui.window.setImageModel('SANA-Sprint');
  ui.window.setBusy(true);
  assert.equal(ui.element('menuLoadImage').disabled, true);
  assert.equal(ui.element('agentControls').hidden, true);
});

test('generated images remain available to save after switching back to a chat model', (context) => {
  const ui = images(context);
  ui.window.startImage('A fox', 1);
  ui.window.finishImage(1, 'data:image/png;base64,aGVsbG8=', 32, 32, 1, 1);
  ui.window.setModel('Frisky');
  ui.element('response-image-1').querySelector('.image-save').click();
  assert.deepEqual(ui.calls, [['save', 1]]);
});
