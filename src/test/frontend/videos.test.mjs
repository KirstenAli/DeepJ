import test from 'node:test';
import assert from 'node:assert/strict';
import { fixture } from './fixture.mjs';

function videos(context) {
  const ui = fixture(context);
  ui.window.videoBridge = Object.fromEntries(
    ['chooseModel', 'generate', 'stop', 'save'].map((name) => [
      name,
      (...args) => ui.calls.push([name, ...args]),
    ]),
  );
  ui.window.setVideoModel('SANA-Video 2B');
  return ui;
}

test('video mode uses its own validated options and restores chat when switching', (context) => {
  const ui = videos(context);
  ui.element('prompt').value = 'A fox walking';
  ui.window.submitPrompt();
  assertVideoDefaults(ui);
  assert.equal(ui.element('send').title, 'Generate video');
  ui.click('videoSettingsButton');
  assert.equal(ui.window.document.activeElement, ui.element('videoSize'));
  ui.window.setModel('Frisky');
  assert.equal(ui.element('videoSettings').hidden, true);
  assert.equal(ui.element('prompt').placeholder, 'Ask DeepJ');
});

function assertVideoDefaults(ui) {
  assert.deepEqual(JSON.parse(ui.calls[0][2]), {
    width: 832,
    height: 480,
    frames: 9,
    steps: 50,
    fps: 16,
    seed: 42,
    guidance: 6,
    dtype: 'FLOAT32',
  });
}

test('video precision is explicitly selected and included in the worker request', (context) => {
  const ui = videos(context);
  ui.element('videoDtype').value = 'BFLOAT16';
  ui.element('prompt').value = 'A fox walking';
  ui.window.submitPrompt();
  const request = JSON.parse(ui.calls[0][2]);
  assert.equal(request.dtype, 'BFLOAT16');
  assert.equal(request.width, 832);
  assert.equal(request.height, 480);
});

test('invalid frame counts and blank settings never reach the generator', (context) => {
  const ui = videos(context);
  ui.element('prompt').value = 'A fox';
  for (const value of ['8', '82', '']) {
    ui.element('videoFrames').value = value;
    ui.window.submitPrompt();
    assert.equal(ui.window.document.activeElement, ui.element('videoFrames'));
  }
  assert.deepEqual(ui.calls, []);
  ui.element('videoFrames').value = '17';
  ui.element('videoFrames').dispatchEvent(new ui.window.Event('change'));
  ui.element('videoSize').value = '480x832';
  ui.element('videoSize').dispatchEvent(new ui.window.Event('change'));
  assert.equal(ui.element('videoSizeHint').textContent, '480 × 832 · 17 frames');
});

test('video picker and generation progress use consistent desktop controls', (context) => {
  const ui = videos(context);
  ui.click('menuLoadVideo');
  assert.deepEqual(ui.calls, [['chooseModel']]);
  delete ui.window.videoBridge;
  ui.click('menuLoadVideo');
  assert.match(ui.element('toast').textContent, /not available/);
  ui.window.startVideo('A fox', 2);
  [
    'Encoding Gemma',
    'Loading model',
    'Video denoising step 2/50',
    'Decoding frames',
    'Ready',
  ].forEach((text) => ui.window.videoProgress(2, text));
  assert.equal(ui.element('status').textContent, 'Ready');
  ui.window.videoProgress(99, 'Ignored');
});

test('completed video exposes themed controls and the MP4 download bridge', (context) => {
  const ui = videos(context);
  ui.window.startVideo('A fox', 3);
  ui.window.finishVideo(
    3,
    JSON.stringify(['file:///tmp/0.png', 'file:///tmp/1.png']),
    832,
    480,
    16,
  );
  const player = ui.element('response-video-3');
  assert.equal(player.querySelector('.video-frame').src, 'file:///tmp/0.png');
  assert.equal(player.querySelector('.video-play').getAttribute('aria-label'), 'Play video');
  player.querySelector('.video-save').click();
  assert.deepEqual(ui.calls, [['save', 3]]);
  ui.window.newChat();
  assert.equal(ui.element('response-video-3'), null);
  ui.window.finishVideo(99, '[]', 32, 32, 16);
});

test('stop, cancellation, and failure keep the video UI usable', (context) => {
  const ui = videos(context);
  ui.window.startVideo('A fox', 4);
  ui.click('send');
  assert.deepEqual(ui.calls, [['stop']]);
  ui.window.stoppingVideo(4);
  assert.equal(ui.element('send').disabled, true);
  ui.window.cancelVideo(4);
  assert.match(ui.element('response-video-4').textContent, /stopped/);
  ui.window.failVideo(4, 'Not enough memory');
  assert.match(ui.element('response-video-4').textContent, /Couldn’t generate a video/);
  ui.window.cancelVideo(99);
  ui.window.failVideo(99, 'Ignored');
});
