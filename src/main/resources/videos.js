let videoMode = false;

function setVideoMode(enabled) {
  videoMode = enabled;
  ['videoModeLabel', 'videoSettingsButton', 'videoSettings'].forEach(
    (id) => (element(id).hidden = !enabled),
  );
  if (!enabled) return;
  element('chatSettings').hidden = true;
  element('agentControls').hidden = true;
  element('newChatLabel').textContent = 'New video';
  promptBox.placeholder = 'Describe the video you want to create';
  promptBox.setAttribute('aria-label', 'Video description');
  element('welcome').querySelector('h1').textContent = 'Create a video';
  element('welcome').querySelector('p').textContent = 'Describe a scene and its movement.';
  document.querySelector('.disclaimer').textContent =
    'Videos are generated locally on your device.';
}

function setVideoModel(name) {
  setModel(name);
  setVideoMode(true);
  updateSend();
}

function sendLabel() {
  if (generating) return 'Stop generating';
  if (videoMode) return 'Generate video';
  return imageMode ? 'Generate image' : 'Send message';
}

function stopActiveMedia() {
  if (videoMode) return videoBridge.stop();
  return imageMode ? imageBridge.stop() : chatBridge.stop();
}

function chooseVideoModel() {
  closeModelMenu();
  if (window.videoBridge) videoBridge.chooseModel();
  else toast('Desktop model picker is not available in this preview.');
}

function submitVideo(prompt) {
  const fields = ['Frames', 'Steps', 'Fps', 'Seed', 'Guidance'].map((name) =>
    element('video' + name),
  );
  const invalid = fields.find((field) => !field.value.trim() || !field.checkValidity());
  if (invalid) {
    openSettings();
    invalid.focus();
    toast('Check the video settings before generating.');
    return;
  }
  const [width, height] = element('videoSize').value.split('x').map(Number);
  const options = Object.fromEntries(
    fields.map((field) => [field.id.slice(5).toLowerCase(), Number(field.value)]),
  );
  videoBridge.generate(prompt, JSON.stringify({ width, height, ...options }));
}

function startVideo(prompt, id) {
  startResponse(prompt, 'video-' + id);
  const content = element('response-video-' + id).querySelector('.message-content');
  content.replaceChildren(imagePlaceholder(), imageProgressIndicator());
  content.querySelector('.image-placeholder').replaceChildren(icon('video'));
  setGenerating(true);
  videoProgress(id, 'Preparing your video…');
}

function videoProgress(id, text) {
  const label = element('response-video-' + id)?.querySelector('.image-progress-label');
  if (!label) return;
  label.textContent = videoStage(text);
  setStatus(label.textContent);
}

function videoStage(text) {
  if (text.includes('Gemma')) return 'Preparing your prompt…';
  if (text.includes('Loading')) return 'Preparing the video model…';
  if (text.includes('Decoding')) return 'Finishing your video…';
  const step = text.match(/denoising step (\d+)\/(\d+)/);
  return step ? 'Creating your video · Step ' + step[1] + ' of ' + step[2] : text;
}

function finishVideo(id, urls, width, height, fps) {
  const response = element('response-video-' + id);
  if (!response) return;
  response.classList.remove('pending');
  const content = response.querySelector('.message-content');
  content.classList.add('image-card');
  const frames = JSON.parse(urls);
  content.replaceChildren(
    videoPlayer(id, frames, width, height, fps),
    videoActions(id, frames.length, fps),
  );
  followLatest();
}

function videoActions(id, frames, fps) {
  const footer = document.createElement('div');
  footer.className = 'image-actions';
  const info = document.createElement('span');
  info.className = 'image-meta';
  info.textContent = (frames / fps).toFixed(2) + ' s · ' + fps + ' FPS · MP4';
  const save = document.createElement('button');
  save.type = 'button';
  save.className = 'done-button video-save';
  save.append(icon('download'), document.createTextNode('Save video'));
  save.addEventListener('click', () => videoBridge.save(id));
  footer.append(info, save);
  return footer;
}

function stoppingVideo(id) {
  videoProgress(id, 'Stopping video generation…');
  element('send').disabled = true;
}

function cancelVideo(id) {
  const response = element('response-video-' + id);
  if (!response) return;
  response.classList.remove('pending');
  response.querySelector('.message-content').textContent = 'Video generation stopped.';
}

function failVideo(id, message) {
  const response = element('response-video-' + id);
  showFailure(response, message);
  const title = response?.querySelector('.error-title');
  if (title) title.textContent = 'Couldn’t generate a video';
}

function videoSizeHint() {
  element('videoSizeHint').textContent =
    element('videoSize').value.replace('x', ' × ') +
    ' · ' +
    element('videoFrames').value +
    ' frames';
}

element('menuLoadVideo').onclick = chooseVideoModel;
element('videoSettingsButton').onclick = () => {
  openSettings();
  element('videoSize').focus();
};
element('videoSize').onchange = videoSizeHint;
element('videoFrames').onchange = videoSizeHint;
