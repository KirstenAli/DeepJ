let imageMode = false;

function setImageMode(enabled) {
  imageMode = enabled;
  if (enabled) {
    element('agentControls').hidden = true;
    showAgentMenu(false);
  }
  element('imageSettingsButton').hidden = !enabled;
  element('imageSettings').hidden = !enabled;
  element('chatSettings').hidden = enabled;
  imageComposer(enabled);
  imageWelcome(enabled);
}

function imageComposer(enabled) {
  element('newChatLabel').textContent = enabled ? 'New image' : 'New chat';
  promptBox.placeholder = enabled ? 'Describe the image you want to create' : 'Ask DeepJ';
  promptBox.setAttribute('aria-label', enabled ? 'Image description' : 'Message DeepJ');
}

function imageWelcome(enabled) {
  element('welcome').querySelector('h1').textContent = enabled
    ? 'Create an image'
    : 'What can I help with?';
  element('welcome').querySelector('p').textContent = enabled
    ? 'Describe an image. DeepJ will create it locally.'
    : 'Your local model is ready.';
  document.querySelector('.disclaimer').textContent = enabled
    ? 'Images are generated locally on your device.'
    : 'DeepJ runs locally. Small models may produce inaccurate text.';
}

function setImageModel(name) {
  setModel(name);
  setImageMode(true);
  updateSend();
}

function chooseImageModel() {
  closeModelMenu();
  if (window.imageBridge) window.imageBridge.chooseModel();
  else toast('Desktop model picker is not available in this preview.');
}

function openImageSettings() {
  openSettings();
  element('imageSize').focus();
}

function validImageSettings() {
  const invalid = ['imageSeed', 'imageGuidance']
    .map(element)
    .find((field) => !field.value.trim() || !field.checkValidity());
  if (invalid) {
    openSettings();
    invalid.focus();
    toast('Check the image settings before generating.');
    return false;
  }
  return true;
}

function submitImage(prompt) {
  if (!validImageSettings()) return;
  const [width, height] = element('imageSize').value.split('x').map(Number);
  imageBridge.generateWithPrecision(
    prompt,
    width,
    height,
    Number(element('imageSteps').value),
    Number(element('imageSeed').value),
    Number(element('imageGuidance').value),
    element('imageDtype').value,
  );
}

function startImage(prompt, id) {
  startResponse(prompt, 'image-' + id);
  element('response-image-' + id).dataset.imagePrompt = prompt;
  setGenerating(true);
  setStatus('Preparing your image…');
  const content = element('response-image-' + id).querySelector('.message-content');
  content.replaceChildren(imagePlaceholder(), imageProgressIndicator());
}

function imagePlaceholder() {
  const placeholder = document.createElement('div');
  placeholder.className = 'image-placeholder';
  placeholder.setAttribute('aria-hidden', 'true');
  placeholder.append(icon('image'));
  return placeholder;
}

function imageProgressIndicator() {
  const status = document.createElement('div');
  status.className = 'image-progress';
  status.setAttribute('role', 'status');
  const spinner = document.createElement('span');
  spinner.className = 'image-spinner';
  const label = document.createElement('span');
  label.className = 'image-progress-label';
  label.textContent = 'Preparing your image…';
  status.append(spinner, label);
  return status;
}

function imageProgress(id, message) {
  const label = element('response-image-' + id)?.querySelector('.image-progress-label');
  if (!label) return;
  label.textContent = imageStage(message);
  setStatus(label.textContent);
}

function imageStage(message) {
  if (message.includes('Gemma')) return 'Preparing your prompt…';
  if (message.includes('Loading')) return 'Preparing the image model…';
  if (message.includes('Decoding')) return 'Finishing your image…';
  const step = message.match(/Denoising step (\d+)\/(\d+)/);
  return step ? 'Creating your image · Step ' + step[1] + ' of ' + step[2] : message;
}

function finishImage(id, dataUrl, width, height, steps, seed) {
  const response = element('response-image-' + id);
  if (!response) return;
  const follow = nearLatest();
  const content = response.querySelector('.message-content');
  response.classList.remove('pending');
  content.classList.add('image-card');
  content.replaceChildren(
    imagePreview(dataUrl, width, height, response.dataset.imagePrompt),
    imageActions(id, width, height, steps, seed),
  );
  if (follow) followLatest();
}

function imagePreview(dataUrl, width, height, prompt) {
  const image = document.createElement('img');
  image.className = 'generated-image';
  image.alt = 'Generated image: ' + prompt;
  image.width = width;
  image.height = height;
  image.onload = () => {
    if (nearLatest()) followLatest();
  };
  image.src = dataUrl;
  return image;
}

function imageActions(id, width, height, steps, seed) {
  const footer = document.createElement('div');
  footer.className = 'image-actions';
  const info = document.createElement('span');
  info.className = 'image-meta';
  info.textContent = width + ' × ' + height + ' · ' + steps + (steps === 1 ? ' step' : ' steps');
  info.title = 'Seed: ' + seed;
  footer.append(info, imageSave(id));
  return footer;
}

function imageSave(id) {
  const save = document.createElement('button');
  save.type = 'button';
  save.className = 'done-button image-save';
  save.title = 'Save as PNG';
  save.append(icon('download'), document.createTextNode('Save image'));
  save.addEventListener('click', () => imageBridge.save(id));
  return save;
}

function stoppingImage(id) {
  imageProgress(id, 'Stopping image generation…');
  element('send').disabled = true;
}

function cancelImage(id) {
  const response = element('response-image-' + id);
  if (response) {
    response.classList.remove('pending');
    response.querySelector('.message-content').textContent = 'Image generation stopped.';
  }
}

function failImage(id, message) {
  showFailure(element('response-image-' + id), message);
  const heading = element('response-image-' + id)?.querySelector('.error-title');
  if (heading) heading.textContent = 'Couldn’t generate an image';
}

element('menuLoadImage').onclick = chooseImageModel;
element('imageSettingsButton').onclick = openImageSettings;
element('imageSize').onchange = () => {
  element('imageSizeHint').textContent = element('imageSize').value.replace('x', ' × ');
};
