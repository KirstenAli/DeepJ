const app = document.getElementById('app');
const promptBox = document.getElementById('prompt');
const conversation = document.getElementById('conversation');
const messages = document.getElementById('messages');
const modelMenu = document.getElementById('modelMenu');
const settings = document.getElementById('settingsBackdrop');
let modelLoaded = false;
let busy = false;
let generating = false;
let toastTimer;

function element(id) {
  return document.getElementById(id);
}
function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', '#icon-' + name);
  svg.append(use);
  return svg;
}

function toast(message) {
  const node = element('toast');
  node.textContent = message;
  node.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove('visible'), 3500);
}

function setStatus(message) {
  element('status').textContent = message;
  app.classList.toggle('ready', modelLoaded);
}

function setModel(name) {
  modelLoaded = true;
  element('modelName').textContent = name;
  element('composerHint').textContent = name;
  element('welcome').querySelector('p').textContent = 'Your local model is ready.';
  updateSend();
}

function setBusy(value) {
  busy = value;
  promptBox.disabled = value;
  ['composerModel', 'menuLoad'].forEach((id) => (element(id).disabled = value));
  document.querySelectorAll('.alternatives-button').forEach((button) => (button.disabled = value));
  updateSend();
}

function setGenerating(value) {
  generating = value;
  updateSend();
}

function updateSend() {
  const button = element('send');
  button.disabled = !generating && (busy || !promptBox.value.trim());
  button.replaceChildren(icon(generating ? 'stop' : 'send'));
  button.setAttribute('aria-label', generating ? 'Stop generating' : 'Send message');
  button.title = generating ? 'Stop generating' : 'Send message';
}

function resizePrompt() {
  promptBox.style.height = 'auto';
  promptBox.style.height = Math.min(promptBox.scrollHeight, 138) + 'px';
  updateSend();
}

function followLatest() {
  conversation.scrollTop = conversation.scrollHeight;
}

function nearLatest() {
  return conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight < 90;
}

function addMessage(kind, content) {
  const message = document.createElement('article');
  const text = document.createElement('div');
  message.className = 'message ' + kind;
  text.className = 'message-content';
  text.textContent = content;
  message.append(text);
  messages.append(message);
  return message;
}

function addCopy(message) {
  const actions = document.createElement('div');
  const button = document.createElement('button');
  button.className = 'copy-button';
  button.setAttribute('aria-label', 'Copy response');
  button.title = 'Copy response';
  button.append(icon('copy'));
  button.addEventListener('click', () => copyResponse(message, button));
  actions.className = 'message-actions';
  actions.append(button);
  message.append(actions);
}

function copyResponse(message, button) {
  if (!window.chatBridge) return;
  chatBridge.copy(message.querySelector('.message-content').textContent);
  button.replaceChildren(icon('check'));
  button.title = 'Copied';
  button.setAttribute('aria-label', 'Copied');
  setTimeout(() => resetCopy(button), 1800);
}

function resetCopy(button) {
  button.replaceChildren(icon('copy'));
  button.title = 'Copy response';
  button.setAttribute('aria-label', 'Copy response');
}

function addAlternativesButton(message, id) {
  const button = document.createElement('button');
  button.className = 'alternatives-button';
  button.textContent = 'See alternative responses';
  button.disabled = busy;
  button.addEventListener('click', () => toggleAlternatives(message, button, id));
  message.querySelector('.message-actions').append(button);
}

function toggleAlternatives(message, button, id) {
  let panel = message.querySelector('.alternatives');
  if (!panel) {
    panel = document.createElement('div');
    panel.className = 'alternatives';
    panel.setAttribute('aria-label', 'Alternative responses');
    message.append(panel);
    chatBridge.alternatives(id);
  } else panel.hidden = !panel.hidden;
  button.textContent = panel.hidden ? 'See alternative responses' : 'Hide alternative responses';
}

function clearAlternativeActions() {
  document.querySelectorAll('.alternatives-button').forEach((button) => button.remove());
}

function startResponse(prompt, id) {
  app.classList.remove('empty');
  addMessage('user', prompt);
  const response = addMessage('assistant pending', 'Thinking…');
  response.id = 'response-' + id;
  promptBox.value = '';
  resizePrompt();
  followLatest();
}

function updateResponse(id, text) {
  updateGenerated(element('response-' + id), text);
}

function updateGenerated(response, text) {
  if (!response) return;
  const follow = nearLatest();
  response.classList.remove('pending');
  response.querySelector('.message-content').textContent = text;
  if (!response.querySelector('.message-actions')) addCopy(response);
  if (follow) followLatest();
}

function finishResponse(id, text) {
  if (text && text.trim()) {
    updateResponse(id, text);
    addAlternativesButton(element('response-' + id), id);
    return;
  }
  element('response-' + id)?.remove();
  toast('No text generated. Try another prompt.');
}

function alternativeId(id, option) {
  return 'response-' + id + '-alternative-' + option;
}

function startAlternative(id, option) {
  const follow = nearLatest();
  const alternative = document.createElement('div');
  alternative.id = alternativeId(id, option);
  alternative.className = 'alternative pending';
  alternative.setAttribute('aria-label', 'Alternative response');
  const content = document.createElement('div');
  content.className = 'message-content';
  content.textContent = 'Thinking…';
  alternative.append(content);
  element('response-' + id)
    .querySelector('.alternatives')
    .append(alternative);
  if (follow) followLatest();
}

function updateAlternative(id, option, text) {
  updateGenerated(element(alternativeId(id, option)), text);
}

function finishAlternative(id, option, text) {
  if (text && text.trim()) return updateAlternative(id, option, text);
  failAlternative(id, option, 'No text generated.');
}

function failAlternative(id, option, message) {
  showFailure(element(alternativeId(id, option)), message);
}

function stopAlternative(id, option) {
  const alternative = element(alternativeId(id, option));
  if (alternative?.classList.contains('pending')) alternative.remove();
  toast('Generation stopped.');
}

function failResponse(id, message) {
  showFailure(element('response-' + id), message);
}

function showFailure(response, message) {
  if (!response) return;
  response.classList.remove('pending');
  response.classList.add('error');
  response.querySelector('.message-content').textContent = message;
  response.querySelector('.message-actions')?.remove();
  followLatest();
}

function stopResponse(id) {
  const response = element('response-' + id);
  if (response?.classList.contains('pending')) response.remove();
  toast('Generation stopped.');
}

function generationValue(id) {
  return Number(element(id).value);
}

function validGenerationSettings() {
  const invalid = ['temperature', 'topK', 'maxTokens', 'seed', 'repetitionPenalty']
    .map(element)
    .find((input) => !input.value.trim() || !input.checkValidity());
  if (!invalid) return true;
  settings.hidden = false;
  element('generationOptions').open = true;
  invalid.focus();
  toast('Check the generation settings before sending.');
  return false;
}

function submitPrompt() {
  const prompt = promptBox.value.trim();
  if (!prompt || busy) return;
  if (!modelLoaded) return chooseModel();
  if (!validGenerationSettings()) return;
  chatBridge.sendWithSettings(
    prompt,
    generationValue('maxTokens'),
    generationValue('temperature'),
    generationValue('topK'),
    generationValue('seed'),
    generationValue('repetitionPenalty'),
  );
}

function chooseModel() {
  closeModelMenu();
  if (window.chatBridge) chatBridge.chooseModel();
  else toast('Desktop model picker is not available in this preview.');
}

function toggleModelMenu() {
  modelMenu.hidden = !modelMenu.hidden;
  element('modelPicker').setAttribute('aria-expanded', String(!modelMenu.hidden));
}

function closeModelMenu() {
  modelMenu.hidden = true;
  element('modelPicker').setAttribute('aria-expanded', 'false');
}

function newChat() {
  if (window.chatBridge) chatBridge.newChat();
  messages.replaceChildren();
  app.classList.add('empty');
  promptBox.value = '';
  resizePrompt();
  promptBox.focus();
}

function submitOrStop() {
  if (generating) return chatBridge.stop();
  submitPrompt();
}

function setTheme(name) {
  document.body.className = 'theme-' + name;
  document
    .querySelectorAll('.theme-choice')
    .forEach((button) => button.classList.toggle('selected', button.dataset.theme === name));
}

function openSettings() {
  settings.hidden = false;
  element('closeSettings').focus();
}

function closeSettings() {
  if (settings.hidden) return;
  settings.hidden = true;
  element('openSettings').focus();
}

function trapSettingsTab(event) {
  if (event.key !== 'Tab') return;
  const controls = [...settings.querySelectorAll('button, input, summary')].filter(
    (control) => control.offsetParent !== null,
  );
  const last = controls[controls.length - 1];
  const target = event.shiftKey ? last : controls[0];
  const boundary = event.shiftKey ? controls[0] : last;
  if (document.activeElement !== boundary) return;
  target.focus();
  event.preventDefault();
}
function bridgeReady() {
  setStatus('Choose a model to begin');
}

element('collapseSidebar').onclick = () => app.classList.add('collapsed');
element('reopenSidebar').onclick = () => app.classList.remove('collapsed');
element('newChat').onclick = newChat;
element('composerModel').onclick = chooseModel;
element('menuLoad').onclick = chooseModel;
element('modelPicker').onclick = toggleModelMenu;
element('openSettings').onclick = openSettings;
element('closeSettings').onclick = closeSettings;
element('doneSettings').onclick = closeSettings;
element('send').onclick = submitOrStop;
promptBox.oninput = resizePrompt;
window.addEventListener('resize', resizePrompt);
promptBox.onkeydown = (event) => {
  if (event.key !== 'Enter' || event.shiftKey) return;
  event.preventDefault();
  submitPrompt();
};
settings.onclick = (event) => {
  if (event.target === settings) closeSettings();
};
settings.addEventListener('keydown', trapSettingsTab);
document
  .querySelectorAll('.theme-choice')
  .forEach((button) => (button.onclick = () => setTheme(button.dataset.theme)));
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  closeSettings();
  closeModelMenu();
});
document.addEventListener('click', (event) => {
  if (!event.target.closest('.model-control')) closeModelMenu();
});
