const app = document.getElementById('app');
const promptBox = document.getElementById('prompt');
const conversation = document.getElementById('conversation');
const messages = document.getElementById('messages');
const modelMenu = document.getElementById('modelMenu');
const settings = document.getElementById('settingsBackdrop');
let modelLoaded = false;
let busy = false;
let toastTimer;

function element(id) { return document.getElementById(id); }
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
  element('modelBadge').textContent = name;
  element('composerHint').textContent = name;
  element('welcome').querySelector('p').textContent = 'Your local model is ready.';
  updateSend();
}

function setBusy(value) {
  busy = value;
  promptBox.disabled = value;
  ['sidebarModel', 'composerModel', 'menuLoad'].forEach(id => element(id).disabled = value);
  updateSend();
}

function updateSend() {
  element('send').disabled = busy || !promptBox.value.trim();
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

function startResponse(prompt, id) {
  app.classList.remove('empty');
  addMessage('user', prompt);
  const response = addMessage('assistant pending', 'Thinking…');
  response.id = 'response-' + id;
  addCopy(response);
  promptBox.value = '';
  followLatest();
}

function updateResponse(id, text) {
  const response = element('response-' + id);
  if (!response) return;
  const follow = nearLatest();
  response.classList.remove('pending');
  response.querySelector('.message-content').textContent = text;
  if (follow) followLatest();
}

function finishResponse(id, text) {
  if (text && text.trim()) return updateResponse(id, text);
  element('response-' + id)?.remove();
  toast('No text generated. Try another prompt.');
}

function failResponse(id, message) {
  const response = element('response-' + id);
  response.classList.remove('pending');
  response.classList.add('error');
  response.querySelector('.message-content').textContent = message;
  followLatest();
}

function generationValue(id) { return Number(element(id).value); }
function submitPrompt() {
  const prompt = promptBox.value.trim();
  if (!prompt || busy) return;
  if (!modelLoaded) return chooseModel();
  chatBridge.send(prompt, generationValue('maxTokens'), generationValue('temperature'),
      generationValue('topK'), generationValue('seed'));
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
  messages.replaceChildren();
  app.classList.add('empty');
  promptBox.value = '';
  updateSend();
  promptBox.focus();
}

function setTheme(name) {
  document.body.className = 'theme-' + name;
  document.querySelectorAll('.theme-choice').forEach(button =>
      button.classList.toggle('selected', button.dataset.theme === name));
}

function closeSettings() { settings.hidden = true; }
function bridgeReady() { setStatus('Choose a model to begin'); }

element('collapseSidebar').onclick = () => app.classList.add('collapsed');
element('reopenSidebar').onclick = () => app.classList.remove('collapsed');
element('newChat').onclick = newChat;
element('sidebarModel').onclick = chooseModel;
element('composerModel').onclick = chooseModel;
element('menuLoad').onclick = chooseModel;
element('modelPicker').onclick = toggleModelMenu;
element('openSettings').onclick = () => settings.hidden = false;
element('closeSettings').onclick = closeSettings;
element('doneSettings').onclick = closeSettings;
element('send').onclick = submitPrompt;
promptBox.oninput = updateSend;
promptBox.onkeydown = event => {
  if (event.key !== 'Enter' || event.shiftKey) return;
  event.preventDefault();
  submitPrompt();
};
settings.onclick = event => { if (event.target === settings) closeSettings(); };
document.querySelectorAll('.theme-choice').forEach(button =>
    button.onclick = () => setTheme(button.dataset.theme));
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  closeSettings();
  closeModelMenu();
});
document.addEventListener('click', event => {
  if (!event.target.closest('.model-control')) closeModelMenu();
});
