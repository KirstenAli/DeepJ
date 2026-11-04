const app = document.getElementById('app');
const promptBox = document.getElementById('prompt');
const conversation = document.getElementById('conversation');
const messages = document.getElementById('messages');
const modelMenu = document.getElementById('modelMenu');
const agentMenu = document.getElementById('agentMenu');
const settings = document.getElementById('settingsBackdrop');
const approval = document.getElementById('approvalBackdrop');
let modelLoaded = false;
let busy = false;
let generating = false;
let toastTimer;
let approvalFocus;

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
  ['composerModel', 'menuLoad', 'agentPicker'].forEach((id) => (element(id).disabled = value));
  if (value) showAgentMenu(false);
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
  const title = document.createElement('strong');
  title.className = 'error-title';
  title.textContent = 'Couldn’t generate a response';
  response.prepend(title);
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
  sendPrompt(prompt);
}

function sendPrompt(prompt) {
  const values = [
    prompt,
    generationValue('maxTokens'),
    generationValue('temperature'),
    generationValue('topK'),
    generationValue('seed'),
    generationValue('repetitionPenalty'),
  ];
  if (element('rollingCache').checked) chatBridge.sendWithWindowSettings(...values, true);
  else chatBridge.sendWithSettings(...values);
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

function setAgentWorkspace(directory) {
  element('agentControls').hidden = !directory;
  showAgentMenu(false);
  element('agentMode').checked = false;
  showAgentMode(false);
  setAgentDirectory(directory);
}

function setAgentDirectory(directory) {
  const path = element('agentDirectory');
  path.textContent = directory.slice(directory.lastIndexOf('/') + 1) || directory;
  path.title = directory;
}

function showAgentMenu(open) {
  agentMenu.hidden = !open;
  element('agentPicker').setAttribute('aria-expanded', String(open));
}

function showAgentMode(enabled) {
  element('agentLabel').textContent = enabled ? 'Agent on' : 'Tools';
  element('agentPicker').classList.toggle('agent-on', enabled);
}

function changeAgentMode() {
  const enabled = element('agentMode').checked;
  window.agentBridge.setEnabled(enabled);
  showAgentMode(enabled);
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

function openApproval(command, directory) {
  approvalFocus = document.activeElement;
  element('approvalCommand').textContent = command;
  element('approvalDirectory').textContent = directory;
  element('approvalDescription').textContent = 'The agent is asking to run this command once.';
  setCommandEditing(false);
  approval.hidden = false;
  element('denyCommand').focus();
}

function closeApproval() {
  approval.hidden = true;
  approvalFocus?.focus();
  approvalFocus = null;
}

function decideCommand(allow) {
  window.approvalBridge.decide(allow);
}

function setCommandEditing(editing) {
  element('approvalPreview').hidden = editing;
  element('approvalActions').hidden = editing;
  element('commandEditor').hidden = !editing;
}

function editCommand() {
  const [executable, ...args] = JSON.parse(element('approvalCommand').textContent);
  const field = element('commandExecutable');
  fillCommandField(field, executable);
  element('commandArguments').replaceChildren();
  args.forEach(addCommandArgument);
  element('commandEditError').hidden = true;
  setCommandEditing(true);
  field.focus();
}

function fillCommandField(field, value) {
  field.value = value;
  field.dataset.original = value;
  field.dataset.initial = field.value;
}

function commandFieldValue(field) {
  return field.value === field.dataset.initial ? field.dataset.original : field.value;
}

function addCommandArgument(value) {
  const row = element('commandArgumentTemplate').content.firstElementChild.cloneNode(true);
  const field = row.querySelector('textarea');
  fillCommandField(field, value);
  row.querySelector('button').onclick = () => {
    row.remove();
    element('addCommandArgument').focus();
  };
  element('commandArguments').append(row);
  return field;
}

function reviewCommandEdit(event) {
  event.preventDefault();
  const fields = [...element('commandEditor').querySelectorAll('textarea')];
  const argv = fields.map(commandFieldValue);
  const result = JSON.parse(window.approvalBridge.reviewEdit(JSON.stringify(argv)));
  if (result.error) {
    const error = element('commandEditError');
    error.textContent = result.error;
    error.hidden = false;
    return;
  }
  element('approvalCommand').textContent = result.preview;
  element('approvalDescription').textContent = 'Review your edited command before running it.';
  setCommandEditing(false);
  element('denyCommand').focus();
}

function trapModalTab(event) {
  if (event.key !== 'Tab') return;
  const controls = [
    ...event.currentTarget.querySelectorAll('button, input, textarea, summary, [tabindex="0"]'),
  ].filter((control) => control.offsetParent !== null);
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
element('agentPicker').onclick = () => showAgentMenu(agentMenu.hidden);
element('agentMode').onchange = changeAgentMode;
element('chooseAgentDirectory').onclick = () =>
  setAgentDirectory(window.agentBridge.chooseDirectory());
element('openSettings').onclick = openSettings;
element('closeSettings').onclick = closeSettings;
element('doneSettings').onclick = closeSettings;
element('denyCommand').onclick = () => decideCommand(false);
element('allowCommand').onclick = () => decideCommand(true);
element('editCommand').onclick = editCommand;
element('addCommandArgument').onclick = () => addCommandArgument('').focus();
element('cancelCommandEdit').onclick = () => {
  setCommandEditing(false);
  element('editCommand').focus();
};
element('commandEditor').onsubmit = reviewCommandEdit;
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
settings.addEventListener('keydown', trapModalTab);
approval.addEventListener('keydown', trapModalTab);
document
  .querySelectorAll('.theme-choice')
  .forEach((button) => (button.onclick = () => setTheme(button.dataset.theme)));
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  if (!approval.hidden) return decideCommand(false);
  closeSettings();
  closeModelMenu();
  if (!agentMenu.hidden) {
    showAgentMenu(false);
    element('agentPicker').focus();
  }
});
document.addEventListener('click', (event) => {
  if (!event.target.closest('.model-control')) closeModelMenu();
  if (!event.target.closest('#agentControls')) showAgentMenu(false);
});
