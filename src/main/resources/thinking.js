function setThinkingAvailable(available) {
  element('thinkingRow').hidden = !available;
  element('thinkingMode').checked = false;
  showAgentMode(element('agentMode').checked);
  element('agentControls').hidden = !available && element('agentModeRow').hidden;
}

function showGenerationStatus(content) {
  if (element('thinkingMode').checked) {
    renderThinking(content, '<think>');
    return;
  }
  const status = document.createElement('span');
  status.className = 'generation-status';
  status.setAttribute('role', 'status');
  status.textContent = 'Preparing response…';
  content.replaceChildren(status);
}

function renderThinking(content, text) {
  let trimmed = text.trimStart();
  if (!trimmed) return;
  if ('<think>'.startsWith(trimmed)) trimmed = '<think>';
  if (!trimmed.startsWith('<think>')) {
    content.textContent = text;
    return;
  }
  const end = trimmed.indexOf('</think>');
  const panel = content.querySelector('.thinking-panel') || thinkingPanel(content);
  panel.querySelector('.thinking-content').textContent = trimmed
    .slice(7, end < 0 ? undefined : end)
    .trim();
  panel.querySelector('summary').textContent = end < 0 ? 'Thinking…' : 'Thought process';
  content.querySelector('.thinking-answer').textContent =
    end < 0 ? '' : trimmed.slice(end + 8).trimStart();
  collapseThinking(panel, end);
}

function collapseThinking(panel, end) {
  if (end < 0) {
    delete panel.dataset.complete;
  } else if (!panel.dataset.complete) {
    panel.open = false;
    panel.dataset.complete = 'true';
  }
}

function thinkingPanel(content) {
  const panel = document.createElement('details');
  panel.className = 'thinking-panel';
  panel.open = false;
  const summary = document.createElement('summary');
  const reasoning = document.createElement('div');
  reasoning.className = 'thinking-content';
  const answer = document.createElement('div');
  answer.className = 'thinking-answer';
  panel.append(summary, reasoning);
  content.replaceChildren(panel, answer);
  return panel;
}

function finishThinking(response) {
  const panel = response.querySelector('.thinking-panel');
  if (panel && !panel.dataset.complete) {
    panel.querySelector('summary').textContent = 'Thinking incomplete';
    response.querySelector('.thinking-answer').textContent =
      'The response ended before an answer. Increase maximum tokens or turn Thinking off.';
  }
}

function thinkingLabel(agent) {
  return agent ? 'Agent + thinking' : 'Thinking on';
}

element('thinkingMode').onchange = () => showAgentMode(element('agentMode').checked);
