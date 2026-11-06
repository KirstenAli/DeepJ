import { readFileSync } from 'node:fs';
import { Script } from 'node:vm';
import { URL, fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const html = readFileSync(new URL('../../main/resources/chat.html', import.meta.url), 'utf8');
const source = new URL('../../main/resources/chat.js', import.meta.url);
const imageSource = new URL('../../main/resources/images.js', import.meta.url);
const imageScript = new Script(readFileSync(imageSource, 'utf8'), {
  filename: fileURLToPath(imageSource),
});
const script = new Script(readFileSync(source, 'utf8'), { filename: fileURLToPath(source) });

function fakeTimers(window) {
  const pending = new Map();
  let next = 1;
  window.setTimeout = (callback) => {
    pending.set(next, callback);
    return next++;
  };
  window.clearTimeout = (id) => pending.delete(id);
  return () => {
    const callbacks = [...pending.values()];
    pending.clear();
    callbacks.forEach((callback) => callback());
  };
}

export function fixture(context) {
  const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true });
  const window = dom.window;
  const calls = [];
  const timers = fakeTimers(window);
  window.chatBridge = recordingBridge(calls);
  window.agentBridge = {
    setEnabled: (enabled) => calls.push(['setEnabled', enabled]),
    chooseDirectory: () => {
      calls.push(['chooseDirectory']);
      return '/tmp/other project';
    },
  };
  script.runInContext(dom.getInternalVMContext());
  imageScript.runInContext(dom.getInternalVMContext());
  extraScripts(dom);
  context.after(() => window.close());
  const element = (id) => window.document.getElementById(id);
  return { window, calls, timers, element, click: (id) => element(id).click() };
}

function extraScripts(dom) {
  ['thinking', 'videos', 'video-player'].forEach((name) => {
    const file = new URL('../../main/resources/' + name + '.js', import.meta.url);
    new Script(readFileSync(file, 'utf8'), { filename: fileURLToPath(file) }).runInContext(
      dom.getInternalVMContext(),
    );
  });
}

function recordingBridge(calls) {
  return Object.fromEntries(
    [
      'chooseModel',
      'sendWithSettings',
      'sendWithThinkingSettings',
      'sendWithWindowSettings',
      'setConversationHistory',
      'stop',
      'copy',
      'newChat',
      'alternatives',
    ].map((name) => [name, (...args) => calls.push([name, ...args])]),
  );
}

export function finish(ui, id = 1, text = 'A useful response') {
  ui.window.startResponse('Hello', id);
  ui.window.finishResponse(id, text);
  return ui.element('response-' + id);
}

export function keyboard(ui, target, key, shiftKey = false) {
  const event = new ui.window.KeyboardEvent('keydown', {
    key,
    shiftKey,
    bubbles: true,
    cancelable: true,
  });
  target.dispatchEvent(event);
  return event;
}

export function scrollAway(ui) {
  Object.defineProperties(ui.element('conversation'), {
    scrollHeight: { value: 1000, configurable: true },
    clientHeight: { value: 100, configurable: true },
  });
  ui.element('conversation').scrollTop = 0;
}
