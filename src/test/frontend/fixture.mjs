import { readFileSync } from 'node:fs';
import { Script } from 'node:vm';
import { URL, fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const html = readFileSync(new URL('../../main/resources/chat.html', import.meta.url), 'utf8');
const source = new URL('../../main/resources/chat.js', import.meta.url);
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
  window.chatBridge = Object.fromEntries(
    ['chooseModel', 'send', 'stop', 'copy', 'newChat', 'alternatives'].map((name) => [
      name,
      (...args) => calls.push([name, ...args]),
    ]),
  );
  script.runInContext(dom.getInternalVMContext());
  context.after(() => window.close());
  const element = (id) => window.document.getElementById(id);
  return { window, calls, timers, element, click: (id) => element(id).click() };
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
