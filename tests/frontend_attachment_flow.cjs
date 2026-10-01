const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function element() {
  const listeners = {};
  const style = {};
  const el = {
    listeners, children: [], dataset: {}, files: [], value: '', textContent: '',
    disabled: false, checked: false,
    classList: {add() {}, remove() {}},
    addEventListener(name, callback) {listeners[name] = callback;},
    appendChild(child) {this.children.push(child);},
    async dispatch(name) {await listeners[name]?.({preventDefault() {}});},
  };
  Object.defineProperty(el, 'style', {
    get() {return style;}, set(css) {style.cssText = css;},
  });
  Object.defineProperty(el, 'innerHTML', {
    get() {return '';}, set() {el.children = [];},
  });
  return el;
}

async function main() {
  const nodes = Object.fromEntries([
    'askBtn', 'attachInput', 'attachList', 'attachMsg', 'clearAttachmentsBtn',
    'useUploads', 'question', 'askMsg', 'answerPre', 'rationalePre',
  ].map(id => [id, element()]));
  let ready;
  let finishUpload;
  let failAsk = false;
  const questions = [];
  const document = {
    body: element(), head: element(),
    createElement: element,
    getElementById(id) {return nodes[id] || null;},
    addEventListener(name, callback) {if (name === 'DOMContentLoaded') ready = callback;},
  };
  const sandbox = {
    document, window: {BASE_PATH: '', location: {pathname: '/'}}, console,
    FormData: class {append() {}},
    fetch: async (url, options) => {
      if (url === '/attach') {
        return new Promise(resolve => {
          finishUpload = () => resolve({
            ok: true,
            text: async () => JSON.stringify({ok: true, items: [{id: 'selected_protocol'}]}),
          });
        });
      }
      assert.equal(url, '/ask');
      questions.push(JSON.parse(options.body));
      return {
        ok: !failAsk, status: failAsk ? 422 : 200,
        text: async () => JSON.stringify(failAsk
          ? {ok: false, error: 'Attachment unreadable', error_code: 'attachment_unreadable'}
          : {ok: true, answer: 'Protocol with an error estimate.', rationale: ''}),
      };
    },
  };
  vm.runInNewContext(fs.readFileSync('static/app.js', 'utf8'), sandbox);
  ready();
  nodes.question.value = 'Optimize the attached protocol.';
  nodes.attachInput.files = [{name: 'protocol.txt'}];
  const upload = nodes.attachInput.dispatch('change');
  assert.equal(nodes.askBtn.disabled, true);
  await nodes.askBtn.dispatch('click');
  assert.equal(questions.length, 0, 'Ask must wait for the attachment upload');
  finishUpload();
  await upload;
  assert.equal(nodes.askBtn.disabled, false);
  assert.equal(nodes.attachList.children[0].textContent, 'protocol.txt');

  await nodes.askBtn.dispatch('click');
  assert.deepEqual(questions[0].attachments, ['selected_protocol']);
  assert.equal(questions[0].use_uploads, false);
  assert.equal(nodes.askMsg.textContent, 'Done.');
  assert.equal(nodes.attachList.children.length, 1, 'Keep selected files after success');

  nodes.question.value = 'What if the hold is shorter?';
  nodes.useUploads.checked = true;
  await nodes.askBtn.dispatch('click');
  assert.deepEqual(questions[1].attachments, ['selected_protocol']);
  assert.equal(questions[1].use_uploads, true);

  failAsk = true;
  await nodes.askBtn.dispatch('click');
  assert.match(nodes.askMsg.textContent, /attachment_unreadable/);
  assert.equal(nodes.attachList.children.length, 1, 'Keep files available for a retry');
  failAsk = false;
  await nodes.askBtn.dispatch('click');
  assert.deepEqual(questions[3].attachments, ['selected_protocol']);

  await nodes.clearAttachmentsBtn.dispatch('click');
  assert.equal(nodes.attachList.children.length, 0);
  assert.equal(nodes.clearAttachmentsBtn.disabled, true);
  nodes.useUploads.checked = false;
  await nodes.askBtn.dispatch('click');
  assert.deepEqual(questions[4].attachments, []);
  assert.equal(questions[4].use_uploads, false);
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
