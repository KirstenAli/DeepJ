import eslint from '@eslint/js';
import sonarjs from 'eslint-plugin-sonarjs';

const bridgeCallbacks = [
  'setModel',
  'setAgentWorkspace',
  'openApproval',
  'closeApproval',
  'setBusy',
  'setGenerating',
  'startResponse',
  'finishResponse',
  'failResponse',
  'stopResponse',
  'clearAlternativeActions',
  'startAlternative',
  'finishAlternative',
  'stopAlternative',
  'bridgeReady',
  'imageMode',
  'setImageMode',
  'submitImage',
  'setImageModel',
  'startImage',
  'imageProgress',
  'finishImage',
  'stoppingImage',
  'cancelImage',
  'failImage',
].join('|');

export default [
  eslint.configs.recommended,
  {
    plugins: { sonarjs },
    rules: {
      complexity: ['error', 8],
      'sonarjs/cognitive-complexity': ['error', 15],
      'max-depth': ['error', 2],
      'max-lines-per-function': ['error', { max: 20, skipBlankLines: true }],
      'max-statements': ['error', 15],
    },
  },
  {
    files: ['src/main/resources/*.js'],
    languageOptions: {
      globals: {
        document: 'readonly',
        window: 'readonly',
        chatBridge: 'readonly',
        imageBridge: 'readonly',
        clearTimeout: 'readonly',
        setTimeout: 'readonly',
      },
    },
    rules: {
      'no-unused-vars': ['error', { varsIgnorePattern: `^(${bridgeCallbacks})$` }],
    },
  },
  {
    files: ['src/main/resources/chat.js'],
    languageOptions: {
      globals: { setImageMode: 'readonly', imageMode: 'readonly', submitImage: 'readonly' },
    },
  },
  {
    files: ['src/main/resources/images.js'],
    languageOptions: {
      globals: Object.fromEntries(
        [
          'messages',
          'app',
          'promptBox',
          'element',
          'icon',
          'setModel',
          'updateSend',
          'closeModelMenu',
          'toast',
          'openSettings',
          'startResponse',
          'setGenerating',
          'setStatus',
          'followLatest',
          'showFailure',
          'showAgentMenu',
        ].map((name) => [name, 'readonly']),
      ),
    },
  },
];
