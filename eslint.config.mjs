import eslint from '@eslint/js';

const bridgeCallbacks = [
  'setModel',
  'setBusy',
  'setGenerating',
  'startResponse',
  'finishResponse',
  'failResponse',
  'stopResponse',
  'bridgeReady',
].join('|');

export default [
  eslint.configs.recommended,
  {
    rules: {
      complexity: ['error', 8],
      'max-depth': ['error', 2],
      'max-lines-per-function': ['error', { max: 20, skipBlankLines: true }],
      'max-statements': ['error', 15],
    },
  },
  {
    files: ['src/main/resources/chat.js'],
    languageOptions: {
      globals: {
        document: 'readonly',
        window: 'readonly',
        chatBridge: 'readonly',
        clearTimeout: 'readonly',
        setTimeout: 'readonly',
      },
    },
    rules: {
      'no-unused-vars': ['error', { varsIgnorePattern: `^(${bridgeCallbacks})$` }],
    },
  },
];
