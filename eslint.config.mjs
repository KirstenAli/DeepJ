import eslint from '@eslint/js';
import sonarjs from 'eslint-plugin-sonarjs';

const bridgeCallbacks = [
  'setModel',
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
