// Mock for react-markdown (ESM-only module)
const React = require('react');

function ReactMarkdown({ children }) {
  return React.createElement('div', { 'data-testid': 'markdown' }, children);
}

module.exports = ReactMarkdown;
module.exports.default = ReactMarkdown;
