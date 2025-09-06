// MathJax v3 配置
window.MathJax = {
  loader: { load: ['[tex]/ams', '[tex]/require'] },
  tex: {
    packages: { '[+]': ['ams'] },  // 启用 amsmath + amssymb
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: '.*',
    processHtmlClass: 'arithmatex'
  }
};

// MkDocs Material 是单页应用，需要在页面切换后重新 typeset
document$.subscribe(() => {
  MathJax.startup?.output?.clearCache?.();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
