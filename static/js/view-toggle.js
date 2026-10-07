(function () {
  const root = document.documentElement;
  const btn = document.getElementById('viewToggle');
  if (!btn) return;
  function apply(mode) {
    root.classList.toggle('mobile-view', mode === 'mobile');
    btn.textContent = mode === 'mobile' ? '🖥️ Desktop view' : '📱 Mobile view';
    try { localStorage.setItem('viewMode', mode); } catch (e) {}
  }
  let saved = 'desktop';
  try { saved = localStorage.getItem('viewMode') || 'desktop'; } catch (e) {}
  apply(saved);
  btn.addEventListener('click', () =>
    apply(root.classList.contains('mobile-view') ? 'desktop' : 'mobile'));
})();
