// Toàn bộ JS của dự án — không thêm thư viện, không viết XHR tay.

// 1) progress bar cho upload (HTMX phát sẵn event, không cần XHR tự viết)
document.body.addEventListener('htmx:xhr:progress', e => {
  const b = document.getElementById('bar');
  if (b && e.detail.lengthComputable) b.value = e.detail.loaded / e.detail.total * 100;
});

// 2) countdown theo GIỜ SERVER (bù lệch đồng hồ client)
function fmt(ms) {
  const s = Math.floor(ms / 1000);
  const h = String(Math.floor(s / 3600)).padStart(2, '0');
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const sec = String(s % 60).padStart(2, '0');
  return `${h}:${m}:${sec}`;
}

function startCountdowns(root) {
  root.querySelectorAll('[data-close]').forEach(el => {
    if (el.dataset.countdownBound) return;
    el.dataset.countdownBound = '1';
    const skew = new Date(el.dataset.now) - new Date();
    const close = new Date(el.dataset.close);
    setInterval(() => {
      const ms = close - (Date.now() + skew);
      el.textContent = ms <= 0 ? 'Đã hết giờ' : fmt(ms);
      if (ms > 0 && ms < 600000) el.classList.add('text-red-600', 'animate-pulse');
    }, 1000);
  });
}

document.addEventListener('DOMContentLoaded', () => startCountdowns(document));
document.body.addEventListener('htmx:afterSwap', e => startCountdowns(e.detail.target));
