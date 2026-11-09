// Site plugins for Docsify 5. Load this file after the window.$docsify config and before docsify.min.js.
// A config that loads later replaces the plugin list. Docsify reads the list when it starts.
(function () {
  // The tools page stores each contact email as the reversed address in Base64, in the data-m attribute of an a.mail link.
  // This hides the addresses from bots that read the raw markdown. Decode them into mailto links after each page renders.
  function decodeEmails(hook) {
    hook.doneEach(function () {
      document.querySelectorAll('a.mail[data-m]').forEach(function (a) {
        var m = atob(a.dataset.m).split('').reverse().join('');
        a.href = 'mailto:' + m;
        a.textContent = m;
      });
    });
  }

  // Reading progress bar: shows how far the reader has scrolled. The styles are in assets/css/site.css.
  // This bar replaces the docsify-progress plugin, which rebuilt the page body and broke the sidebar toggle in Docsify 5.
  function readingProgress(hook) {
    var bar;
    function update() {
      var max = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.width = (max > 0 ? Math.min(100, (window.scrollY / max) * 100) : 0) + '%';
    }
    hook.mounted(function () {
      bar = document.createElement('div');
      bar.className = 'reading-progress';
      document.body.appendChild(bar);
      window.addEventListener('scroll', update, { passive: true });
      window.addEventListener('resize', update);
    });
    hook.doneEach(function () {
      update();
    });
  }

  window.$docsify = window.$docsify || {};
  window.$docsify.plugins = [decodeEmails, readingProgress].concat(window.$docsify.plugins || []);
})();
