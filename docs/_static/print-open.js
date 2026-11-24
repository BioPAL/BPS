// SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS
// SPDX-License-Identifier: Apache-2.0
// Ctrl+P: open the folded menus (<details>, sphinx-design dropdowns) so that
// their content is printed, and fold them back afterwards.
(function () {
  var opened = [];

  function openAll() {
    opened = [];
    document.querySelectorAll("details:not([open])").forEach(function (d) {
      d.setAttribute("open", "");
      opened.push(d);
    });
  }

  function restore() {
    opened.forEach(function (d) {
      d.removeAttribute("open");
    });
    opened = [];
  }

  // Mermaid diagrams are rendered lazily (only when scrolled into view, and
  // not at all inside folded menus): render the pending ones ahead of the print
  // so that the page never prints their source. The mermaid module is the one
  // the page already loaded (same URL, so the same instance and layouts).
  var printing = false;
  var rendering = null;

  function mermaidUrl() {
    var scripts = document.querySelectorAll('script[type="module"]');
    for (var i = 0; i < scripts.length; i++) {
      var m = /import\s+mermaid\s+from\s+"([^"]+)"/.exec(scripts[i].textContent);
      if (m) { return m[1]; }
    }
    return null;
  }

  // not rendered yet, or rendered as an empty/failed drawing (a diagram
  // rendered while hidden or at the same time as another one)
  function pending() {
    return Array.prototype.slice.call(
      document.querySelectorAll(".mermaid")
    ).filter(function (el) {
      var svg = el.querySelector("svg");
      return el.getAttribute("data-processed") !== "true" ||
        el.hasAttribute("data-mermaid-render-failed") ||
        !svg || !svg.getAttribute("height");
    });
  }

  function pause(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  function renderPending() {
    if (rendering) { return rendering; }
    var url = mermaidUrl();
    if (!url || pending().length === 0) { return Promise.resolve(); }
    var wasOpen = opened.length > 0;
    if (!wasOpen) { openAll(); }
    rendering = import(url).then(function (mod) {
      var mermaid = mod.default;
      var chain = Promise.resolve();
      pending().forEach(function (el) {
        chain = chain.then(function () {
          var code = el.getAttribute("data-original-code");
          if (code !== null) {
            // the attribute keeps the source HTML-escaped ("--&gt;"): decode it,
            // or Mermaid reads "&gt;" and fails ("Syntax error in text")
            var box = document.createElement("textarea");
            box.innerHTML = code;
            el.textContent = box.value;
          }
          el.removeAttribute("data-processed");
          el.removeAttribute("data-mermaid-render-failed");
          return mermaid.run({ nodes: [el] }).then(function () {
            el.removeAttribute("data-mermaid-deferred");
            return pause(20);
          });
        }).catch(function (e) {
          console.warn("Mermaid pre-render failed:", e && (e.message || e.str || JSON.stringify(e)));
        });
      });
      return chain;
    }).catch(function (e) {
      console.warn("Mermaid pre-render unavailable:", e);
    }).then(function () {
      rendering = null;
      if (!wasOpen && !printing) { restore(); }
    });
    return rendering;
  }

  window.addEventListener("beforeprint", function () {
    printing = true;
    openAll();
    renderPending();
  });
  window.addEventListener("afterprint", function () {
    printing = false;
    restore();
  });
  // after the page's own rendering pass, finish the diagrams it deferred
  window.addEventListener("load", function () {
    setTimeout(renderPending, 2500);
  });
  // Chrome/Safari print through matchMedia as well
  if (window.matchMedia) {
    var mq = window.matchMedia("print");
    var handler = function (e) {
      if (e.matches) { printing = true; openAll(); renderPending(); }
      else { printing = false; restore(); }
    };
    if (mq.addEventListener) { mq.addEventListener("change", handler); }
    else if (mq.addListener) { mq.addListener(handler); }
  }
})();
