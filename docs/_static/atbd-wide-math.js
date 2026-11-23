// SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS
// SPDX-License-Identifier: Apache-2.0
//
// Converted Science Guide pages: an inline formula wider than its text column
// gets the class atbd-wide-math (custom.css lets it scroll in its own box).
// Run after MathJax has typeset the page, and again on resize.
(function () {
  function mark() {
    document.querySelectorAll(".bd-content span.math").forEach(function (span) {
      var box = span.closest("p, li, td, th, dd") || span.parentElement;
      if (!box) return;
      span.classList.remove("atbd-wide-math");
      if (span.scrollWidth > box.clientWidth + 8) span.classList.add("atbd-wide-math");
    });
  }
  function whenTypeset() {
    if (window.MathJax && MathJax.startup && MathJax.startup.promise) {
      MathJax.startup.promise.then(mark);
    } else {
      setTimeout(whenTypeset, 200);
    }
  }
  document.addEventListener("DOMContentLoaded", whenTypeset);
  window.addEventListener("resize", function () { clearTimeout(window.__atbdWm); window.__atbdWm = setTimeout(mark, 200); });
})();
