// SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
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

  window.addEventListener("beforeprint", openAll);
  window.addEventListener("afterprint", restore);
  // Chrome/Safari print through matchMedia as well
  if (window.matchMedia) {
    var mq = window.matchMedia("print");
    var handler = function (e) {
      if (e.matches) { openAll(); } else { restore(); }
    };
    if (mq.addEventListener) { mq.addEventListener("change", handler); }
    else if (mq.addListener) { mq.addListener(handler); }
  }
})();
