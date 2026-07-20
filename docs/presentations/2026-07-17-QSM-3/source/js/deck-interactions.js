/**
 * BPS First Dev Meeting — interaction layer (click / scroll only, no autoplay).
 */
(() => {
  'use strict';

  const GH = 'https://github.com/BioPAL/BPS';
  const TEMPLATE_URLS = {
    '01': `${GH}/issues/new?template=01_bug_report.yml`,
    '02': `${GH}/issues/new?template=02_feature_request.yml`,
    '03': `${GH}/issues/new?template=03_algorithm_proposal.yml`,
    '04': `${GH}/issues/new?template=04_documentation_issue.yml`,
    '05': `${GH}/issues/new?template=05_security_report.yml`,
  };

  const DISCUSSION_URLS = {
    announcements: `${GH}/discussions/new?category=announcements`,
    'q-a': `${GH}/discussions/new?category=q-a`,
    ideas: `${GH}/discussions/new?category=ideas`,
    scientific: `${GH}/discussions/new?category=scientific-discussions`,
    governance: `${GH}/discussions/new?category=governance`,
    'show-and-tell': `${GH}/discussions/new?category=show-and-tell`,
    general: `${GH}/discussions/new?category=general`,
  };

  const CICD_ZONE_RATIOS = [0, 0.38, 0.72];

  function openUrl(url) {
    if (url) window.open(url, '_blank', 'noopener,noreferrer');
  }

  function stopNav(e) {
    e.stopPropagation();
  }

  function clearChildren(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  /** Browser mocks: scrollable live-site iframe + clickable chrome. */
  function initBrowserMocks(root) {
    root.querySelectorAll('[data-browser-url]').forEach((section) => {
      const url = section.getAttribute('data-browser-url');
      if (!url) return;

      section.querySelectorAll('[data-browser-frame]').forEach((frame) => {
        const chrome = frame.querySelector('[data-browser-chrome]');
        const viewport = frame.querySelector('[data-browser-viewport]');
        if (!viewport || viewport.dataset.deckReady) return;

        viewport.dataset.deckReady = '1';
        viewport.classList.add('deck-browser-viewport');
        if (!viewport.style.height) viewport.style.height = '640px';

        clearChildren(viewport);
        const iframe = document.createElement('iframe');
        iframe.src = url;
        iframe.title = url;
        iframe.loading = 'lazy';
        iframe.setAttribute('sandbox', 'allow-scripts allow-same-origin allow-popups');
        viewport.appendChild(iframe);

        if (chrome) {
          chrome.classList.add('deck-browser-chrome');
          chrome.querySelectorAll('a[href]').forEach((a) => {
            a.addEventListener('click', stopNav);
          });
          chrome.querySelectorAll('[data-open-url]').forEach((el) => {
            el.addEventListener('click', (e) => {
              stopNav(e);
              openUrl(el.getAttribute('data-open-url') || url);
            });
          });
        }

        viewport.addEventListener('wheel', stopNav, { passive: true });
        viewport.addEventListener('click', stopNav);
      });
    });
  }

  /** Journey hexagons — click advances active step. */
  function initJourney(section) {
    if (section.dataset.journeyReady) return;
    const steps = [...section.querySelectorAll('[data-journey-step]')];
    if (!steps.length) return;
    section.dataset.journeyReady = '1';

    let active = 0;
    const arrows = [...section.querySelectorAll('[data-journey-arrow]')];

    function paint() {
      steps.forEach((step, i) => {
        step.classList.toggle('deck-active', i === active);
        step.classList.toggle('deck-dim', i !== active);
      });
      arrows.forEach((arrow, i) => {
        arrow.classList.toggle('deck-lit', i < active);
      });
    }

    steps.forEach((step, i) => {
      step.addEventListener('click', (e) => {
        stopNav(e);
        active = i;
        paint();
      });
    });

    paint();
  }

  /** Flow slides — click path, then click steps in sequence. */
  function initFlow(section) {
    if (section.dataset.flowReady) return;
    const paths = [...section.querySelectorAll('[data-flow-path]')];
    const steps = [...section.querySelectorAll('[data-flow-step]')];
    if (!paths.length && !steps.length) return;
    section.dataset.flowReady = '1';

    let pathIdx = 0;
    let stepIdx = 0;

    function paint() {
      paths.forEach((p, i) => {
        p.classList.toggle('deck-active', i === pathIdx);
        p.classList.toggle('deck-dim', paths.length > 1 && i !== pathIdx);
      });
      steps.forEach((s, i) => {
        const lit = i <= stepIdx;
        s.classList.toggle('deck-active', lit);
        s.classList.toggle('deck-dim', !lit);
      });
    }

    paths.forEach((p, i) => {
      p.addEventListener('click', (e) => {
        stopNav(e);
        pathIdx = i;
        stepIdx = 0;
        paint();
      });
    });

    steps.forEach((s, i) => {
      s.addEventListener('click', (e) => {
        stopNav(e);
        stepIdx = i;
        paint();
      });
    });

    paint();
  }

  /** Template rows — click opens GitHub template URL. */
  function initTemplates(section) {
    if (section.dataset.templatesReady) return;
    section.dataset.templatesReady = '1';

    section.querySelectorAll('[data-template-row]').forEach((row) => {
      const id = row.getAttribute('data-template-id');
      const url = TEMPLATE_URLS[id];
      if (!url) return;

      row.addEventListener('click', (e) => {
        stopNav(e);
        section.querySelectorAll('[data-template-row]').forEach((r) => r.classList.remove('deck-active'));
        row.classList.add('deck-active');
        openUrl(url);
      });
    });
  }

  /** Discussion categories — click opens GitHub Discussions new-post URL. */
  function initDiscussions(section) {
    if (section.dataset.discussionsReady) return;
    section.dataset.discussionsReady = '1';

    section.querySelectorAll('[data-discussion-row]').forEach((row) => {
      const id = row.getAttribute('data-discussion-id');
      const url = DISCUSSION_URLS[id];
      if (!url) return;

      row.addEventListener('click', (e) => {
        stopNav(e);
        section.querySelectorAll('[data-discussion-row]').forEach((r) => r.classList.remove('deck-active'));
        row.classList.add('deck-active');
        openUrl(url);
      });
    });
  }

  /** CI/CD tour — scroll the diagram; zone clicks jump to each section. */
  function initCicdTour(section) {
    const zones = [...section.querySelectorAll('[data-cicd-zone]')];
    const viewport = section.querySelector('[data-cicd-viewport]');
    const panImg = viewport && viewport.querySelector('[data-cicd-pan]');
    const hint = section.querySelector('[data-cicd-hint]');
    if (!zones.length || !viewport || !panImg) return;

    function maxScroll() {
      return Math.max(0, viewport.scrollHeight - viewport.clientHeight);
    }

    function paintZones(active) {
      zones.forEach((z, i) => {
        z.classList.toggle('deck-active', i === active);
        z.classList.toggle('deck-dim', i !== active);
      });
      if (hint) {
        hint.textContent = active < zones.length - 1
          ? 'Scroll or click zones to explore →'
          : 'End of tour — scroll or click zones to revisit';
      }
    }

    function activeFromScroll() {
      const max = maxScroll();
      if (max <= 0) return 0;
      const ratio = viewport.scrollTop / max;
      let active = 0;
      for (let i = CICD_ZONE_RATIOS.length - 1; i >= 0; i--) {
        if (ratio >= CICD_ZONE_RATIOS[i] - 0.06) {
          active = i;
          break;
        }
      }
      return active;
    }

    function onScroll() {
      paintZones(activeFromScroll());
    }

    function scrollToZone(i) {
      const max = maxScroll();
      const ratio = CICD_ZONE_RATIOS[i] ?? 0;
      viewport.scrollTo({ top: max * ratio, behavior: 'smooth' });
      paintZones(i);
    }

    if (!section.dataset.cicdReady) {
      section.dataset.cicdReady = '1';
      viewport.classList.add('deck-cicd-viewport');
      viewport.addEventListener('wheel', stopNav, { passive: true });
      viewport.addEventListener('scroll', onScroll, { passive: true });
      zones.forEach((z, i) => {
        z.addEventListener('click', (e) => {
          stopNav(e);
          scrollToZone(i);
        });
      });
      if (hint) hint.classList.add('deck-cicd-hint');
    }

    onScroll();
  }

  const STAGGER_ANIMS = {
    in: 'journey-in .6s ease forwards',
    arrow: 'journey-arrow .5s ease forwards',
    specs: 'journey-specs .5s ease forwards',
    numeral: 'chapter-numeral-in .75s cubic-bezier(.2,.8,.2,1) forwards',
  };

  function applyStaggerItem(el, delaySec) {
    const variant = el.dataset.staggerVariant || 'in';
    const anim = STAGGER_ANIMS[variant] || STAGGER_ANIMS.in;
    el.style.opacity = '0';
    el.style.animation = anim;
    el.style.animationDelay = `${delaySec}s`;
  }

  /** Staggered entrance — data-stagger-item with optional data-stagger-order / data-stagger-delay. */
  function initStagger(section) {
    if (!section.hasAttribute('data-deck-stagger')) return;
    if (section.dataset.staggerReady) return;
    section.dataset.staggerReady = '1';

    const base = parseFloat(section.dataset.staggerBase || '0.15');
    const step = parseFloat(section.dataset.staggerStep || '0.11');
    const indexed = [...section.querySelectorAll('[data-stagger-item]')].map((el, i) => ({ el, i }));
    indexed.sort((a, b) => {
      const oa = a.el.dataset.staggerOrder != null ? parseFloat(a.el.dataset.staggerOrder) : null;
      const ob = b.el.dataset.staggerOrder != null ? parseFloat(b.el.dataset.staggerOrder) : null;
      if (oa != null && ob != null && oa !== ob) return oa - ob;
      if (oa != null && ob == null) return -1;
      if (oa == null && ob != null) return 1;
      return a.i - b.i;
    });

    let seq = 0;
    indexed.forEach(({ el }) => {
      const delay = el.dataset.staggerDelay != null
        ? parseFloat(el.dataset.staggerDelay)
        : base + seq * step;
      if (el.dataset.staggerDelay == null) seq += 1;
      applyStaggerItem(el, delay);
    });
  }

  function replayCssAnimations(section) {
    section.querySelectorAll('[style*="animation:"]').forEach((el) => {
      const anim = el.style.animation;
      if (!anim || anim === 'none') return;
      el.style.animation = 'none';
      void el.offsetHeight;
      el.style.animation = anim;
    });
  }

  /** Scroll-sync — scrolling the screenshot highlights matching form fields. */
  function initScrollSync(section) {
    section.querySelectorAll('[data-scroll-sync]').forEach((scroller) => {
      const name = scroller.getAttribute('data-scroll-sync');
      const container = section.querySelector(`[data-sync-items="${name}"]`);
      if (!container) return;
      scroller.classList.add('deck-scroll-sync');

      const items = [...container.children];
      const color = container.getAttribute('data-sync-color') || '#FF7E79';

      function paint() {
        const maxScroll = scroller.scrollHeight - scroller.clientHeight;
        const ratio = maxScroll > 0 ? scroller.scrollTop / maxScroll : 0;
        const activeIdx = Math.min(Math.floor(ratio * items.length), items.length - 1);
        items.forEach((el, i) => {
          if (i === activeIdx) {
            el.style.background = color;
            el.style.border = 'none';
            el.style.transform = 'scale(1.03)';
            el.style.boxShadow = '0 2px 12px rgba(0,0,0,.3)';
          } else {
            el.style.background = 'rgba(255,255,255,.08)';
            el.style.border = '1px solid rgba(255,255,255,.15)';
            el.style.transform = 'scale(1)';
            el.style.boxShadow = 'none';
          }
        });
      }

      if (!scroller.dataset.scrollSyncReady) {
        scroller.dataset.scrollSyncReady = '1';
        scroller.addEventListener('scroll', paint, { passive: true });
        scroller.addEventListener('wheel', stopNav, { passive: true });
        scroller.addEventListener('click', stopNav);
      }
      paint();
    });
  }

  /** Keep tooltip only on the first marked occurrence of each acronym in deck order. */
  function initAcronyms(deck) {
    const seen = new Set();
    deck.querySelectorAll('section.slide').forEach((section) => {
      section.querySelectorAll('.bps-acronym[data-def]').forEach((el) => {
        const key = (el.dataset.acronym || el.textContent.trim()).toUpperCase();
        if (seen.has(key)) {
          el.classList.remove('bps-acronym');
          el.removeAttribute('data-def');
          el.removeAttribute('data-acronym');
          el.removeAttribute('tabindex');
        } else {
          seen.add(key);
          el.setAttribute('tabindex', '0');
        }
      });
    });
  }

  function initSection(section) {
    if (section.hasAttribute('data-deck-chapter')) replayCssAnimations(section);
    if (section.hasAttribute('data-browser-url')) initBrowserMocks(section);
    if (section.hasAttribute('data-deck-journey')) initJourney(section);
    if (section.hasAttribute('data-deck-flow')) initFlow(section);
    if (section.hasAttribute('data-deck-templates')) initTemplates(section);
    if (section.hasAttribute('data-deck-discussions')) initDiscussions(section);
    if (section.hasAttribute('data-deck-cicd-tour')) initCicdTour(section);
    if (section.querySelector('[data-scroll-sync]')) initScrollSync(section);
    initStagger(section);
    replayCssAnimations(section);
  }

  /** Ribbon progress — content slides only (excludes title + chapter dividers). */
  function isContentSlide(section) {
    return section.matches('section.slide')
      && section.querySelector('.slide-ribbon')
      && !section.classList.contains('slide--chapter');
  }

  function ensureRibbonProgress(section) {
    const accent = section.querySelector('.slide-ribbon__accent');
    if (!accent) return null;
    let fill = accent.querySelector('.slide-ribbon__progress');
    if (!fill) {
      fill = document.createElement('div');
      fill.className = 'slide-ribbon__progress';
      fill.setAttribute('aria-hidden', 'true');
      accent.appendChild(fill);
    }
    return fill;
  }

  function initDeckProgress(deck) {
    const contentSlides = [...deck.querySelectorAll('section.slide')].filter(isContentSlide);
    const total = contentSlides.length;
    contentSlides.forEach((slide, i) => {
      const fill = ensureRibbonProgress(slide);
      if (!fill || total === 0) return;
      const pct = ((i + 1) / total) * 100;
      slide.dataset.deckProgressPct = String(pct);
      fill.style.width = '0%';
    });
  }

  function updateDeckProgress(detail) {
    const slide = detail && detail.slide;
    if (!slide || !isContentSlide(slide)) return;
    const fill = ensureRibbonProgress(slide);
    if (!fill) return;

    const target = parseFloat(slide.dataset.deckProgressPct || '0');
    let from = 0;
    const prev = detail.previousSlide;
    if (prev && isContentSlide(prev)) {
      from = parseFloat(prev.dataset.deckProgressPct || '0');
    }

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {
      fill.style.width = `${target}%`;
      return;
    }

    fill.style.transition = 'none';
    fill.style.width = `${from}%`;
    void fill.offsetWidth;
    fill.style.transition = '';
    fill.style.width = `${target}%`;
  }

  function initAll(deck) {
    deck.querySelectorAll('section').forEach(initSection);
  }

  function bindDeck(deck) {
    initDeckProgress(deck);
    initAcronyms(deck);
    initAll(deck);
    deck.addEventListener('slidechange', (e) => {
      if (e.detail) updateDeckProgress(e.detail);
      const slide = e.detail && e.detail.slide;
      if (slide) initSection(slide);
    });
  }

  function boot() {
    const deck = document.querySelector('deck-stage');
    if (deck) {
      bindDeck(deck);
      return;
    }
    customElements.whenDefined('deck-stage').then(() => {
      const el = document.querySelector('deck-stage');
      if (el) bindDeck(el);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
