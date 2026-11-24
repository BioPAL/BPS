<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Second Developer Meeting

```{meeting-status} 2026-11-02-second-dev-meeting

```

**2 November 2026** · Open Source & Open Science · 40 slides

## Recording

Public session recording (≈ 1 h 43 min). © 2026 ESA / BioPAL · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

:::{div} bps-recording

<div class="bps-recording" markdown="0">
  <video
    id="bps-recording-2026-11-02"
    class="bps-recording__video"
    controls
    preload="metadata"
    playsinline
    poster="../media/meetings/2026-11-02-second-dev-meeting/poster.jpg"
  >
    <source src="https://www.biomass-disc.info/api/user-manager/v1/files/media/2026-11-02_second_dev_meeting.mp4" type="video/mp4">
    <track kind="chapters" src="../media/meetings/2026-11-02-second-dev-meeting/chapters.vtt" srclang="en" label="Chapters" default>
    Your browser does not support HTML5 video. You can
    <a href="https://www.biomass-disc.info/api/user-manager/v1/files/media/2026-11-02_second_dev_meeting.mp4">download the recording</a>.
  </video>
  <p class="bps-recording__meta">
    <a class="bps-recording__download" href="https://www.biomass-disc.info/api/user-manager/v1/files/media/2026-11-02_second_dev_meeting.mp4" download>
      <i class="fa-solid fa-download" aria-hidden="true"></i> Download MP4
    </a>
    <span class="bps-recording__credit">BioPAL Second Developer Meeting (2 November 2026). © ESA / BioPAL, CC BY 4.0.</span>
  </p>
</div>

:::

### Chapters

:::{div} bps-chapters

<nav class="bps-chapters" aria-label="Recording chapters" data-video="bps-recording-2026-11-02" markdown="0">
  <ol class="bps-chapters__list">
    <li><button type="button" class="bps-chapters__item" data-start="0" aria-label="Play from 00:00, Welcome and introduction">
      <span class="bps-chapters__time">00:00</span>
      <span class="bps-chapters__text"><span class="bps-chapters__title">Welcome and introduction</span><span class="bps-chapters__speaker">Klaus Scipal · ESA</span></span>
    </button></li>
    <li><button type="button" class="bps-chapters__item" data-start="139" aria-label="Play from 02:19, BioPAL GitHub overview">
      <span class="bps-chapters__time">02:19</span>
      <span class="bps-chapters__text"><span class="bps-chapters__title">BioPAL GitHub overview</span><span class="bps-chapters__speaker">Yoann Rey-Ricord · ACRI-ST</span></span>
    </button></li>
    <li><button type="button" class="bps-chapters__item" data-start="2376" aria-label="Play from 39:36, Live demo">
      <span class="bps-chapters__time">39:36</span>
      <span class="bps-chapters__text"><span class="bps-chapters__title">Live demo</span><span class="bps-chapters__speaker">Yoann Rey-Ricord · ACRI-ST</span></span>
    </button></li>
    <li><button type="button" class="bps-chapters__item" data-start="3424" aria-label="Play from 57:04, MAAP: introduction and access procedures">
      <span class="bps-chapters__time">57:04</span>
      <span class="bps-chapters__text"><span class="bps-chapters__title">MAAP: introduction and access procedures</span><span class="bps-chapters__speaker">Joseph Melizza · SERCO</span></span>
    </button></li>
    <li><button type="button" class="bps-chapters__item" data-start="4639" aria-label="Play from 1:17:19, BPS on MAAP">
      <span class="bps-chapters__time">1:17:19</span>
      <span class="bps-chapters__text"><span class="bps-chapters__title">BPS on MAAP</span><span class="bps-chapters__speaker">Clément Albinet · ESA</span></span>
    </button></li>
    <li><button type="button" class="bps-chapters__item" data-start="5724" aria-label="Play from 1:35:24, Questions and answers">
      <span class="bps-chapters__time">1:35:24</span>
      <span class="bps-chapters__text"><span class="bps-chapters__title">Questions and answers</span><span class="bps-chapters__speaker">All participants</span></span>
    </button></li>
  </ol>
</nav>
<script>
(function () {
  var nav = document.currentScript && document.currentScript.previousElementSibling;
  if (!nav || !nav.classList.contains("bps-chapters")) { return; }
  var video = document.getElementById(nav.getAttribute("data-video"));
  if (!video) { return; }
  var items = Array.prototype.slice.call(nav.querySelectorAll(".bps-chapters__item"));
  var starts = items.map(function (b) { return parseFloat(b.getAttribute("data-start")) || 0; });
  items.forEach(function (btn, i) {
    btn.addEventListener("click", function () {
      video.currentTime = starts[i];
      var p = video.play();
      if (p && p.catch) { p.catch(function () {}); }
      video.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });
  function sync() {
    var t = video.currentTime, active = -1;
    for (var i = 0; i < starts.length; i++) { if (t >= starts[i]) { active = i; } }
    if (video.paused && t === 0) { active = -1; }
    items.forEach(function (btn, i) {
      if (i === active) { btn.setAttribute("aria-current", "true"); }
      else { btn.removeAttribute("aria-current"); }
    });
  }
  video.addEventListener("timeupdate", sync);
  video.addEventListener("seeked", sync);
})();
</script>

:::

## Programme

Monday 2 November 2026, 15:00 to 17:00 CET, online. About 48 participants joined the session.

| Time (CET) | Session | Speaker | Duration |
| --- | --- | --- | --- |
| 15:00 | Welcome and introduction | Klaus Scipal (ESA) | 5 min |
| 15:05 | BioPAL GitHub overview | Yoann Rey-Ricord (ACRI-ST) | 45 min |
| 15:50 | MAAP: introduction and access procedures | Joseph Melizza (SERCO) | 15 min |
| 16:05 | BPS on MAAP | Clément Albinet (ESA) | 15 min |
| 16:20 | Questions and answers | All participants | 40 min |
| 17:00 | Closing | | |

## What is new since the first meeting

The [First Developer Meeting](developer-meeting.md) explained why ESA is opening the BioMASS Processing Suite and what was already live. This second session moves from intent to practice. It walks through the contributor path step by step: forking the repository and opening a draft pull request directly from the GitHub web interface, how branches and releases are governed (CI/CD tiers and branch protection), and how to use a GitHub personal access token on MAAP Coding. A live demo shows the full loop end to end, and guest speakers from SERCO and ESA introduce MAAP access and how BPS runs on the platform, with a pointer to the MAAP hackathon repository.

## Slides

The published deck is the BioPAL GitHub overview (40 slides, four parts: introduction and intent, what is live today, how you contribute, and how to join). The slides of the guest speakers are only visible in the recording.

:::{div} bps-deck-hero

<div class="bps-deck-preview" markdown="0">
  <iframe src="../presentations/2026-11-02-second-dev-meeting/index.html" title="Second Developer Meeting slides" loading="lazy" tabindex="-1"></iframe>
  <a class="bps-deck-preview__overlay" href="../presentations/2026-11-02-second-dev-meeting/index.html" target="_blank" rel="noopener noreferrer">
    <span class="bps-deck-preview__play"><i class="fa-solid fa-play" aria-hidden="true"></i> Open full-screen slides</span>
  </a>
</div>

:::

Keyboard in the deck: **← / →** or **Space** · **F** present mode · **R** reset

---

**Previous:** [First Developer Meeting](developer-meeting.md) | **Next:** [Conflict resolution](conflict-resolution.md)
