// Records a scripted session of the real app (localhost:8001) for the demo film.
// Chrome's screencast at 1440x810 CSS px, 2x device pixels -> assets/rec/frames/*.jpg (variable rate),
// then a 30 fps CFR video assets/rec/session.mp4, plus assets/rec/track.js: the cursor path, clicks and
// key presses with their times on the video's clock (seconds from its first frame).
// Run from demo/: node scripts/record.cjs
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const { chromium } = require("/Users/kai/Documents/anomality/scripts/node_modules/playwright");

const URL = "http://localhost:8001";
const W = 1440, H = 810;
const OUT = path.join(__dirname, "..", "assets", "rec");
const FR = path.join(OUT, "frames");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let paused = 0;   // seconds cut out of the film by pauseRec (a slow API call), subtracted from every later time
const now = () => Date.now() / 1000 - paused;

(async () => {
  fs.rmSync(FR, { recursive: true, force: true });
  fs.mkdirSync(FR, { recursive: true });
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
  // warm the server's caches for NVDA so the recorded load is quick
  const warm = await ctx.newPage();
  await warm.goto(URL);
  await warm.waitForSelector("#decision-hero:not([hidden]) .dec-grid", { timeout: 120000 });
  await warm.fill("#search-input", "NVDA");
  await warm.press("#search-input", "Enter");
  await warm.waitForFunction(() => /NVDA/.test(document.querySelector("#regime-card")?.textContent || ""), null, { timeout: 120000 });
  await warm.close();

  const p = await ctx.newPage();
  await p.addInitScript(() => { try { localStorage.clear(); } catch (e) {} });
  await p.goto(URL);
  await p.waitForSelector("#decision-hero:not([hidden]) .dec-grid", { timeout: 120000 });
  await p.waitForSelector("#regime-card .mscale", { timeout: 120000 });
  await p.waitForSelector("#dec-chart .main-svg", { timeout: 120000 });
  await sleep(1500);

  // ---- screencast
  const cdp = await ctx.newCDPSession(p);
  const frames = [];
  let n = 0;
  cdp.on("Page.screencastFrame", async (f) => {
    const file = path.join(FR, String(n++).padStart(5, "0") + ".jpg");
    fs.writeFileSync(file, Buffer.from(f.data, "base64"));
    frames.push({ file, ts: f.metadata.timestamp - paused });
    try { await cdp.send("Page.screencastFrameAck", { sessionId: f.sessionId }); } catch (e) {}
  });
  await cdp.send("Page.startScreencast", { format: "jpeg", quality: 92, maxWidth: W * 2, maxHeight: H * 2, everyNthFrame: 1 });
  const ev = [];   // {t, x, y} moves, {t, click:true}, {t, key:"N"}, {t, mark:"name"}
  let cx = W + 60, cy = H - 120;
  const mark = (name) => ev.push({ t: now(), mark: name });
  const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  async function moveTo(x, y, ms = 800) {
    const x0 = cx, y0 = cy, steps = Math.max(2, Math.round(ms / 16));
    for (let i = 1; i <= steps; i++) {
      const u = ease(i / steps);
      cx = x0 + (x - x0) * u; cy = y0 + (y - y0) * u;
      await p.mouse.move(cx, cy);
      ev.push({ t: now(), x: cx, y: cy });
      await sleep(16);
    }
  }
  async function center(sel) {
    const b = await (await p.waitForSelector(sel, { timeout: 60000 })).boundingBox();
    return [b.x + b.width / 2, b.y + b.height / 2];
  }
  async function clickOn(sel, ms = 800) {
    const [x, y] = await center(sel);
    await moveTo(x, y, ms);
    await sleep(180);
    ev.push({ t: now(), click: true });
    await p.mouse.down(); await sleep(70); await p.mouse.up();
  }
  async function scrollTo(sel, offset = 70, ms = 1300) {
    const target = await p.$eval(sel, (e, o) => e.getBoundingClientRect().top + scrollY - o, offset);
    const y0 = await p.evaluate(() => scrollY), steps = Math.round(ms / 16);
    for (let i = 1; i <= steps; i++) {
      const y = y0 + (target - y0) * ease(i / steps);
      await p.evaluate((v) => window.scrollTo(0, v), y);
      await sleep(16);
    }
  }
  // scroll inside a scrollable element (the assistant panel) until `sel` is near its top
  async function scrollIn(box, sel, ms = 1300) {
    const [y0, target] = await p.$eval(box, (b, s) => {
      const e = b.querySelector(s) || document.querySelector(s);
      return [b.scrollTop, b.scrollTop + e.getBoundingClientRect().top - b.getBoundingClientRect().top - 40];
    }, sel);
    const steps = Math.round(ms / 16);
    for (let i = 1; i <= steps; i++) {
      await p.$eval(box, (b, v) => { b.scrollTop = v; }, y0 + (target - y0) * ease(i / steps));
      await sleep(16);
    }
  }
  // stop the screencast while `fn` runs, then cut the gap out of the film's clock
  async function pauseRec(fn) {
    await cdp.send("Page.stopScreencast");
    const a = Date.now() / 1000;
    await fn();
    paused += Date.now() / 1000 - a;
    await cdp.send("Page.startScreencast", { format: "jpeg", quality: 92, maxWidth: W * 2, maxHeight: H * 2, everyNthFrame: 1 });
    await sleep(150);
  }
  await sleep(300);
  const t0 = now();
  mark("start");
  await sleep(1400);

  // 1) the weekly report: what do the levels mean?
  mark("levels");
  await clickOn("#regime-hero .mood-help", 1000);
  await p.waitForFunction(() => document.querySelector("#levels-modal").open);
  await sleep(2800);
  await clickOn("#levels-modal-close", 700);
  await sleep(600);

  // 2) any asset: type NVDA
  mark("search");
  await clickOn("#search-input", 900);
  await sleep(250);
  for (const k of "NVDA") { ev.push({ t: now(), key: k }); await p.keyboard.type(k); await sleep(150); }
  await sleep(250);
  ev.push({ t: now(), key: "Enter" });
  await p.keyboard.press("Enter");
  await p.waitForFunction(() => /NVDA/.test(document.querySelector("#regime-card")?.textContent || "") && document.querySelector("#regime-chart .main-svg"), null, { timeout: 60000 });
  await sleep(500);
  mark("asset");
  await scrollTo("#regime-card", 70, 1400);
  await moveTo(560, 330, 900);
  await sleep(2600);

  // 3) crash-proofed: the 2020 crash, against the original research model
  mark("crash");
  await clickOn('#regime-range button[data-r="2020 crash"]', 900);
  await sleep(1300);
  await clickOn("#cmp-original", 800);
  await sleep(2800);

  // 4) the decision layer: drag the try-it slider into Stormy
  mark("rule");
  await scrollTo("#decision-hero", 70, 1500);
  await sleep(700);
  const s = await (await p.$("#rule-slider")).boundingBox();
  const [mn, mx, val] = await p.$eval("#rule-slider", (e) => [+e.min, +e.max, +e.value]);
  const at = (v) => s.x + 8 + (s.width - 16) * (v - mn) / (mx - mn);
  await moveTo(at(val), s.y + s.height / 2, 900);
  await sleep(200);
  ev.push({ t: now(), click: true });
  await p.mouse.down();
  mark("drag");
  await moveTo(at(0.95), s.y + s.height / 2, 1700);
  await p.mouse.up();
  mark("half");
  await sleep(2600);

  // 5) the backtest: the COVID crash
  mark("backtest");
  await scrollTo("#decision-hero .dec-bt", 80, 1300);
  await sleep(500);
  await clickOn('#dec-period button[data-p="COVID crash"]', 900);
  mark("covid");
  await sleep(3200);

  // 6) the assistant bubble: what would you like to see?
  mark("assist");
  await clickOn("#assist-fab", 1000);
  await sleep(1800);

  // 7) the Time Machine: opens on the eve of the COVID crash, known vs. what happened next
  mark("tm");
  await clickOn('.assist-choices button[data-tab="tm"]', 700);
  await p.waitForSelector("#tm-body .tm-grid", { timeout: 60000 });
  await sleep(300);
  await moveTo(650, 300, 900);
  await sleep(2400);
  mark("tm-after");
  await moveTo(1150, 345, 900);
  await sleep(2400);
  mark("tm-chart");
  await scrollIn(".assist-body", "#tm-chart", 1300);
  await sleep(2600);

  // 8) alerts: each one shows its measurement and threshold
  mark("alerts");
  await clickOn("#assist-back", 900);
  await sleep(700);
  await clickOn('.assist-choices button[data-tab="al"]', 800);
  await p.waitForSelector("#alerts .al-item", { timeout: 60000 });
  await sleep(300);
  await moveTo(1300, 560, 1000);
  await sleep(3200);

  // 9) the AI analyst: Gemini explains the backend's facts; every number checked
  mark("analyst");
  await clickOn("#assist-back", 900);
  await sleep(700);
  await clickOn('.assist-choices button[data-tab="an"]', 800);
  await p.waitForSelector("#analyst .an-bar", { timeout: 60000 });
  await sleep(700);
  await clickOn('.an-bar button[data-kind="summary"]', 900);
  await sleep(900);
  // Gemini takes ~15 s: pause the recording while it answers, so the film cuts straight to the result
  await pauseRec(() => p.waitForSelector("#an-out .an-src", { timeout: 120000 }).then(() => sleep(600)));
  mark("answer");
  await moveTo(640, 420, 1000);
  await sleep(4200);
  mark("end");
  const tEnd = now() + 0.4;   // the screencast only sends frames when the page changes: hold the last one to here
  await cdp.send("Page.stopScreencast");
  await sleep(300);
  await browser.close();

  // ---- frames -> 30 fps video; events -> the video's clock
  const first = frames[0].ts;
  const list = frames.map((f, i) => `file '${f.file}'\nduration ${((frames[i + 1] ? frames[i + 1].ts : Math.max(tEnd, f.ts + 0.1)) - f.ts).toFixed(4)}`).join("\n") + `\nfile '${frames[frames.length - 1].file}'\n`;
  fs.writeFileSync(path.join(OUT, "frames.txt"), list);
  execFileSync("ffmpeg", ["-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", path.join(OUT, "frames.txt"),
    "-vf", "fps=30,scale=2880:1620:flags=lanczos,format=yuv420p", "-t", (Math.max(tEnd, frames[frames.length - 1].ts + 0.1) - first).toFixed(3), "-c:v", "libx264", "-crf", "14", "-preset", "slow",
    path.join(OUT, "session.mp4")]);
  const track = { w: W, h: H, start: +(t0 - first).toFixed(3), duration: +(Math.max(tEnd, frames[frames.length - 1].ts + 0.1) - first).toFixed(3),
    events: ev.map((e) => ({ ...e, t: +(e.t - first).toFixed(3) })) };
  fs.writeFileSync(path.join(OUT, "track.js"), "window.TRACK = " + JSON.stringify(track) + ";\n");
  console.log(`${frames.length} frames, ${track.duration}s; marks:`, track.events.filter((e) => e.mark).map((e) => `${e.mark}@${e.t}`).join(" "));
})().catch((e) => { console.error(e); process.exit(1); });
