// Browser smoke test: checks the running app's core features end to end.
// Usage (with the app running on :8000):
//   cd scripts && npm install && npx playwright install chromium && node smoke_test.js [http://localhost:8000]
const { chromium } = require("playwright");

const URL = process.argv[2] || "http://localhost:8000";
const checks = [];
const check = (name, ok, detail = "") => checks.push({ name, ok: !!ok, detail });

async function page(browser, opts) {
  const p = await browser.newPage(opts);
  p.errors = [];
  p.on("pageerror", (e) => p.errors.push(e.message));
  p.on("console", (m) => m.type() === "error" && p.errors.push(m.text()));
  await p.addInitScript(() => { try { localStorage.clear(); localStorage.setItem("anomality.selected", JSON.stringify("NVDA")); } catch (e) {} });
  await p.goto(URL);
  return p;
}

(async () => {
  const browser = await chromium.launch();
  const T = { timeout: 240000 };

  // Desktop: anomaly report, levels modal, full report, asset card, classic model
  const p = await page(browser, { viewport: { width: 1440, height: 1000 } });
  await p.waitForSelector("#regime-hero .hero-grid", T);
  const hero = await p.textContent("#regime-hero");
  check("Anomaly report banner shows", hero.includes("This week's anomaly report"));
  check("Market mood is one of the four levels", /Calm|Normal|Choppy|Stormy/.test(hero));
  check("Expected weekly move is shown", /±\d+\.\d%/.test(hero));
  await p.click("#regime-hero .mood-help");
  check("Levels modal opens with 4 levels", await p.evaluate(() => document.querySelector("#levels-modal").open && document.querySelectorAll("#levels-modal .lg-list li").length === 4));
  await p.keyboard.press("Escape");
  check("Levels modal closes with Escape", !(await p.evaluate(() => document.querySelector("#levels-modal").open)));
  await p.click("#report-more summary");
  await p.waitForSelector("#herding-chart .main-svg", { timeout: 30000 });
  check("Full report opens with its charts", await p.$("#hero-chart .main-svg"));
  await p.waitForSelector("#regime-card .mscale", T);
  await p.waitForSelector("#regime-chart .main-svg", T);
  const card = await p.textContent("#regime-card");
  check("Asset card shows the expected move in dollars", /about ±\$[\d,.]+ on today/.test(card));
  check("Asset card lists surprise episodes", (await p.$$("#surprise-list li")).length > 0);
  await p.click('#regime-range button[data-r="2020 crash"]');
  await p.click("#cmp-original");
  check("Original-model comparison draws a third line", await p.evaluate(() => document.querySelectorAll("#regime-chart .scatterlayer .trace").length >= 4));
  // Decision layer: market band, try-it slider, backtest periods and comparison, asset card
  await p.waitForSelector("#dec-chart .main-svg", T);
  const band = await p.textContent("#decision-hero");
  check("Decision band shows today's call from the market mood", /Fully invested|Half invested/.test(band) && /Stormy/.test(band));
  check("Backtest compares the rule with buy and hold", (await p.$$("#dec-table tbody tr")).length >= 2 && /worst drop/i.test(band));
  await p.$eval("#rule-slider", (el) => { el.value = "0.95"; el.dispatchEvent(new Event("input", { bubbles: true })); });
  check("Try-it slider cuts to half when the mood is Stormy", (await p.textContent("#rule-result")).includes("Half invested"));
  await p.click('#dec-period button[data-p="2022 bear market"]');
  check("Period buttons update the backtest story", (await p.textContent("#dec-table")).includes("2022 bear market"));
  await p.waitForFunction(() => /95% range/.test(document.querySelector("#dec-sig")?.textContent || ""), null, T);
  check("Backtest headline shows the significance check", /More than luck/.test(await p.textContent("#dec-sig")));
  await p.click("#dec-robust summary");
  await p.waitForFunction(() => /QLIKE/.test(document.querySelector("#dec-robust-body")?.textContent || ""), null, T);
  const robustText = await p.textContent("#dec-robust-body");
  check("Robustness section shows costs, conventions and forecast quality", /0\.20%/.test(robustText) && /true buy-and-hold basket/.test(robustText) && /QLIKE/.test(robustText));
  // Tools bubble: the Time Machine, alerts and the AI analyst open from the round button, one tab at a time
  check("Tools stay in a closed bubble until opened", await p.evaluate(() => document.querySelector("#assist").hidden));
  await p.click("#assist-fab");
  check("The assistant asks first: a question and three choices, no tool shown yet", await p.evaluate(() => {
    const a = document.querySelector("#assist");
    return a.dataset.mode === "chat" && /What would you like to see/.test(a.textContent) && a.querySelectorAll(".assist-choices button").length === 3
      && getComputedStyle(a.querySelector(".assist-body")).display === "none";
  }));
  await p.click('#assist .assist-choices [data-tab="tm"]');
  check("Choosing a tool opens only that tool", await p.evaluate(() => document.querySelector("#assist").dataset.mode === "view"
    && getComputedStyle(document.querySelector("#alerts")).display === "none"));
  // Historical Time Machine: known-at-t and what-happened-next stay in separate panels
  await p.waitForSelector("#tm-chart .main-svg", T);
  check("Time Machine shows what was known and, apart, what happened next", (await p.$(".tm-known")) && /not known on/.test(await p.textContent(".tm-after")));
  await p.click('#tm-presets button[data-date="2020-02-19"]');
  await p.waitForFunction(() => /Feb 19, 2020/.test(document.querySelector(".tm-known")?.textContent || ""), null, T);
  check("Time Machine preset jumps to the COVID crash with a reconstructed decision", /Reconstructed/.test(await p.textContent(".tm-known")));
  await p.fill("#tm-date", "2020-03-14");
  await p.click('#tm-form button[type="submit"]');
  await p.waitForFunction(() => /not a trading day/.test(document.querySelector("#tm-body")?.textContent || ""), null, T);
  check("Time Machine explains a weekend date uses the previous close", /Mar 13, 2020/.test(await p.textContent(".tm-known")));
  // Intelligent alerts: monitors from the backend, filters, evaluation on a past day
  await p.click("#assist-back");
  await p.click('#assist .assist-choices [data-tab="al"]');
  await p.waitForSelector("#al-date", T);
  check("Alerts section lists every monitor with its status", (await p.$$(".al-checks tbody tr")).length >= 8);
  await p.click("#al-tm");
  await p.waitForFunction(() => document.querySelector("#al-date")?.value === "2020-03-13", null, T);
  check("Alerts on the Time Machine day show the COVID market stress", /Market stress: Stormy mood/.test(await p.textContent(".al-list")));
  await p.selectOption("#al-type", "data_quality");
  check("Alert filters narrow the list", !/Market stress: Stormy mood/.test(await p.textContent("#al-body")));
  await p.selectOption("#al-type", "all");
  // AI analyst: nothing is requested until a click; without a key it shows the backend's facts
  await p.click("#assist-back");
  await p.click('#assist .assist-choices [data-tab="an"]');
  check("AI analyst waits for a click", /Pick a question/.test(await p.textContent("#analyst")));
  await p.click('#analyst [data-kind="alert"]');
  await p.waitForFunction(() => /Measured facts/.test(document.querySelector("#an-out")?.textContent || ""), null, T);
  const an = await p.textContent("#an-out");
  check("AI analyst explains the chosen alert from cited backend facts", /Market stress: Stormy mood/.test(an) && /F\d+/.test(an) && /(Gemini|Deterministic)/.test(an));
  const voiceStatus = await (await p.request.get(`${URL}/api/voice/status`)).json();
  check("Listen button is off (with a reason) when voice isn't configured, on when it is",
    (await p.$eval("#an-listen", (b) => b.disabled)) === !voiceStatus.configured);

  // Voice playback and its error path, with the voice API faked in the browser (no ElevenLabs call)
  const v = await page(browser, { viewport: { width: 1280, height: 900 } });
  await v.route("**/api/voice/status", (r) => r.fulfill({ json: { configured: true, model: "test", max_chars: 4000 } }));
  let voiceMode = "ok";
  await v.route("**/api/voice", (r) => voiceMode === "ok"
    ? r.fulfill({ status: 200, contentType: "audio/mpeg", body: Buffer.from("ID3" + "\0".repeat(500)) })
    : r.fulfill({ status: 502, json: { detail: { status: "provider_error", message: "ElevenLabs returned HTTP 500." } } }));
  await v.reload(); // page() already loaded the app; reload so it reads the faked voice status
  await v.click("#assist-fab");
  await v.click('#assist .assist-choices [data-tab="an"]');
  await v.waitForSelector('#analyst [data-kind="summary"]', T);
  await v.click('#analyst [data-kind="summary"]');
  await v.waitForSelector("#an-listen:not([disabled])", T);
  await v.click("#an-listen");
  await v.waitForFunction(() => !document.querySelector("#an-audio").hidden, null, T);
  check("Listen to report loads the audio of the displayed report", /report shown above|press play/.test(await v.textContent("#an-voice-msg")));
  voiceMode = "fail";
  await v.click("#an-listen");
  await v.waitForFunction(() => /Voice unavailable/.test(document.querySelector("#an-voice-msg").textContent), null, T);
  check("A voice failure keeps the written report and says so", /written report above is complete/.test(await v.textContent("#an-voice-msg")) && /Measured facts/.test(await v.textContent("#an-out")));
  await v.close();
  await p.click("#assist-close");
  check("Closing the bubble hides the tools", await p.evaluate(() => document.querySelector("#assist").hidden));
  await p.click("#dec-prev");
  check("Previous rule can be added for comparison", (await p.$$("#dec-table tbody tr")).length === 3);
  await p.waitForSelector("#decision-card .pos.lg", T);
  check("Asset card shows the rule's position and the market mood", /Fully invested|Half invested/.test(await p.textContent("#decision-card")) && (await p.$("#decision-card .smeter")) !== null);
  await p.waitForSelector("#price-chart .main-svg", T);
  check("Classic model verdict still renders", (await p.textContent("#detail")).includes("Classic model verdict"));
  await p.waitForFunction(() => !document.querySelector(".skeleton"), null, T);
  check("Watchlist shows weekly moves", (await p.$$(".vol-pill")).length >= 10);
  check("No errors on desktop", p.errors.length === 0, p.errors.join(" | "));
  await p.close();

  // Phone: layout fits the screen
  const m = await page(browser, { viewport: { width: 390, height: 844 } });
  await m.waitForSelector("#regime-card .mscale", T);
  await m.waitForSelector("#decision-card .pos.lg", T);
  await m.waitForTimeout(1000);
  const width = await m.evaluate(() => document.documentElement.scrollWidth);
  check("Phone layout has no sideways scroll", width <= 390, `scrollWidth ${width}`);
  check("No errors on phone", m.errors.length === 0, m.errors.join(" | "));
  await m.close();
  await browser.close();

  for (const c of checks) console.log(`${c.ok ? "PASS" : "FAIL"}  ${c.name}${c.ok || !c.detail ? "" : `  (${c.detail})`}`);
  const failed = checks.filter((c) => !c.ok).length;
  console.log(`\n${checks.length - failed}/${checks.length} checks passed`);
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
