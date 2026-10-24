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
  await p.waitForSelector("#price-chart .main-svg", T);
  check("Classic model verdict still renders", (await p.textContent("#detail")).includes("Classic model verdict"));
  await p.waitForFunction(() => !document.querySelector(".skeleton"), null, T);
  check("Watchlist shows weekly moves", (await p.$$(".vol-pill")).length >= 10);
  check("No errors on desktop", p.errors.length === 0, p.errors.join(" | "));
  await p.close();

  // Phone: layout fits the screen
  const m = await page(browser, { viewport: { width: 390, height: 844 } });
  await m.waitForSelector("#regime-card .mscale", T);
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
