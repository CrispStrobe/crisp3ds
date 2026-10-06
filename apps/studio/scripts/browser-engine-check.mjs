// Drives "This browser" (the engine in a worker of the page, WebAssembly + WebGPU) through
// the app's own screens in headless Chromium with a real GPU, and checks what comes out.
//
//   npx playwright install chromium          # the full browser: the headless shell has no WebGPU
//   npm run build && npm run preview &
//   STUDIO_INPUTS=/path/to/inputs node scripts/browser-engine-check.mjs
//
// Environment:
//   STUDIO_URL       where the built app is served (default http://127.0.0.1:1431/)
//   STUDIO_INPUTS    an inputs folder (cameras.json, photos, masks); it is picked through the form
//   STUDIO_SETTINGS  JSON of setting name -> text typed into the form (default: none, the defaults)
//   STUDIO_CANCEL=1  cancel the run once it is under way instead of waiting for it
//   STUDIO_OUT       folder for screenshots (optional)
//   STUDIO_TIMEOUT   seconds to wait for the run (default 1800)

import { statSync } from "node:fs";
import { resolve } from "node:path";
import { chromium } from "playwright";

const base = process.env.STUDIO_URL ?? "http://127.0.0.1:1431/";
const inputs = process.env.STUDIO_INPUTS;
const settings = JSON.parse(process.env.STUDIO_SETTINGS ?? "{}");
const out = process.env.STUDIO_OUT ?? "";
const timeout = Number(process.env.STUDIO_TIMEOUT ?? 1800) * 1000;
if (!inputs) {
  console.error("STUDIO_INPUTS is needed: an inputs folder.");
  process.exit(2);
}

const launch = ["--enable-unsafe-webgpu", "--enable-features=WebGPU"];
if (process.platform === "darwin") launch.push("--use-angle=metal");
else launch.push("--enable-features=Vulkan", "--use-angle=vulkan");
const browser = await chromium.launch({ channel: "chromium", headless: true, args: launch });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
const page = await context.newPage();
const problems = [];
// Fetching a revoked object URL logs an error by design; that check switches this off.
let listening = true;
page.on("pageerror", (error) => problems.push(`page error: ${error.message}`));
page.on("console", (message) => {
  if (listening && message.type() === "error") problems.push(`console: ${message.text()}`);
});
page.on("request", (request) => {
  const url = new URL(request.url());
  if (!["127.0.0.1", "localhost"].includes(url.hostname) && !["data:", "blob:"].includes(url.protocol)) problems.push(`external request: ${request.url()}`);
});
const shot = async (name) => {
  if (out !== "") await page.screenshot({ path: resolve(out, name), type: "jpeg", quality: 68 });
};

try {
  await page.goto(base + "#/");
  const card = page.locator("section.card", { has: page.getByRole("heading", { name: "This browser" }) });
  await card.waitFor();
  console.log("connection card:", (await card.innerText()).split("\n").slice(-1)[0]);
  await card.getByRole("link", { name: /Use this browser|Open runs/ }).click();
  await page.getByRole("heading", { name: "Runs", level: 1 }).waitFor();
  console.log("runs screen:", (await page.locator(".page-head .sub").innerText()).trim());
  await page.getByRole("link", { name: /New run|Start the first run/ }).first().click();
  await page.locator("#s-sizes").waitFor({ timeout: 60_000 });

  // The folder goes in through the form's own folder input, as a person's choice would.
  await page.locator("#f-inputs-files").setInputFiles(inputs);
  await page.locator(".picked").waitFor({ timeout: 120_000 });
  console.log("picked:", await page.locator(".picked").innerText());
  console.log("intermediate surfaces:", (await page.locator("#f-previews").isChecked()) ? "on" : "off", "|", await page.locator("#f-previews-help").innerText());
  await page.locator("details.settings-group").evaluateAll((groups) => groups.forEach((group) => group.setAttribute("open", "")));
  for (const [name, value] of Object.entries(settings)) await page.locator(`#s-${name}`).fill(value);
  await shot("browser-new-run.jpg");
  const started = Date.now();
  await page.getByRole("button", { name: "Start run" }).click();
  await page.waitForURL(/#\/engine\/run\//, { timeout: 60_000 });
  console.log("run:", decodeURIComponent(page.url().split("/").pop()));

  if (process.env.STUDIO_CANCEL === "1") {
    await page.locator(".stage-running").first().waitFor({ timeout: 120_000 });
    await page.getByRole("button", { name: "Cancel run" }).click();
    await page.locator(".badge.status-cancelled").waitFor({ timeout: 300_000 });
    console.log(`cancelled after ${((Date.now() - started) / 1000).toFixed(1)} s:`, (await page.locator(".notice").first().innerText()).split("\n")[0]);
    await shot("browser-cancelled.jpg");
  } else {
    let last = "";
    const watch = setInterval(async () => {
      const text = await page.locator(".stage-running").first().innerText().catch(() => "");
      const line = text.replace(/\s+/g, " ").trim();
      if (line !== "" && line.split(" ").slice(0, 2).join(" ") !== last) {
        last = line.split(" ").slice(0, 2).join(" ");
        console.log(`  ${((Date.now() - started) / 1000).toFixed(0)} s: ${line}`);
      }
    }, 2000);
    await page.locator(".run-meta .badge.status-complete, .run-meta .badge.status-failed").waitFor({ timeout });
    clearInterval(watch);
    const seconds = (Date.now() - started) / 1000;
    const status = await page.locator(".run-meta .badge").first().innerText();
    console.log(`run ${status} after ${seconds.toFixed(1)} s | ${await page.locator(".run-meta").innerText().then((text) => text.replace(/\s+/g, " "))}`);
    if (status !== "Complete") throw new Error("the run did not complete: " + (await page.locator(".notice.bad").first().innerText().catch(() => "")));
    const load = page.getByRole("button", { name: /^Load final/ });
    if ((await load.count()) > 0) {
      console.log("final mesh waits for its button:", await load.innerText());
      await load.click();
    }
    await page.locator("#mesh-heading + .sub", { hasText: "Final surface" }).waitFor({ timeout: 300_000 });
    console.log("surface:", await page.locator("#mesh-heading + .sub").innerText());
    // Sheets load when they scroll into view; bring each one in.
    for (const sheet of await page.locator(".sheet").all()) {
      await sheet.scrollIntoViewIfNeeded();
      await sheet.locator("img").evaluate((image) => image.complete && image.naturalWidth > 0 || new Promise((done) => image.addEventListener("load", done, { once: true })));
    }
    await page.evaluate(() => scrollTo(0, 0));
    console.log(`sheets: ${await page.locator(".sheet").count()} (all loaded from memory) | reports: ${(await page.locator(".report h3").allInnerTexts()).join(", ")}`);
    await shot("browser-done.jpg");

    // The STL through the Download button: its size gives the triangle count.
    const waiting = page.waitForEvent("download");
    await page.getByRole("button", { name: /^Download STL/ }).click();
    const download = await waiting;
    const file = await download.path();
    const bytes = statSync(file).size;
    console.log(`downloaded ${download.suggestedFilename()}: ${bytes} bytes = ${(bytes - 84) / 50} triangles`);
    if ((bytes - 84) % 50 !== 0) problems.push("the downloaded STL has an odd size");

    // A sheet through the viewer's Download button.
    await page.locator(".sheet-button").first().click();
    await page.locator(".lightbox-stage img:not(.loading)").waitFor();
    const sheet = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download", exact: true }).click();
    const saved = await sheet;
    console.log(`downloaded ${saved.suggestedFilename()}: ${statSync(await saved.path()).size} bytes`);
    await page.keyboard.press("Escape");

    // Leaving the run view revokes its object URLs: the sheets' URLs must be dead afterwards.
    const urls = await page.locator(".sheet img").evaluateAll((images) => images.map((image) => image.src));
    await page.goto(base + "#/engine");
    await page.locator(".run-row").first().waitFor();
    listening = false;
    const alive = await page.evaluate(async (list) => {
      let count = 0;
      for (const url of list) if (await fetch(url).then(() => true, () => false)) count += 1;
      return count;
    }, urls);
    console.log(`object URLs of the sheets still alive after leaving the run: ${alive} of ${urls.length}`);
    if (alive > 0) problems.push("object URLs were not revoked");
    console.log("runs list:", (await page.locator(".run-row").first().innerText()).replace(/\s+/g, " "));
  }
} catch (problem) {
  problems.push(String(problem?.message ?? problem).split("\n")[0]);
  await shot("browser-problem.jpg");
} finally {
  await browser.close();
}
if (problems.length > 0) {
  console.error("Problems:\n  " + [...new Set(problems)].join("\n  "));
  process.exitCode = 1;
}
