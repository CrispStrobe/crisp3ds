// Downloads an example object through the app's own Example objects screen in headless Chrome
// (WebGPU) and reconstructs it in the browser: what is shown before the download, the time the
// download took, the run, and the attribution kept with it.
//
//   STUDIO_URL=https://crispstrobe.github.io/crisp3ds/ STUDIO_OBJECT=Dragon node scripts/examples-check.mjs

import { chromium } from "playwright";

const base = process.env.STUDIO_URL ?? "http://127.0.0.1:1431/";
const object = process.env.STUDIO_OBJECT ?? "Dragon";
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const problems = [];
page.on("pageerror", (error) => problems.push(`page error: ${error.message}`));
page.on("console", (message) => message.type() === "error" && problems.push(`console: ${message.text()}`));
try {
  await page.goto(base + "#/");
  await page.waitForFunction(() => globalThis.crossOriginIsolated, null, { timeout: 15000 }).catch(() => undefined);
  await page.waitForLoadState("load");
  await page.getByRole("link", { name: "Choose an object" }).click();
  await page.getByRole("heading", { name: "Example objects", level: 1 }).waitFor();
  await page.locator(".example").first().waitFor({ timeout: 60000 });
  console.log("source shown:", (await page.locator(".attribution-card").first().innerText()).replace(/\s+/g, " ").slice(0, 500));
  console.log("sections:", (await page.locator(".example-source h2").allInnerTexts()).join(" | "));
  console.log("objects:", (await page.locator(".example-text strong").allInnerTexts()).join(", "));
  // The object's name, or the start of it ("Rhino" for "Rhino figurine (...)").
  const row = page.locator(".example", { has: page.locator("strong", { hasText: new RegExp(`^${object}`) }) }).first();
  console.log("row:", (await row.locator(".example-text").innerText()).replace(/\s+/g, " "));
  await row.getByRole("button", { name: /Download and reconstruct|Reconstruct/ }).first().click();
  const confirm = row.locator(".example-confirm");
  if ((await confirm.count()) > 0) {
    console.log("asked before download:", (await confirm.locator("p").innerText()).replace(/\s+/g, " "));
    await confirm.getByRole("button", { name: "Download and reconstruct" }).click();
  }
  const started = Date.now();
  await page.waitForURL(/#\/engine\/run\//, { timeout: 900000 });
  console.log(`downloaded and started after ${((Date.now() - started) / 1000).toFixed(1)} s: ${decodeURIComponent(page.url().split("/").pop())}`);
  const runStart = Date.now();
  await page.locator(".run-meta .badge.status-complete, .run-meta .badge.status-failed").waitFor({ timeout: 1800000 });
  console.log(`run ${await page.locator(".run-meta .badge").first().innerText()} after ${((Date.now() - runStart) / 1000).toFixed(1)} s | ${(await page.locator(".run-meta").innerText()).replace(/\s+/g, " ")}`);
  console.log("notice:", (await page.locator(".notice").first().innerText().catch(() => "")).replace(/\s+/g, " ").slice(0, 400));
  console.log("stages:", (await page.locator(".stage").allInnerTexts()).map((text) => text.replace(/\s+/g, " ")).join(" | "));
  const load = page.getByRole("button", { name: /^Load final/ });
  if ((await load.count()) > 0) await load.click();
  await page.locator("#mesh-heading + .sub", { hasText: "Final surface" }).waitFor({ timeout: 300000 }).catch(() => undefined);
  console.log("surface:", await page.locator("#mesh-heading + .sub").innerText());
  console.log("numbers:", (await page.locator(".metrics > div").allInnerTexts()).map((row) => row.replace(/\s+/g, " ")).filter((row) => /Registered|Input photos|Reprojection median|Silhouette iou/i.test(row)).join(" | "));
  await page.locator("details.attribution summary").click();
  console.log("kept with the run:", (await page.locator("details.attribution pre").innerText()).replace(/\s+/g, " ").slice(0, 300));
} catch (problem) {
  problems.push(String(problem?.message ?? problem).split("\n")[0]);
} finally {
  await browser.close();
}
if (problems.length > 0) {
  console.error("Problems:\n  " + [...new Set(problems)].join("\n  "));
  process.exitCode = 1;
}
