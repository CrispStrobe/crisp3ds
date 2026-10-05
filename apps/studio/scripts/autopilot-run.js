// Runs INSIDE the native app's web view (debug builds only):
//
//   CRISP3DS_STUDIO_AUTOPILOT=scripts/autopilot-run.js <the debug binary>
//
// It waits for the shell's own engine, starts a small run through the New run form
// (finding the inputs with the folder browser), waits for it to finish, checks that
// surfaces and sheets were shown and that the Content Security Policy blocked nothing,
// and reports each step on the app's stderr as "[autopilot] ...". Whoever started the
// app can then take a screenshot; the script quits the app after AUTOPILOT_LINGER ms.
//
// The scene to use is the first folder marked "Inputs" found by walking the data folder.

(async () => {
  const internals = window.__TAURI_INTERNALS__;
  const log = (line) => internals.invoke("autopilot_log", { line: String(line) });
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const violations = [];
  document.addEventListener("securitypolicyviolation", (event) => {
    violations.push(`${event.violatedDirective}: ${event.blockedURI}`);
  });
  window.addEventListener("error", (event) => log(`page error: ${event.message}`));

  async function until(what, test, timeout = 60000) {
    const end = Date.now() + timeout;
    for (;;) {
      const value = test();
      if (value) return value;
      if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
      await sleep(100);
    }
  }
  const one = (selector, text) =>
    [...document.querySelectorAll(selector)].find((element) => text === undefined || element.textContent.includes(text));
  function type(selector, value) {
    const input = document.querySelector(selector);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  const SETTINGS = {
    sizes: "64, 128", grid: "96", planes: "48", neighbours: "4", best_of: "2", vote_neighbours: "4",
    min_votes: "2, 2", crop_padding: "6", hull_dilate: "1", windows: "5, 7", aggregates: "1, 1",
  };

  try {
    await log(`origin ${location.origin}, hash ${location.hash || "(none)"}`);
    await until("the runs screen of the local engine", () => location.hash.startsWith("#/engine") && one("h1", "Runs"), 90000);
    await log(`runs screen: ${one(".page-head .sub").textContent.trim()}`);

    location.hash = "#/engine/new";
    await until("the settings form", () => document.querySelector("#s-sizes"));
    one(".path-row button", "Browse").click();
    // Walk down until a folder marked as inputs shows up.
    for (let depth = 0; depth < 4; depth++) {
      await until("a folder listing", () => document.querySelector("#f-inputs-browser .browser-list"));
      const use = document.querySelector("#f-inputs-browser li.pickable .button");
      if (use) {
        use.click();
        await sleep(300);
        break;
      }
      const folder = document.querySelector("#f-inputs-browser .browser-entry");
      if (!folder) throw new Error("the data folder has no inputs folder in it");
      folder.click();
      await sleep(400);
    }
    const inputs = document.querySelector("#f-inputs").value;
    await log(`inputs chosen with the folder browser: ${inputs}`);
    if (!inputs) throw new Error("no inputs were chosen");
    for (const [name, value] of Object.entries(SETTINGS)) type(`#s-${name}`, value);
    type("#f-name", "native app check");
    const device = document.querySelector("#f-device");
    device.value = "cpu";
    device.dispatchEvent(new Event("change", { bubbles: true }));
    // Let the form re-render: the submit handler must see what was just typed.
    await sleep(500);
    const changed = one(".settings-head .sub").textContent.trim();
    await log(`form: ${changed}, device ${document.querySelector("#f-device").value}`);
    if (!changed.startsWith(String(Object.keys(SETTINGS).length))) throw new Error("the settings did not take");
    one("button[type=submit]", "Start run").click();

    await until("the run view", () => location.hash.startsWith("#/engine/run/"), 30000);
    await log(`run started: ${decodeURIComponent(location.hash.split("/").pop())}`);
    let lastSurface = "";
    await until(
      "the run to finish",
      () => {
        const surface = document.querySelector("#mesh-heading + .sub")?.textContent ?? "";
        if (surface && surface !== lastSurface) {
          lastSurface = surface;
          log(`surface on screen: ${surface}`);
        }
        const badge = document.querySelector(".run-meta .badge")?.textContent;
        return badge && badge !== "Running" && badge !== "Waiting";
      },
      600000,
    );
    const status = document.querySelector(".run-meta .badge").textContent;
    await until("the final surface", () => (document.querySelector("#mesh-heading + .sub")?.textContent ?? "").includes("Final surface"), 60000);
    await sleep(1500);
    const images = [...document.querySelectorAll(".sheet img")];
    const loaded = images.filter((image) => image.naturalWidth > 0).length;
    const reports = [...document.querySelectorAll(".report h3")].map((heading) => heading.textContent);
    const canvas = document.querySelector(".viewer canvas");
    await log(`run ${status}; stages: ${[...document.querySelectorAll(".stage")].map((s) => s.querySelector(".stage-name").textContent + "=" + s.querySelector(".stage-state").textContent.trim()).join(", ")}`);
    await log(`surface: ${document.querySelector("#mesh-heading + .sub").textContent}; canvas ${canvas.width}x${canvas.height}`);
    await log(`sheets loaded: ${loaded} of ${document.querySelectorAll(".sheet").length}; reports: ${reports.join(", ")}`);
    await log(`CSP violations: ${violations.length === 0 ? "none" : violations.join(" | ")}`);
    const ok = status === "Complete" && loaded > 0 && loaded === images.length && violations.length === 0;
    await log(ok ? "AUTOPILOT DONE ok" : "AUTOPILOT DONE with problems");
  } catch (problem) {
    await log(`CSP violations: ${violations.length === 0 ? "none" : violations.join(" | ")}`);
    await log(`AUTOPILOT FAILED: ${problem.message}`);
  }
  const linger = Number(new URLSearchParams(location.search).get("linger") ?? window.AUTOPILOT_LINGER ?? 15000);
  await sleep(linger);
  await internals.invoke("autopilot_quit");
})();
