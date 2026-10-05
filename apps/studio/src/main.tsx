import { render } from "preact";
import { App } from "./ui/app";
import { applyTheme, loadPrefs } from "./ui/prefs";
import "./styles.css";

applyTheme(loadPrefs().theme);
render(<App />, document.getElementById("app")!);
