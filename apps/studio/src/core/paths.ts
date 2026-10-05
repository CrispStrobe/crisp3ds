/** Paths relative to the engine's data directory, as typed into a field. Forward slashes, no leading slash. */

/** `a//b/` -> `a/b`; back-slashes become slashes; leading slashes go. */
export function cleanPath(text: string): string {
  return text
    .trim()
    .replace(/\\/g, "/")
    .split("/")
    .filter((part) => part !== "" && part !== ".")
    .join("/");
}

/** What is typed so far, split into the folder to list and the beginning of a name in it. */
export function splitTyped(text: string): { parent: string; leaf: string } {
  const normal = text.trim().replace(/\\/g, "/").replace(/^\/+/, "");
  const at = normal.lastIndexOf("/");
  if (at < 0) return { parent: "", leaf: normal };
  return { parent: cleanPath(normal.slice(0, at)), leaf: normal.slice(at + 1) };
}

export function joinPath(parent: string, name: string): string {
  return parent === "" ? name : `${parent}/${name}`;
}

export function parentPath(path: string): string {
  const at = path.lastIndexOf("/");
  return at < 0 ? "" : path.slice(0, at);
}

/** `a/b/c` -> [["a","a"],["b","a/b"],["c","a/b/c"]] for a breadcrumb. */
export function crumbs(path: string): { name: string; path: string }[] {
  const parts = cleanPath(path).split("/").filter((part) => part !== "");
  return parts.map((name, index) => ({ name, path: parts.slice(0, index + 1).join("/") }));
}

/** A path that tries to climb out is never sent to the engine for listing. */
export function listable(path: string): boolean {
  return !path.split("/").includes("..");
}
