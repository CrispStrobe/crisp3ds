import type { RunSource } from "../sources/types";

/** `mesh/mesh.stl` of run `20261006-bunny` -> `20261006-bunny-mesh.stl`: unique enough in a downloads folder. */
export function downloadName(run: string, path: string): string {
  const safe = run.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  const parts = path.split("/").filter((part) => part !== "");
  const file = parts.pop() ?? "file";
  // The folder says what it is (mesh/mesh.stl, stereo/depth-merged.png) unless the name already does.
  const folder = parts.pop();
  const name = folder !== undefined && !file.startsWith(folder) ? `${folder}-${file}` : file;
  return safe === "" ? name : `${safe}-${name}`;
}

/** Whether saving a run file through the browser makes sense: in the app the files are already on disk. */
export function canDownload(source: RunSource): boolean {
  return source.kind !== "local";
}

/**
 * Saves a file of the run through the browser's download. The bytes are fetched like any
 * other artifact (from memory for a run made in this browser) and handed over as an
 * object URL that is revoked right after.
 */
export async function downloadFile(source: RunSource, path: string): Promise<void> {
  const bytes = await source.fetchBytes(path);
  const url = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = downloadName(source.title.split("/").filter(Boolean).pop() ?? "run", path);
  document.body.append(link);
  link.click();
  link.remove();
  // The download has its own reference by now.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
