/** Parses a binary STL off the main thread and hands the buffers back without copying. */

import { parseBinaryStl, type ParsedMesh } from "../core/stl";

export interface StlRequest {
  id: number;
  buffer: ArrayBuffer;
}

export type StlResponse = { id: number; ok: true; mesh: ParsedMesh } | { id: number; ok: false; error: string };

// Typed by hand: pulling in the WebWorker lib next to the DOM lib makes their globals collide.
const scope = self as unknown as {
  onmessage: ((message: MessageEvent<StlRequest>) => void) | null;
  postMessage(message: StlResponse, transfer?: Transferable[]): void;
};

scope.onmessage = (message) => {
  const { id, buffer } = message.data;
  try {
    const mesh = parseBinaryStl(buffer);
    scope.postMessage({ id, ok: true, mesh }, [mesh.positions.buffer, mesh.normals.buffer, mesh.indices.buffer]);
  } catch (problem) {
    scope.postMessage({ id, ok: false, error: problem instanceof Error ? problem.message : String(problem) });
  }
};
