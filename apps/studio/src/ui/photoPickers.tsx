import { useEffect, useRef, useState } from "preact/hooks";
import { checkCalibration, photoFiles, type BrowserEngine, type PickedPhotos } from "../sources/browserEngine";
import { describe } from "../sources/transport";

interface Props {
  engine: BrowserEngine;
  error?: string;
  /** Called with a name once something usable is chosen, or "" when it is not. */
  onPicked(name: string): void;
}

/** The photos for a run in this browser: several image files, or a folder of them. */
export function PhotosPicker({ engine, error, onPicked }: Props) {
  const files = useRef<HTMLInputElement>(null);
  const folder = useRef<HTMLInputElement>(null);
  const [picked, setPicked] = useState<PickedPhotos | null>(() => engine.pickedPhotos());
  const [previews, setPreviews] = useState(false);
  const [problem, setProblem] = useState("");

  const accept = (list: File[], name: string) => {
    const photos = photoFiles(list.map((file) => ({ name: file.name, file })));
    if (photos.length < 3) {
      setProblem(photos.length === 0 ? "No PNG or JPEG photos among the chosen files." : "At least three photos are needed: one per turntable position.");
      engine.setPhotos(null);
      setPicked(null);
      onPicked("");
      return;
    }
    const chosen = { name, files: photos };
    setProblem("");
    engine.setPhotos(chosen, previews);
    setPicked(chosen);
    onPicked(name);
  };

  const megabytes = picked === null ? 0 : picked.files.reduce((sum, [, file]) => sum + file.size, 0) / 1e6;
  return (
    <div class="field">
      <span class="field-label" id="f-photos-label">
        Photos
      </span>
      <div class="path-row">
        <button type="button" class="button" onClick={() => files.current?.click()}>
          Choose photos
        </button>
        <button type="button" class="button" onClick={() => folder.current?.click()}>
          Choose a folder
        </button>
        <input
          ref={files}
          id="f-photos-files"
          class="visually-hidden"
          type="file"
          multiple
          accept="image/png,image/jpeg"
          tabIndex={-1}
          aria-labelledby="f-photos-label"
          onChange={(event) => {
            const list = [...(event.currentTarget.files ?? [])];
            event.currentTarget.value = "";
            if (list.length > 0) accept(list, "photos");
          }}
        />
        <input
          ref={folder}
          id="f-photos-folder"
          class="visually-hidden"
          type="file"
          multiple
          tabIndex={-1}
          aria-labelledby="f-photos-label"
          {...{ webkitdirectory: true }}
          onChange={(event) => {
            const list = [...(event.currentTarget.files ?? [])];
            event.currentTarget.value = "";
            const first = list[0] as (File & { webkitRelativePath?: string }) | undefined;
            if (list.length > 0) accept(list, first?.webkitRelativePath?.split("/")[0] || "photos");
          }}
        />
      </div>
      <span class="help">
        One photo per turntable position, one turn, in capture order by name (PNG or JPEG). They are read here; nothing is uploaded.
      </span>
      {picked !== null && (
        <>
          <span class="picked" role="status">
            <strong>{picked.name}</strong>: {picked.files.length} photos, {Math.round(megabytes)} MB, from {picked.files[0]![0]} to{" "}
            {picked.files[picked.files.length - 1]![0]}.
          </span>
          <label class="check" for="f-photo-previews">
            <input
              id="f-photo-previews"
              type="checkbox"
              checked={previews}
              onChange={(event) => {
                setPreviews(event.currentTarget.checked);
                engine.setPhotos(picked, event.currentTarget.checked);
              }}
            />
            <span>Show intermediate surfaces while computing</span>
          </label>
          <span class="help">Off by default for photos: they need about a third more memory, and a browser gives the engine at most 4 GiB.</span>
        </>
      )}
      {(problem !== "" || error !== undefined) && (
        <span class="field-error" role="alert">
          {problem !== "" ? problem : error}
        </span>
      )}
    </div>
  );
}

/** The lens calibration for a run in this browser: one that comes with the app, or a file from this device. */
export function CalibrationPicker({ engine, error, onPicked }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [shipped, setShipped] = useState<{ label: string; path: string }[]>([]);
  const [current, setCurrent] = useState(() => engine.pickedCalibration()?.label ?? "");
  const [problem, setProblem] = useState("");
  useEffect(() => {
    let alive = true;
    void engine.calibrations().then((list) => alive && setShipped(list));
    return () => {
      alive = false;
    };
  }, [engine]);
  const done = (label: string) => {
    setProblem("");
    setCurrent(label);
    onPicked(label);
  };
  const fail = (reason: unknown) => {
    engine.setCalibration(null);
    setCurrent("");
    setProblem(describe(reason));
    onPicked("");
  };
  return (
    <div class="field">
      <span class="field-label" id="f-calibration-label">
        Lens calibration
      </span>
      <div class="path-row">
        <button type="button" class="button" onClick={() => input.current?.click()}>
          Choose a calibration file
        </button>
        <input
          ref={input}
          id="f-calibration-file"
          class="visually-hidden"
          type="file"
          accept=".json,application/json"
          tabIndex={-1}
          aria-labelledby="f-calibration-label"
          onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            event.currentTarget.value = "";
            if (file === undefined) return;
            file
              .text()
              .then((text) => {
                engine.setCalibration({ label: file.name, text: checkCalibration(text) });
                done(file.name);
              })
              .catch(fail);
          }}
        />
      </div>
      {shipped.length > 0 && (
        <div class="calibrations" role="group" aria-label="Calibrations that come with the app">
          <span class="help">Included:</span>
          {shipped.map((calibration) => {
            const name = calibration.path.split("/").pop() ?? calibration.path;
            return (
              <button
                key={calibration.path}
                type="button"
                class="button small"
                aria-pressed={current === name}
                onClick={() => void engine.useShippedCalibration(calibration.path).then(() => done(name), fail)}
              >
                {calibration.label}
              </button>
            );
          })}
        </div>
      )}
      <span class="help">Focal length, principal point and radial distortion of the camera's lens.</span>
      {current !== "" && (
        <span class="picked" role="status">
          Calibration: <strong>{current}</strong>
        </span>
      )}
      {(problem !== "" || error !== undefined) && (
        <span class="field-error" role="alert">
          {problem !== "" ? problem : error}
        </span>
      )}
    </div>
  );
}
