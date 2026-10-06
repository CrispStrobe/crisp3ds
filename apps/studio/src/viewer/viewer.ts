/**
 * The 3D view. One mesh on screen at a time; swapping keeps the camera because every mesh
 * of a run is in the same coordinate frame. The frame has no physical scale and no known
 * up direction, so the view frames the bounding box and lets the user say which axis is up.
 *
 * Renders on demand only (no animation loop): a phone showing a still model costs nothing.
 */

import {
  AmbientLight,
  Box3,
  BufferAttribute,
  BufferGeometry,
  Color,
  DirectionalLight,
  DoubleSide,
  Group,
  HemisphereLight,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  Quaternion,
  Scene,
  Sphere,
  Spherical,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { ParsedMesh } from "../core/stl";

export const UP_AXES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"] as const;
export type UpAxis = (typeof UP_AXES)[number];

const AXIS_VECTORS: Record<UpAxis, [number, number, number]> = {
  "+X": [1, 0, 0],
  "-X": [-1, 0, 0],
  "+Y": [0, 1, 0],
  "-Y": [0, -1, 0],
  "+Z": [0, 0, 1],
  "-Z": [0, 0, -1],
};

export interface ViewerColors {
  background: string;
  surface: string;
}

export class Viewer {
  private readonly renderer: WebGLRenderer;
  private readonly scene = new Scene();
  private readonly camera = new PerspectiveCamera(35, 1, 0.01, 100);
  private readonly controls: OrbitControls;
  /** Rotates the model so that the chosen up axis points to the top of the screen. */
  private readonly orientation = new Group();
  private readonly material = new MeshStandardMaterial({
    color: 0xb9bec6,
    roughness: 0.72,
    metalness: 0.0,
    side: DoubleSide,
    flatShading: false,
  });
  private mesh: Mesh | null = null;
  private framed = false;
  /** The person has moved the camera since the last framing. */
  private moved = false;
  private pending = 0;
  private disposed = false;
  private readonly observer: ResizeObserver;

  constructor(
    canvas: HTMLCanvasElement,
    private readonly host: HTMLElement,
  ) {
    this.renderer = new WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "default" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.scene.add(this.orientation);

    // Neutral lighting: soft sky/ground fill plus a headlight that travels with the camera,
    // so the side being looked at is always lit and shape reads from shading alone.
    this.scene.add(new HemisphereLight(0xffffff, 0x8a8f98, 1.1));
    this.scene.add(new AmbientLight(0xffffff, 0.25));
    const head = new DirectionalLight(0xffffff, 1.9);
    head.position.set(-0.6, 0.9, 1.2);
    this.camera.add(head);
    this.camera.add(head.target);
    head.target.position.set(0, 0, -1);
    this.scene.add(this.camera);

    this.camera.position.set(2.2, 1.4, 2.6);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.controls.zoomToCursor = true;
    this.controls.addEventListener("change", () => this.requestRender());
    // Once the person turns or zooms, new surfaces keep their view.
    this.controls.addEventListener("start", () => {
      this.moved = true;
    });

    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(host);
    this.resize();
  }

  /**
   * Shows `parsed` in place of whatever is shown. Each new surface is framed until the person
   * moves the camera, so what ends up on screen does not depend on which surfaces happened to
   * be shown before it (a fast replay skips some).
   */
  setMesh(parsed: ParsedMesh): void {
    this.clearMesh();
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(parsed.positions, 3));
    geometry.setAttribute("normal", new BufferAttribute(parsed.normals, 3));
    geometry.setIndex(new BufferAttribute(parsed.indices, 1));
    geometry.boundingBox = new Box3(new Vector3(...parsed.min), new Vector3(...parsed.max));
    geometry.boundingSphere = geometry.boundingBox.getBoundingSphere(new Sphere());
    this.mesh = new Mesh(geometry, this.material);
    this.orientation.add(this.mesh);
    if (!this.framed || !this.moved) {
      this.frame();
      this.framed = true;
    } else {
      this.fitClipPlanes();
    }
    this.requestRender();
  }

  /** Removes the mesh and frees its GPU buffers. The parsed arrays stay with whoever owns them. */
  clearMesh(): void {
    if (this.mesh === null) return;
    this.orientation.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.mesh = null;
    this.requestRender();
  }

  hasMesh(): boolean {
    return this.mesh !== null;
  }

  setFlatShading(flat: boolean): void {
    if (this.material.flatShading === flat) return;
    this.material.flatShading = flat;
    this.material.needsUpdate = true;
    this.requestRender();
  }

  setUp(axis: UpAxis): void {
    const next = new Quaternion().setFromUnitVectors(new Vector3(...AXIS_VECTORS[axis]), new Vector3(0, 1, 0));
    if (next.equals(this.orientation.quaternion)) return;
    this.orientation.quaternion.copy(next);
    this.orientation.updateMatrixWorld(true);
    if (this.mesh !== null) this.frame();
    this.requestRender();
  }

  setColors(colors: ViewerColors): void {
    this.scene.background = new Color(colors.background);
    this.material.color = new Color(colors.surface);
    this.requestRender();
  }

  /** Puts the whole bounding box in view, looking from the current direction. */
  frame(): void {
    const sphere = this.worldSphere();
    if (sphere === null) return;
    const direction = this.camera.position.clone().sub(this.controls.target);
    if (direction.lengthSq() === 0) direction.set(1, 0.6, 1);
    direction.normalize();
    const vertical = (this.camera.fov * Math.PI) / 180;
    const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * this.camera.aspect);
    // The sphere around the box is a loose bound (the box's corners are rarely occupied),
    // so sit a little closer than it strictly allows.
    const distance = (sphere.radius / Math.sin(Math.min(vertical, horizontal) / 2)) * 0.82;
    this.controls.target.copy(sphere.center);
    this.camera.position.copy(sphere.center).addScaledVector(direction, distance);
    this.fitClipPlanes();
    this.controls.update();
    this.requestRender();
  }

  /** Turns the camera around the model (radians); for keyboard control. */
  orbit(azimuth: number, polar: number): void {
    this.moved = true;
    const offset = this.camera.position.clone().sub(this.controls.target);
    const spherical = new Spherical().setFromVector3(offset);
    spherical.theta += azimuth;
    spherical.phi = Math.min(Math.PI - 0.01, Math.max(0.01, spherical.phi + polar));
    this.camera.position.copy(this.controls.target).add(offset.setFromSpherical(spherical));
    this.controls.update();
    this.requestRender();
  }

  /** Moves the camera towards (factor < 1) or away from the model. */
  zoom(factor: number): void {
    this.moved = true;
    const offset = this.camera.position.clone().sub(this.controls.target);
    const length = Math.min(this.controls.maxDistance, Math.max(this.controls.minDistance, offset.length() * factor));
    this.camera.position.copy(this.controls.target).add(offset.setLength(length));
    this.controls.update();
    this.requestRender();
  }

  /** Standard three-quarter view of the framed box. */
  resetView(): void {
    this.camera.position.copy(this.controls.target).add(new Vector3(1, 0.62, 1.15));
    this.camera.up.set(0, 1, 0);
    this.moved = false;
    this.frame();
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.pending);
    this.observer.disconnect();
    this.controls.dispose();
    this.clearMesh();
    this.material.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
  }

  private worldSphere(): Sphere | null {
    if (this.mesh === null || this.mesh.geometry.boundingBox === null) return null;
    this.orientation.updateMatrixWorld(true);
    const box = this.mesh.geometry.boundingBox.clone().applyMatrix4(this.orientation.matrixWorld);
    const sphere = box.getBoundingSphere(new Sphere());
    if (!(sphere.radius > 0) || !Number.isFinite(sphere.radius)) sphere.radius = 1;
    return sphere;
  }

  private fitClipPlanes(): void {
    const sphere = this.worldSphere();
    if (sphere === null) return;
    // Generous, scale-free limits: the model can be any size in its own units.
    this.camera.near = sphere.radius / 200;
    this.camera.far = sphere.radius * 400;
    this.controls.minDistance = sphere.radius * 0.05;
    this.controls.maxDistance = sphere.radius * 60;
    this.camera.updateProjectionMatrix();
  }

  private resize(): void {
    const width = Math.max(1, Math.floor(this.host.clientWidth));
    const height = Math.max(1, Math.floor(this.host.clientHeight));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    // Render now rather than on the next frame, or the canvas flashes empty while being resized.
    this.render();
  }

  private requestRender(): void {
    if (this.pending !== 0 || this.disposed) return;
    this.pending = requestAnimationFrame(() => {
      this.pending = 0;
      this.render();
    });
  }

  /**
   * Renders now and samples the picture on a 24 x 24 grid: how many samples differ from the
   * background. Zero with a mesh loaded means the surface is not being drawn (a GPU or driver
   * that renders nothing), which a screenshot alone would not tell.
   */
  probe(): { samples: number; surface: number } {
    this.renderer.render(this.scene, this.camera);
    const gl = this.renderer.getContext();
    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;
    const pixel = new Uint8Array(4);
    const background = new Uint8Array(4);
    gl.readPixels(1, 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, background);
    let surface = 0;
    const steps = 24;
    for (let i = 1; i < steps; i++) {
      for (let j = 1; j < steps; j++) {
        gl.readPixels(Math.floor((width * i) / steps), Math.floor((height * j) / steps), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
        if (Math.abs(pixel[0]! - background[0]!) + Math.abs(pixel[1]! - background[1]!) + Math.abs(pixel[2]! - background[2]!) > 12) surface += 1;
      }
    }
    return { samples: (steps - 1) * (steps - 1), surface };
  }

  private render(): void {
    if (this.disposed) return;
    this.renderer.render(this.scene, this.camera);
  }
}

/** Whether this browser can create a WebGL context at all (checked before the viewer chunk does real work). */
export function webglAvailable(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return canvas.getContext("webgl2") !== null || canvas.getContext("webgl") !== null;
  } catch {
    return false;
  }
}
