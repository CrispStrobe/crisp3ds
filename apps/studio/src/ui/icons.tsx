/** Inline icons (24-unit grid, stroke follows the text colour). Decorative: always paired with a text label or aria-label. */

const PATHS = {
  play: "M7 4.5v15l12-7.5z",
  pause: "M7 4.5h3.5v15H7zM13.5 4.5H17v15h-3.5z",
  prev: "M15.5 5l-7 7 7 7",
  next: "M8.5 5l7 7-7 7",
  first: "M17 5l-7 7 7 7M7 5v14",
  last: "M7 5l7 7-7 7M17 5v14",
  close: "M6 6l12 12M18 6L6 18",
  plus: "M12 5v14M5 12h14",
  minus: "M5 12h14",
  check: "M5 12.5l4.5 4.5L19 7.5",
  fit: "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5",
  sun: "M12 7.5a4.5 4.5 0 100 9 4.5 4.5 0 000-9zM12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8",
  moon: "M20 14.5A8 8 0 019.5 4a8 8 0 1010.5 10.5z",
  auto: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 3v18",
  stop: "M6.5 6.5h11v11h-11z",
  alert: "M12 3.5l9.5 16.5h-19zM12 10v4.5M12 17.2v.3",
  cube: "M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12L4 7.5M12 12v9",
  back: "M10 6l-6 6 6 6M4 12h16",
  refresh: "M19.5 12a7.5 7.5 0 11-2.2-5.3M19.5 4v4.5H15",
} as const;

export type IconName = keyof typeof PATHS;

const FILLED: ReadonlySet<IconName> = new Set(["play", "pause", "stop"]);

export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const filled = FILLED.has(name);
  return (
    <svg
      class="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke={filled ? "none" : "currentColor"}
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
