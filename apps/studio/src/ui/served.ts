/**
 * Whether this page may be served by an engine (`engine_server.py --static`), so that asking
 * the same origin for \`/api/health\` makes sense. An engine serves the app at the root of its
 * origin over http(s); a static host (GitHub Pages, a sub-path anywhere) does not, and asking
 * there only leaves a 404 in the console.
 */
export function servedByEngine(where: { protocol: string; hostname: string; pathname: string }): boolean {
  if (where.protocol !== "http:" && where.protocol !== "https:") return false;
  if (/(^|\.)github\.io$/.test(where.hostname)) return false;
  return where.pathname === "/" || where.pathname === "/index.html";
}
