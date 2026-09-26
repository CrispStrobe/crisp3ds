#!/usr/bin/env python3
"""Check the exact locked application dependencies against the release policy.

This is a metadata gate, not a per-file legal audit. Run after npm ci; Cargo
metadata fetches the crates specified by Cargo.lock. --update-inventory writes
a reviewable snapshot when dependencies intentionally change.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tomllib
import urllib.parse
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "apps" / "desktop"
INVENTORY = ROOT / "dependencies" / "inventory.json"
NATIVE = ROOT / "dependencies" / "native.json"

# Deliberately narrow. Additions need an explicit policy review.
ALLOWED = {
    "0BSD", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0",
    "ISC", "MIT", "MIT-0", "MPL-2.0", "Unlicense", "Zlib", "Unicode-3.0",
    # Selected OpenCV source has these permissive notices; see docs/OPENCV.md.
    "BSL-1.0", "IJG", "Libpng-2.0", "Sun-FDLIBM",
}
ALLOWED_EXCEPTIONS = {("Apache-2.0", "LLVM-exception")}


def selected_license(expression):
    if not isinstance(expression, str) or not expression.strip():
        return None
    # Cargo historically used '/' for dual licensing. Preserve the original
    # field in the inventory, but interpret that documented form as OR.
    normalized = re.sub(r"\s*/\s*", " OR ", expression)
    tokens = re.findall(r"[A-Za-z0-9.-]+|[()]", normalized)
    if "".join(tokens) != re.sub(r"\s+", "", normalized):
        return None
    position = 0

    def atom():
        nonlocal position
        if position >= len(tokens):
            raise ValueError("incomplete expression")
        if tokens[position] == "(":
            position += 1
            value = alternatives()
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("unclosed expression")
            position += 1
            return value
        value = tokens[position]
        if value in {"AND", "OR", ")", "WITH"}:
            raise ValueError("invalid license identifier")
        position += 1
        if position < len(tokens) and tokens[position] == "WITH":
            position += 1
            if position >= len(tokens):
                raise ValueError("missing exception")
            exception = tokens[position]
            position += 1
            return f"{value} WITH {exception}" if (value, exception) in ALLOWED_EXCEPTIONS else None
        return value if value in ALLOWED else None

    def conjunction():
        nonlocal position
        value = atom()
        while position < len(tokens) and tokens[position] == "AND":
            position += 1
            next_value = atom()
            value = f"{value} AND {next_value}" if value and next_value else None
        return value

    def alternatives():
        nonlocal position
        value = conjunction()
        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            next_value = conjunction()
            value = value or next_value
        return value

    try:
        result = alternatives()
        return result if position == len(tokens) else None
    except ValueError:
        return None


def license_allowed(expression):
    return selected_license(expression) is not None


def npm_registry_license(name, version):
    # The lock contains optional packages for other OS/CPU targets which npm ci
    # correctly omits locally. Verify their exact version metadata at npmjs.
    encoded = urllib.parse.quote(name, safe="")
    url = f"https://registry.npmjs.org/{encoded}/{urllib.parse.quote(version, safe='')}"
    with urllib.request.urlopen(url, timeout=20) as response:
        metadata = json.load(response)
    if metadata.get("name") != name or metadata.get("version") != version:
        raise ValueError(f"registry metadata mismatch: {name}@{version}")
    return metadata.get("license", "")


def npm_entries():
    lock_path = DESKTOP / "package-lock.json"
    if not lock_path.exists():
        raise ValueError(f"missing lockfile: {lock_path.relative_to(ROOT)}")
    lock = json.loads(lock_path.read_text())
    if lock.get("lockfileVersion") not in (2, 3):
        raise ValueError("unsupported npm lockfile version")
    results = []
    for location, item in lock.get("packages", {}).items():
        if not location:
            continue
        if not location.startswith("node_modules/"):
            raise ValueError(f"unexpected npm lock path: {location}")
        name = location.rsplit("node_modules/", 1)[-1]
        version = item.get("version")
        if not name or not version:
            raise ValueError(f"incomplete npm lock entry: {location}")
        if not item.get("resolved") or not item.get("integrity"):
            raise ValueError(f"unverified npm lock source: {location}")
        metadata = DESKTOP / location / "package.json"
        if metadata.is_file():
            package = json.loads(metadata.read_text())
            if package.get("name") != name or package.get("version") != version:
                raise ValueError(f"npm lock/install mismatch: {location}")
            license_id = package.get("license") or item.get("license", "")
        elif item.get("optional"):
            license_id = item.get("license") or npm_registry_license(name, version)
        else:
            raise ValueError(f"missing installed npm metadata: {location}; run npm ci")
        results.append({
            "ecosystem": "npm",
            "name": name,
            "version": version,
            "source": item.get("resolved", ""),
            "integrity": item.get("integrity", ""),
            "license": license_id,
            "scope": "build" if item.get("dev") else "runtime",
        })
    return results


def cargo_entries():
    manifest = DESKTOP / "src-tauri" / "Cargo.toml"
    if not manifest.exists():
        return []
    lock_path = manifest.parent / "Cargo.lock"
    if not lock_path.exists():
        raise ValueError(f"missing lockfile: {lock_path.relative_to(ROOT)}")
    result = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1", "--manifest-path", str(manifest)],
        capture_output=True, text=True, check=True,
    )
    metadata = json.loads(result.stdout)
    lock = tomllib.loads(lock_path.read_text())
    checksums = {(p["name"], p["version"], p.get("source")): p.get("checksum", "")
                 for p in lock.get("package", [])}
    packages = {p["id"]: p for p in metadata["packages"]}
    workspace_members = set(metadata["workspace_members"])
    results = []
    for package_id in metadata["resolve"]["nodes"]:
        package = packages[package_id["id"]]
        if not package.get("source"):
            if package["id"] not in workspace_members:
                raise ValueError(f"unreviewed local/path Cargo dependency: {package['name']}")
            continue  # First-party workspace source.
        checksum = checksums.get((package["name"], package["version"], package["source"]), "")
        if not checksum:
            raise ValueError(f"unverified Cargo lock source: {package['name']}@{package['version']}")
        results.append({
            "ecosystem": "cargo",
            "name": package["name"],
            "version": package["version"],
            "source": package["source"],
            "integrity": checksum,
            "license": package.get("license") or "",
            "scope": "runtime-or-build",
        })
    return results


def native_entries():
    source = json.loads(NATIVE.read_text())
    if source.get("schemaVersion") != 1 or not isinstance(source.get("packages"), list):
        raise ValueError("invalid dependencies/native.json")
    for package in source["packages"]:
        if package.get("status") != "approved" or not all(package.get(key) for key in ("name", "version", "source", "license", "integrity", "sourcePath")):
            raise ValueError("native entry is unapproved or unpinned")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", package["integrity"]):
            raise ValueError("native entry has invalid SHA-256 integrity")
        source_path = pathlib.PurePosixPath(package["sourcePath"])
        if not package["source"].startswith("https://") or source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError("native entry has invalid pinned source location")
        if not isinstance(package.get("licenseFiles"), list) or not package["licenseFiles"]:
            raise ValueError("native entry must list source license files")
    return source["packages"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-inventory", action="store_true")
    args = parser.parse_args()
    try:
        entries = sorted(npm_entries() + cargo_entries() + native_entries(),
                         key=lambda p: (p["ecosystem"], p["name"], p["version"], p["source"]))
        rejected = [f"{p.get('ecosystem')}:{p.get('name')}@{p.get('version')} ({p.get('license')!r})"
                    for p in entries if not license_allowed(p.get("license"))]
        if rejected:
            raise ValueError("unknown or disallowed license metadata:\n  " + "\n  ".join(rejected))
        if len({(p["ecosystem"], p["name"], p["version"], p["source"]) for p in entries}) != len(entries):
            raise ValueError("duplicate dependency identity")
        for package in entries:
            package["selectedLicense"] = selected_license(package["license"])
        snapshot = {"schemaVersion": 1, "packages": entries}
        if args.update_inventory:
            INVENTORY.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
            print(f"wrote {len(entries)} entries to {INVENTORY.relative_to(ROOT)}")
        elif not INVENTORY.exists() or json.loads(INVENTORY.read_text()) != snapshot:
            raise ValueError("dependency inventory differs from lockfiles/install; review then run --update-inventory")
        else:
            print(f"approved metadata matches {len(entries)} exact dependency entries")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        print(f"dependency check failed: cargo metadata: {detail or exc}", file=sys.stderr)
        return 1
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"dependency check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
