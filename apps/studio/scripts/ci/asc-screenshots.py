"""Puts App Store screenshots onto the app's versions in App Store Connect, or says what it would do.

    python scripts/ci/asc-screenshots.py <dir with manifest.json> --version 0.2.0 [--apply]

Without --apply (the default) it only reads: finds the iOS and macOS versions with that
version string, their en-US localization, the screenshot sets there, and prints which set
would be replaced by which files. With --apply it deletes the screenshots of each set it
has files for and uploads the new ones (reserve, upload the bytes, commit with the
checksum, wait until App Store Connect has processed them).

Credentials from the environment: ASC_KEY_ID, ASC_ISSUER_ID, ASC_APP_ID, and the key as
ASC_API_KEY_P8_BASE64 (base64 of the .p8) or ASC_API_KEY_PATH. Nothing secret is printed.
Needs `cryptography`.
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

API = "https://api.appstoreconnect.apple.com"
PLATFORM_OF = {"APP_IPHONE_67": "IOS", "APP_IPAD_PRO_3GEN_129": "IOS", "APP_DESKTOP": "MAC_OS"}


def b64url(data: bytes) -> bytes:
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def key_bytes() -> bytes:
    if os.environ.get("ASC_API_KEY_P8_BASE64"):
        text = os.environ["ASC_API_KEY_P8_BASE64"].strip()
        # Either convention: base64 of the file, or the PEM itself.
        return text.encode() if text.startswith("-----BEGIN") else base64.b64decode(text)
    return Path(os.environ["ASC_API_KEY_PATH"]).read_bytes()


def token() -> str:
    now = int(time.time())
    head = b64url(json.dumps({"alg": "ES256", "kid": os.environ["ASC_KEY_ID"], "typ": "JWT"}).encode())
    body = b64url(json.dumps({"iss": os.environ["ASC_ISSUER_ID"], "iat": now, "exp": now + 1000, "aud": "appstoreconnect-v1"}).encode())
    key = serialization.load_pem_private_key(key_bytes(), None)
    r, s = utils.decode_dss_signature(key.sign(head + b"." + body, ec.ECDSA(hashes.SHA256())))
    return (head + b"." + body + b"." + b64url(r.to_bytes(32, "big") + s.to_bytes(32, "big"))).decode()


def call(method, path, payload=None):
    request = urllib.request.Request(API + path, method=method, data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={"Authorization": "Bearer " + token(), "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            text = response.read()
            return response.status, json.loads(text) if text else {}
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def must(status, data, what):
    if status >= 300:
        details = "; ".join(f"{e.get('title')}: {e.get('detail')}" for e in data.get("errors", []))
        sys.exit(f"{what}: {status} {details}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--version", required=True)
    parser.add_argument("--locale", default="en-US")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    folder = Path(args.folder)
    rows = json.loads((folder / "manifest.json").read_text())
    app = os.environ["ASC_APP_ID"]
    by_type = {}
    for row in rows:
        by_type.setdefault(row["displayType"], []).append(folder / row["file"])

    versions = must(*call("GET", f"/v1/apps/{app}/appStoreVersions?filter[versionString]={args.version}&limit=10"), "versions")["data"]
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: version {args.version}, {len(rows)} screenshots")
    for display_type, files in sorted(by_type.items()):
        platform = PLATFORM_OF[display_type]
        version = next((v for v in versions if v["attributes"]["platform"] == platform), None)
        if version is None:
            print(f"  {display_type}: no {platform} version {args.version}; skipped")
            continue
        localizations = must(*call("GET", f"/v1/appStoreVersions/{version['id']}/appStoreVersionLocalizations"), "localizations")["data"]
        localization = next((l for l in localizations if l["attributes"]["locale"] == args.locale), None)
        if localization is None:
            print(f"  {display_type}: the {platform} version has no {args.locale} localization; skipped")
            continue
        sets = must(*call("GET", f"/v1/appStoreVersionLocalizations/{localization['id']}/appScreenshotSets"), "screenshot sets")["data"]
        current = next((s for s in sets if s["attributes"]["screenshotDisplayType"] == display_type), None)
        existing = []
        if current is not None:
            existing = must(*call("GET", f"/v1/appScreenshotSets/{current['id']}/appScreenshots"), "screenshots")["data"]
        print(f"  {display_type} ({platform} {args.version}, {args.locale}): set {'exists' if current else 'would be created'}, "
              f"{len(existing)} screenshot(s) there would be replaced by {len(files)}: {', '.join(f.name for f in files)}")
        if not args.apply:
            continue
        if current is None:
            current = must(*call("POST", "/v1/appScreenshotSets", {"data": {"type": "appScreenshotSets", "attributes": {"screenshotDisplayType": display_type},
                                                                       "relationships": {"appStoreVersionLocalization": {"data": {"type": "appStoreVersionLocalizations", "id": localization["id"]}}}}}), "create set")["data"]
        for old in existing:
            must(*call("DELETE", f"/v1/appScreenshots/{old['id']}"), "delete screenshot")
        for file in files:
            data = file.read_bytes()
            reserved = must(*call("POST", "/v1/appScreenshots", {"data": {"type": "appScreenshots", "attributes": {"fileName": file.name, "fileSize": len(data)},
                                                                         "relationships": {"appScreenshotSet": {"data": {"type": "appScreenshotSets", "id": current["id"]}}}}}), "reserve")["data"]
            for part in reserved["attributes"]["uploadOperations"]:
                chunk = data[part["offset"]:part["offset"] + part["length"]]
                headers = {h["name"]: h["value"] for h in part.get("requestHeaders", [])}
                urllib.request.urlopen(urllib.request.Request(part["url"], method=part["method"], data=chunk, headers=headers)).read()
            must(*call("PATCH", f"/v1/appScreenshots/{reserved['id']}", {"data": {"type": "appScreenshots", "id": reserved["id"],
                                                                                   "attributes": {"uploaded": True, "sourceFileChecksum": hashlib.md5(data).hexdigest()}}}), "commit")
            for _ in range(60):
                state = must(*call("GET", f"/v1/appScreenshots/{reserved['id']}"), "state")["data"]["attributes"]["assetDeliveryState"]["state"]
                if state in ("COMPLETE", "FAILED"):
                    break
                time.sleep(5)
            print(f"    {file.name}: {state}")
            if state != "COMPLETE":
                sys.exit(f"{file.name} was not accepted")


main()
