//! The providers of the photo front stage and what each needs: the table
//! behind `crisp3ds-dense photos --list-providers`, which an application reads
//! to offer only what its platform can run.

use serde_json::{json, Value};

/// One provider of a module (`masks` or `cameras`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Provider {
    pub module: &'static str,
    pub name: &'static str,
    /// How it is selected on the command line.
    pub selector: &'static str,
    pub summary: &'static str,
    /// Platforms it can run on: desktop (macOS, Windows, Linux), mobile (iOS, Android), wasm (browser).
    pub desktop: bool,
    pub mobile: bool,
    pub wasm: bool,
    /// External programs it starts; empty when everything runs in this crate.
    pub external: &'static [&'static str],
    pub license: &'static str,
    pub default: bool,
}

pub const MASKS_THRESHOLD: Provider = Provider {
    module: "masks",
    name: "threshold",
    selector: "--masks threshold",
    summary: "Dark object on a light backdrop: grey threshold (fixed or Otsu), largest dark region, dark-hole cleanup",
    desktop: true,
    mobile: true,
    wasm: true,
    external: &[],
    license: "this crate (AGPL-3.0-only)",
    default: false,
};

pub const MASKS_IMPORT: Provider = Provider {
    module: "masks",
    name: "import",
    selector: "--masks import:DIR",
    summary: "Masks made elsewhere, one 8-bit PNG per photo; the dark-hole cleanup still runs",
    desktop: true,
    mobile: true,
    wasm: true,
    external: &[],
    license: "this crate (AGPL-3.0-only)",
    default: false,
};

pub const MASKS_EXTERNAL_SAM: Provider = Provider {
    module: "masks",
    name: "external-sam",
    selector: "--masks external-sam",
    summary: "SAM 2.1 prompted by the threshold masks, through scripts/turntable_mesh/segment.py in an external Python interpreter",
    desktop: true,
    mobile: false,
    wasm: false,
    external: &["Python interpreter with PyTorch", "SAM 2 source checkout", "SAM 2.1 checkpoint"],
    license: "SAM 2 code and checkpoints Apache-2.0; PyTorch BSD-3-Clause",
    default: true,
};

pub const CAMERAS_COLMAP: Provider = Provider {
    module: "cameras",
    name: "colmap",
    selector: "--cameras colmap",
    summary: "COLMAP executable: SIFT features, exhaustive or ordered matching, incremental mapper with the declared lens fixed",
    desktop: true,
    mobile: false,
    wasm: false,
    external: &["colmap"],
    license: "COLMAP BSD-3-Clause; its dependencies carry their own licenses",
    default: true,
};

pub const CAMERAS_ALICEVISION: Provider = Provider {
    module: "cameras",
    name: "alicevision",
    selector: "--cameras alicevision",
    summary: "AliceVision executables: SIFT features inside the masks, matching, global SfM with the declared lens locked",
    desktop: true,
    mobile: false,
    wasm: false,
    external: &[
        "aliceVision_cameraInit",
        "aliceVision_featureExtraction",
        "aliceVision_imageMatching",
        "aliceVision_featureMatching",
        "aliceVision_globalSfM",
    ],
    license: "AliceVision MPL-2.0 (parts derived from libmv MIT); its dependencies carry their own licenses",
    default: false,
};

pub const CAMERAS_IMPORT: Provider = Provider {
    module: "cameras",
    name: "import",
    selector: "--cameras import:PATH",
    summary: "An existing solution: AliceVision .sfm file or COLMAP model directory (text or binary)",
    desktop: true,
    mobile: true,
    wasm: true,
    external: &[],
    license: "this crate (AGPL-3.0-only)",
    default: false,
};

pub const CAMERAS_MARKERS: Provider = Provider {
    module: "cameras",
    name: "markers",
    selector: "--cameras markers --markers-mat FILE",
    summary: "A printed mat of ArUco markers under the object: pose per photo from the marker corners, in millimetres, right-handed",
    desktop: true,
    mobile: true,
    wasm: true,
    external: &[],
    license: "this crate (AGPL-3.0-only); marker codes of OpenCV's DICT_4X4_50",
    default: false,
};

pub const CAMERAS_TURNTABLE: Provider = Provider {
    module: "cameras",
    name: "turntable",
    selector: "--cameras turntable",
    summary: "Our own solver for one turn of ordered photos with a known lens: SIFT-class features, a repeated rotation as the start, bundle adjustment with free poses",
    desktop: true,
    mobile: true,
    wasm: true,
    external: &[],
    license: "this crate (AGPL-3.0-only)",
    default: false,
};

pub const PROVIDERS: [Provider; 8] = [
    MASKS_THRESHOLD,
    MASKS_IMPORT,
    MASKS_EXTERNAL_SAM,
    CAMERAS_COLMAP,
    CAMERAS_ALICEVISION,
    CAMERAS_TURNTABLE,
    CAMERAS_MARKERS,
    CAMERAS_IMPORT,
];

/// The provider table as JSON. `available` says whether this build can run the
/// provider at all: external programs cannot be started from a browser build.
pub fn listing() -> Value {
    let here = if cfg!(target_arch = "wasm32") {
        "wasm"
    } else if cfg!(any(target_os = "ios", target_os = "android")) {
        "mobile"
    } else {
        "desktop"
    };
    let row = |p: &Provider| {
        let supported = match here {
            "wasm" => p.wasm,
            "mobile" => p.mobile,
            _ => p.desktop,
        };
        json!({
            "name": p.name, "selector": p.selector, "summary": p.summary, "default": p.default,
            "platforms": {"desktop": p.desktop, "mobile": p.mobile, "wasm": p.wasm},
            "available": supported, "external": p.external, "license": p.license,
        })
    };
    let of = |module: &str| PROVIDERS.iter().filter(|p| p.module == module).map(row).collect::<Vec<_>>();
    json!({
        "schema": "crisp3ds_photo_providers_v1", "platform": here, "masks": of("masks"), "cameras": of("cameras"),
        "undistortion": {"name": "native", "summary": "radial k1, k2, k3: photos bilinear, masks nearest; in this crate on every platform",
                         "platforms": {"desktop": true, "mobile": true, "wasm": true}},
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_lists_every_provider_once_with_one_default_per_module() {
        let table = listing();
        assert_eq!(table["schema"], "crisp3ds_photo_providers_v1");
        let expected: [(&str, &[&str]); 2] = [
            ("masks", &["threshold", "import", "external-sam"]),
            ("cameras", &["colmap", "alicevision", "turntable", "markers", "import"]),
        ];
        for (module, names) in expected {
            let rows = table[module].as_array().unwrap();
            assert_eq!(rows.iter().map(|r| r["name"].as_str().unwrap()).collect::<Vec<_>>(), names);
            assert_eq!(rows.iter().filter(|r| r["default"] == true).count(), 1);
            // Whatever starts another program cannot run in a browser.
            assert!(rows.iter().all(|r| r["external"].as_array().unwrap().is_empty() || r["platforms"]["wasm"] == false));
        }
        assert_eq!(table["cameras"][0]["license"].as_str().unwrap().split(';').next(), Some("COLMAP BSD-3-Clause"));
    }
}
