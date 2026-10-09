//! The `turntable` camera provider inside the `photos` command.

use crate::photos::fs::Stored as _;
use anyhow::{anyhow, bail};
use serde_json::json;

use crate::photos::cameras::CameraProvider;
use crate::photos::markers::linalg::Camera;
use crate::photos::options::Options;
use crate::photos::providers::{Provider, CAMERAS_TURNTABLE};
use crate::photos::run::Run;
use crate::photos::solution::{Lens, Solution};
use crate::photos::staging::capture_name;
use crate::photos::util;

use super::solver::{self, Matches};
use super::{features, matching};

pub struct Turntable;

/// A feature off the object that moves less than this between two photos is the background.
const STILL_PIXELS: f64 = 2.0;

fn in_mask(mask: &crate::inputs::Plane<u8>, p: [f64; 2]) -> bool {
    let (x, y) = (p[0].floor(), p[1].floor());
    x >= 0.0 && y >= 0.0 && (x as usize) < mask.width && (y as usize) < mask.height && mask.data[y as usize * mask.width + x as usize] > 127
}

/// Search the mask and a surrounding band one bounding-box side wide. Static
/// off-object features are removed by their lack of motion when matching.
fn turntable_region(mask: &crate::inputs::Plane<u8>) -> crate::inputs::Plane<u8> {
    let (w, h) = (mask.width, mask.height);
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            if mask.data[y * w + x] > 127 {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            }
        }
    }
    let mut region = mask.clone();
    if x1 <= x0 {
        return region;
    }
    let reach = (x1 - x0).max(y1 - y0);
    let (left, right, bottom) = (x0.saturating_sub(reach), (x1 + reach).min(w), (y1 + reach).min(h));
    for y in y0.saturating_sub(reach)..bottom {
        for x in left..right {
            region.data[y * w + x] = 255;
        }
    }
    region
}

impl CameraProvider for Turntable {
    fn info(&self) -> &'static Provider {
        &CAMERAS_TURNTABLE
    }

    fn check(&self, options: &Options) -> anyhow::Result<()> {
        if options.turntable.surface && options.open_turn {
            bail!("surface region currently requires one complete turn");
        }
        if let Some(path) = &options.turntable.matches {
            if !path.stored_file() {
                bail!("--turntable-matches {}: not found", path.display());
            }
        }
        Ok(())
    }

    fn recover(&self, run: &mut Run, lens: Option<&Lens>) -> anyhow::Result<Solution> {
        let lens = *lens.ok_or_else(|| anyhow!("the turntable provider needs the declared lens (--calibration)"))?;
        let (options, out) = (run.options, run.out);
        let camera = Camera::from_lens(&lens);
        let (count, threads, timeouts) = (options.photo_count, options.threads, &options.timeouts);
        let names: Vec<String> = (0..count).map(capture_name).collect();
        // Per photo and feature: inside the object's mask (only known for features found here).
        let mut object_side: Option<Vec<Vec<bool>>> = None;
        let matches = match &options.turntable.matches {
            // Debug input: features and matches made elsewhere (the prototype's `--export-matches`).
            Some(path) => {
                let loaded = Matches::from_json(&util::read_json(path)?)?;
                if loaded.keypoints.len() != count {
                    bail!("--turntable-matches lists {} photos, the capture has {count}", loaded.keypoints.len());
                }
                if options.turntable.surface {
                    let inside: anyhow::Result<Vec<Vec<bool>>> = loaded
                        .keypoints
                        .iter()
                        .enumerate()
                        .map(|(i, points)| {
                            let mask = crate::scene::read_gray(&out.join("masks").join(&names[i]))?;
                            Ok(points.iter().map(|&p| in_mask(&mask, p)).collect())
                        })
                        .collect();
                    object_side = Some(inside?);
                }
                loaded
            }
            None => {
                let settings = features::Settings { maximum: options.turntable.features, ..features::Settings::default() };
                let surface = options.turntable.surface;
                let found =
                    run.internal("cameras", "turntable-features", timeouts.features, 0.04, 0.4, "Detecting features", count, |watch| {
                        let work = |n: usize| -> anyhow::Result<(features::Features, Vec<bool>)> {
                            let photo = crate::photos::staging::open_photo(&out.join("work/contrast").join(&names[n]))?;
                            let mask = crate::scene::read_gray(&out.join("masks").join(&names[n]))?;
                            let region = if surface { turntable_region(&mask) } else { mask.clone() };
                            let found = features::detect(&photo.gray(), Some(&region), &settings);
                            let inside = found.points.iter().map(|p| in_mask(&mask, *p)).collect();
                            Ok((found, inside))
                        };
                        util::parallel(count, threads, work, watch)
                    })?;
                let (found, inside): (Vec<features::Features>, Vec<Vec<bool>>) = found.into_iter().unzip();
                object_side = Some(inside);
                let span = options.turntable.span;
                let pairs = matching::ring_pairs(count, span, options.open_turn);
                let matched = run.internal(
                    "cameras",
                    "turntable-matching",
                    timeouts.matching,
                    0.4,
                    0.6,
                    "Matching features",
                    pairs.len(),
                    |watch| {
                        let inside = object_side.as_ref().expect("set above");
                        let work = |k: usize| -> anyhow::Result<Vec<(u32, u32)>> {
                            let (i, j) = pairs[k];
                            let mut list = matching::match_pair(&found[i], &found[j], 0.8);
                            if surface {
                                // Off the object only what turns with it: the camera stands still, so a
                                // feature of the background is found at the same place in every photo.
                                list.retain(|&(a, b)| {
                                    let (p, q) = (found[i].points[a as usize], found[j].points[b as usize]);
                                    inside[i][a as usize] || inside[j][b as usize] || (p[0] - q[0]).hypot(p[1] - q[1]) >= STILL_PIXELS
                                });
                            }
                            Ok(list)
                        };
                        util::parallel(pairs.len(), threads, work, watch)
                    },
                )?;
                Matches {
                    keypoints: found.iter().map(|f| f.points.clone()).collect(),
                    pairs: pairs.iter().zip(matched).map(|(&(i, j), list)| (i, j, list)).collect(),
                }
            }
        };
        if options.keep_intermediates {
            crate::photos::fs::create_dir_all(out.join("sfm"))?;
            util::write_json(
                &out.join("sfm/photo-matches.json"),
                &json!({"schema":"crisp3ds_turntable_matches_v1","keypoints":matches.keypoints,"pairs":matches.pairs.iter().map(|(i,j,list)|json!({"first":i,"second":j,"matches":list})).collect::<Vec<_>>()}),
                1,
            )?;
        }
        let counted: Vec<(usize, usize, usize)> = matches.pairs.iter().map(|p| (p.0, p.1, p.2.len())).collect();
        let by_distance = matching::matches_by_distance(count, &counted);
        run.note("turntable_order", json!({"median_matches_by_distance": by_distance}));
        matching::check_order(&by_distance).map_err(|reason| anyhow!(reason))?;
        let planar = if options.turntable.surface {
            Some(super::planar::estimate(&matches, object_side.as_ref().expect("photo features"), &camera)?)
        } else {
            None
        };
        let settings = solver::Settings { open_turn: options.open_turn, wide_axis: options.turntable.surface, planar };
        let solved =
            run.internal("cameras", "turntable-solve", timeouts.sfm, 0.6, 0.82, "Recovering cameras (turntable solver)", 1, |_| {
                solver::solve(&matches, &camera, &settings)
            })?;
        let folder = out.join("sfm/turntable");
        crate::photos::fs::create_dir_all(&folder)?;
        let mut report = solved.report.clone();
        let mut counts: Vec<f64> = matches.keypoints.iter().map(|k| k.len() as f64).collect();
        let total: usize = matches.pairs.iter().map(|p| p.2.len()).sum();
        report["features_per_photo"] =
            json!({"min": counts.iter().cloned().fold(f64::INFINITY, f64::min), "median": crate::inputs::median_f64(&mut counts)});
        report["matches"] = json!({"pairs": matches.pairs.len(), "total": total});
        util::write_json(&folder.join("report.json"), &report, 1)?;
        if let Some(object) = report.as_object_mut() {
            object.remove("steps_deg");
            object.remove("adjustments");
        }
        let mut solution = solver::solution(&lens, &names, &matches, &solved);
        if options.turntable.surface {
            // The scene is located by points of the object, not of the turntable around it: a point
            // counts when most of the photos that see it see it inside the mask.
            let inside = object_side.as_ref().expect("surface features are found here");
            let mut votes = vec![(0usize, 0usize); solved.points.len()];
            for &(view, point, feature) in &solved.observations {
                votes[point].1 += 1;
                votes[point].0 += inside[view][feature as usize] as usize;
            }
            let object: Vec<_> = solved.points.iter().zip(&votes).filter(|(_, (a, n))| 2 * a > *n).map(|(p, _)| *p).collect();
            report["surface_points"] = json!({"object": object.len(), "turntable": solved.points.len() - object.len()});
            solution.object_points = Some(object);
        }
        run.note("turntable", report);
        Ok(solution)
    }

    fn intermediates(&self) -> &'static [&'static str] {
        &[]
    }
}
