//! Small shared pieces of the photo front stage: JSON files written like the
//! reference writes them, SHA-256, a bounded worker pool and Python's number
//! formatting.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{bail, Context};
use serde::Serialize;
use serde_json::Value;

/// `json.dumps(value, indent=indent) + "\n"`; keys come out in alphabetical order.
pub fn write_json(path: &Path, value: &Value, indent: usize) -> anyhow::Result<()> {
    let spaces = vec![b' '; indent];
    let mut out = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(&spaces);
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
    value.serialize(&mut serializer)?;
    out.push(b'\n');
    crate::photos::fs::write(path, out).with_context(|| path.display().to_string())
}

pub fn read_json(path: &Path) -> anyhow::Result<Value> {
    let text = crate::photos::fs::read_to_string(path).with_context(|| path.display().to_string())?;
    serde_json::from_str(&text).with_context(|| path.display().to_string())
}

/// A JSON number, or a string holding one (AliceVision writes every number as a string). Booleans are refused.
pub fn number(value: &Value) -> anyhow::Result<f64> {
    let parsed = match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    };
    parsed.ok_or_else(|| anyhow::anyhow!("expected a number, found {value}"))
}

/// An identifier as Python's `str()` gives it for a JSON string or integer.
pub fn identifier(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Python's `repr(float)`: shortest digits that round-trip, `.0` on whole
/// numbers, exponent form below 1e-4 and from 1e16 on.
pub fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("exponent form");
    let exponent: i32 = exponent.parse().expect("exponent");
    if !(-4..16).contains(&exponent) {
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    let plain = format!("{value}");
    if plain.contains('.') {
        plain
    } else {
        plain + ".0"
    }
}

/// How Python's `str.format("{}")` prints a JSON number: integers plainly, floats as `repr`.
pub fn python_number(value: &Value) -> String {
    match value {
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::Number(n) => python_float(n.as_f64().unwrap_or(f64::NAN)),
        other => other.to_string(),
    }
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 of a byte string as lower-case hex (FIPS 180-4).
pub fn sha256(bytes: &[u8]) -> String {
    let mut state: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut tail = bytes[bytes.len() - bytes.len() % 64..].to_vec();
    tail.push(0x80);
    while tail.len() % 64 != 56 {
        tail.push(0);
    }
    tail.extend_from_slice(&((bytes.len() as u64) * 8).to_be_bytes());
    let whole = bytes.len() - bytes.len() % 64;
    for block in bytes[..whole].as_chunks::<64>().0.iter().chain(tail.as_chunks::<64>().0) {
        let mut w = [0u32; 64];
        for (target, word) in w.iter_mut().zip(block.as_chunks::<4>().0) {
            *target = u32::from_be_bytes(*word);
        }
        for n in 16..64 {
            let s0 = w[n - 15].rotate_right(7) ^ w[n - 15].rotate_right(18) ^ (w[n - 15] >> 3);
            let s1 = w[n - 2].rotate_right(17) ^ w[n - 2].rotate_right(19) ^ (w[n - 2] >> 10);
            w[n] = w[n - 16].wrapping_add(s0).wrapping_add(w[n - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for n in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ (!e & g);
            let t1 = h.wrapping_add(s1).wrapping_add(choose).wrapping_add(K[n]).wrapping_add(w[n]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(majority);
            (h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (value, add) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *value = value.wrapping_add(add);
        }
    }
    state.iter().map(|v| format!("{v:08x}")).collect()
}

pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    Ok(sha256(&crate::photos::fs::read(path).with_context(|| path.display().to_string())?))
}

/// Runs `work(index)` for every index on `threads` threads and returns the
/// results in order. `watch(done)` is called on the calling thread after every
/// finished item (and at least twice a second); returning an error stops the
/// pool after the items in flight and becomes the result.
pub fn parallel<T: Send>(
    count: usize,
    threads: usize,
    work: impl Fn(usize) -> anyhow::Result<T> + Sync,
    watch: &mut dyn FnMut(usize) -> anyhow::Result<()>,
) -> anyhow::Result<Vec<T>> {
    // A browser build with shared memory (`+atomics`) whose host started rayon's pool of Web
    // Workers: batches of `threads` items on the pool, `watch` on the calling thread between them.
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    if threads > 1 && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let mut out = Vec::with_capacity(count);
        let batch = threads.min(rayon::current_num_threads());
        for start in (0..count).step_by(batch) {
            let end = (start + batch).min(count);
            let done: Vec<anyhow::Result<T>> = (start..end).into_par_iter().map(&work).collect();
            for result in done {
                out.push(result?);
            }
            watch(end)?;
        }
        return Ok(out);
    }
    // Without threads (a browser): one item after the other on the calling thread.
    if cfg!(target_arch = "wasm32") {
        let mut out = Vec::with_capacity(count);
        for n in 0..count {
            out.push(work(n)?);
            watch(n + 1)?;
        }
        return Ok(out);
    }
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let (sender, receiver) = std::sync::mpsc::channel::<(usize, anyhow::Result<T>)>();
    let mut slots: Vec<Option<T>> = (0..count).map(|_| None).collect();
    let mut failure = None;
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, count.max(1)) {
            let sender = sender.clone();
            let (next, stop, work) = (&next, &stop, &work);
            scope.spawn(move || loop {
                let n = next.fetch_add(1, Ordering::Relaxed);
                if n >= count || stop.load(Ordering::Relaxed) {
                    break;
                }
                if sender.send((n, work(n))).is_err() {
                    break;
                }
            });
        }
        drop(sender);
        let mut done = 0;
        loop {
            match receiver.recv_timeout(std::time::Duration::from_millis(500)) {
                Ok((n, Ok(value))) => {
                    slots[n] = Some(value);
                    done += 1;
                }
                Ok((_, Err(error))) => {
                    failure.get_or_insert(error);
                    stop.store(true, Ordering::Relaxed);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if failure.is_none() {
                if let Err(error) = watch(done) {
                    failure = Some(error);
                    stop.store(true, Ordering::Relaxed);
                }
            }
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    let mut out = Vec::with_capacity(count);
    for slot in slots {
        match slot {
            Some(value) => out.push(value),
            None => bail!("worker pool stopped early"),
        }
    }
    Ok(out)
}

/// Pillow's `convert("L")` of an 8-bit RGB pixel (ITU-R 601-2 luma in 16-bit fixed point).
#[inline]
pub fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * 19595 + g as u32 * 38470 + b as u32 * 7471 + 0x8000) >> 16) as u8
}

/// Writes a single-channel 8-bit PNG.
pub fn save_gray(path: &Path, width: usize, height: usize, data: Vec<u8>) -> anyhow::Result<()> {
    let image = image::GrayImage::from_raw(width as u32, height as u32, data).ok_or_else(|| anyhow::anyhow!("image size mismatch"))?;
    crate::photos::fs::save_image(&image, path)
}

/// Writes an 8-bit RGB PNG.
pub fn save_rgb(path: &Path, width: usize, height: usize, data: Vec<u8>) -> anyhow::Result<()> {
    let image = image::RgbImage::from_raw(width as u32, height as u32, data).ok_or_else(|| anyhow::anyhow!("image size mismatch"))?;
    crate::photos::fs::save_image(&image, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_the_standard_vectors() {
        assert_eq!(sha256(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // One million times "a", which crosses many block boundaries.
        assert_eq!(sha256(&vec![b'a'; 1_000_000]), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    #[test]
    fn floats_print_like_python() {
        for (value, text) in [
            (1.0, "1.0"),
            (0.02, "0.02"),
            (47.957, "47.957"),
            (-0.17180718751091592, "-0.17180718751091592"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e16, "1e+16"),
            (123456789012345.0, "123456789012345.0"),
            (0.0001, "0.0001"),
            (2.5e22, "2.5e+22"),
        ] {
            assert_eq!(python_float(value), text);
        }
        assert_eq!(python_number(&serde_json::json!(0)), "0");
        assert_eq!(python_number(&serde_json::json!(5.0)), "5.0");
    }

    #[test]
    fn pool_keeps_order_reports_progress_and_stops_on_request() {
        let mut seen = 0;
        let squares = parallel(50, 3, |n| Ok(n * n), &mut |done| {
            seen = seen.max(done);
            Ok(())
        })
        .unwrap();
        assert_eq!(squares, (0..50).map(|n| n * n).collect::<Vec<_>>());
        assert_eq!(seen, 50);
        let failed = parallel(50, 3, |n| if n == 7 { anyhow::bail!("seven") } else { Ok(n) }, &mut |_| Ok(()));
        assert_eq!(failed.unwrap_err().to_string(), "seven");
        let stopped = parallel(50, 2, Ok, &mut |done| if done >= 3 { anyhow::bail!("cancelled") } else { Ok(()) });
        assert_eq!(stopped.unwrap_err().to_string(), "cancelled");
        assert!(parallel(0, 2, Ok, &mut |_| Ok(())).unwrap().is_empty());
    }

    #[test]
    fn luma_matches_pillow() {
        assert_eq!((luma(255, 255, 255), luma(255, 0, 0), luma(0, 255, 0), luma(0, 0, 255)), (255, 76, 150, 29));
    }
}
