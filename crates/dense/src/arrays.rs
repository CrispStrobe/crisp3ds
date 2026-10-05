//! Standalone `.npy` files (`sparse_points.npy`), which `npz.rs` does not expose.
//!
//! `.npz` archives go through `crate::npz`. This module only wraps one array in
//! the `.npy` container and back; its parser duplicates the private one in
//! `npz.rs` and should be unified with it.

use std::path::Path;

use anyhow::{anyhow, bail, Context};

use crate::npz::{Array, Data};

/// Reads a little-endian C-ordered `f4`, `f8`, `i4` or `i8` array.
pub fn read_npy(path: &Path) -> anyhow::Result<Array> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    from_npy(&bytes).with_context(|| path.display().to_string())
}

pub fn from_npy(bytes: &[u8]) -> anyhow::Result<Array> {
    if bytes.len() < 12 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not an .npy file");
    }
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        v => bail!("unsupported .npy version {v}"),
    };
    let header = std::str::from_utf8(bytes.get(start..start + header_len).ok_or_else(|| anyhow!("truncated .npy header"))?)?;
    let descr =
        header.split("'descr':").nth(1).and_then(|rest| rest.split('\'').nth(1)).ok_or_else(|| anyhow!("no descr in .npy header"))?;
    if header.split("'fortran_order':").nth(1).map(|s| s.trim_start().starts_with("True")).unwrap_or(false) {
        bail!("Fortran-ordered .npy is not supported");
    }
    let shape: Vec<usize> = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| anyhow!("no shape in .npy header"))?
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse::<usize>)
        .collect::<Result<_, _>>()?;
    let count: usize = shape.iter().product();
    let body = &bytes[start + header_len..];
    fn take<T: bytemuck::Pod>(body: &[u8], count: usize) -> anyhow::Result<Vec<T>> {
        let slice = body.get(..count * std::mem::size_of::<T>()).ok_or_else(|| anyhow!("truncated .npy data"))?;
        let mut out = vec![T::zeroed(); count];
        bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(slice);
        Ok(out)
    }
    let data = match descr {
        "<f4" => Data::F32(take(body, count)?),
        "<f8" => Data::F64(take(body, count)?),
        "<i4" => Data::I32(take(body, count)?),
        "<i8" => Data::I64(take(body, count)?),
        other => bail!("unsupported .npy dtype {other}"),
    };
    Ok(Array { shape, data })
}

/// The `.npy` (format 1.0) encoding of a float64 array.
pub fn npy_f64(shape: &[usize], values: &[f64]) -> Vec<u8> {
    let dims = match shape {
        [] => "()".to_string(),
        [only] => format!("({only},)"),
        more => format!("({})", more.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {dims}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + values.len() * 8);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(bytemuck::cast_slice(values));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_round_trip() {
        let values = [1.5, -2.0, 3.25, 0.0, 1e-300, 7.0];
        let bytes = npy_f64(&[2, 3], &values);
        assert_eq!((bytes.len() - 48) % 64, 0);
        let array = from_npy(&bytes).unwrap();
        assert_eq!(array.shape, [2, 3]);
        assert_eq!(array.to_f64(), values);
        assert!(from_npy(b"not numpy at all").is_err());
    }
}
