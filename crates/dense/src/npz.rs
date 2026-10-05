//! NumPy `.npz` archives, as far as the pipeline files need them.
//!
//! An `.npz` is a zip archive whose members are `.npy` arrays. Reading covers what
//! `numpy.savez` and `numpy.savez_compressed` write (stored or deflated members, with or
//! without zip64 fields) for little-endian `float32`, `float64`, `int32` and `int64`
//! arrays and scalars in C order. Writing produces the same layout, so NumPy reads it.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::{anyhow, bail, ensure, Context, Result};

/// Element storage of one array.
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

/// One array: `shape` is empty for a scalar; elements are in C order.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Data,
}

impl Array {
    pub fn new(shape: &[usize], data: Data) -> Self {
        let array = Array { shape: shape.to_vec(), data };
        assert_eq!(array.len(), shape.iter().product::<usize>(), "shape does not match the element count");
        array
    }

    pub fn scalar_f32(value: f32) -> Self {
        Array { shape: Vec::new(), data: Data::F32(vec![value]) }
    }

    pub fn len(&self) -> usize {
        match &self.data {
            Data::F32(v) => v.len(),
            Data::F64(v) => v.len(),
            Data::I32(v) => v.len(),
            Data::I64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_f32(&self) -> bool {
        matches!(self.data, Data::F32(_))
    }

    /// NumPy's name of the element type.
    pub fn descr(&self) -> &'static str {
        match &self.data {
            Data::F32(_) => "<f4",
            Data::F64(_) => "<f8",
            Data::I32(_) => "<i4",
            Data::I64(_) => "<i8",
        }
    }

    pub fn to_f64(&self) -> Vec<f64> {
        match &self.data {
            Data::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Data::F64(v) => v.clone(),
            Data::I32(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Data::I64(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }

    /// Values as `float32` (what `ndarray.astype(np.float32)` gives).
    pub fn to_f32(&self) -> Vec<f32> {
        match &self.data {
            Data::F32(v) => v.clone(),
            Data::F64(v) => v.iter().map(|&x| x as f32).collect(),
            Data::I32(v) => v.iter().map(|&x| x as f32).collect(),
            Data::I64(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }

    /// Integer values; fails for floating-point arrays.
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        match &self.data {
            Data::I32(v) => Ok(v.iter().map(|&x| i64::from(x)).collect()),
            Data::I64(v) => Ok(v.clone()),
            _ => bail!("expected an integer array, found {}", self.descr()),
        }
    }

    /// The single element of a scalar or one-element array.
    pub fn scalar(&self) -> Result<f64> {
        ensure!(self.len() == 1, "expected a scalar, found shape {:?}", self.shape);
        Ok(self.to_f64()[0])
    }

    fn element_bytes(&self) -> Vec<u8> {
        match &self.data {
            Data::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Data::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Data::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Data::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
}

/// The arrays of one archive by name (the member name without `.npy`).
#[derive(Debug, Default)]
pub struct Npz {
    pub arrays: BTreeMap<String, Array>,
}

impl Npz {
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
        Self::from_bytes(&bytes).with_context(|| format!("{} is not a readable .npz", path.display()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut arrays = BTreeMap::new();
        for member in central_directory(bytes)? {
            let content = member_content(bytes, &member).with_context(|| format!("member {}", member.name))?;
            let name = member.name.strip_suffix(".npy").unwrap_or(&member.name).to_string();
            arrays.insert(name, parse_npy(&content).with_context(|| format!("member {}", member.name))?);
        }
        Ok(Npz { arrays })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<&Array> {
        self.arrays.get(name).ok_or_else(|| anyhow!("array {name:?} is missing"))
    }
}

struct Member {
    name: String,
    method: u16,
    crc: u32,
    compressed: u64,
    size: u64,
    offset: u64,
}

fn u16_at(bytes: &[u8], at: usize) -> Result<u16> {
    let slice = bytes.get(at..at + 2).ok_or_else(|| anyhow!("truncated zip structure"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn u32_at(bytes: &[u8], at: usize) -> Result<u32> {
    let slice = bytes.get(at..at + 4).ok_or_else(|| anyhow!("truncated zip structure"))?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn u64_at(bytes: &[u8], at: usize) -> Result<u64> {
    let slice = bytes.get(at..at + 8).ok_or_else(|| anyhow!("truncated zip structure"))?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

const END_OF_DIRECTORY: u32 = 0x0605_4b50;
const ZIP64_LOCATOR: u32 = 0x0706_4b50;
const ZIP64_END_OF_DIRECTORY: u32 = 0x0606_4b50;
const DIRECTORY_ENTRY: u32 = 0x0201_4b50;
const LOCAL_HEADER: u32 = 0x0403_4b50;

fn central_directory(bytes: &[u8]) -> Result<Vec<Member>> {
    ensure!(bytes.len() >= 22, "too short for a zip archive");
    // The end record is the last structure; only a comment of at most 65535 bytes may follow it.
    let lowest = bytes.len().saturating_sub(22 + 65535);
    let end = (lowest..=bytes.len() - 22)
        .rev()
        .find(|&at| u32_at(bytes, at).ok() == Some(END_OF_DIRECTORY))
        .ok_or_else(|| anyhow!("no zip end-of-directory record"))?;
    let mut count = u64::from(u16_at(bytes, end + 10)?);
    let mut offset = u64::from(u32_at(bytes, end + 16)?);
    if (count == 0xFFFF || offset == 0xFFFF_FFFF) && end >= 20 && u32_at(bytes, end - 20)? == ZIP64_LOCATOR {
        let record = usize::try_from(u64_at(bytes, end - 12)?)?;
        ensure!(u32_at(bytes, record)? == ZIP64_END_OF_DIRECTORY, "damaged zip64 end record");
        count = u64_at(bytes, record + 32)?;
        offset = u64_at(bytes, record + 48)?;
    }
    let mut at = usize::try_from(offset)?;
    let mut members = Vec::new();
    for _ in 0..count {
        ensure!(u32_at(bytes, at)? == DIRECTORY_ENTRY, "damaged zip directory");
        ensure!(u16_at(bytes, at + 8)? & 1 == 0, "encrypted zip members are not supported");
        let method = u16_at(bytes, at + 10)?;
        let crc = u32_at(bytes, at + 16)?;
        let mut compressed = u64::from(u32_at(bytes, at + 20)?);
        let mut size = u64::from(u32_at(bytes, at + 24)?);
        let name_length = usize::from(u16_at(bytes, at + 28)?);
        let extra_length = usize::from(u16_at(bytes, at + 30)?);
        let comment_length = usize::from(u16_at(bytes, at + 32)?);
        let mut local = u64::from(u32_at(bytes, at + 42)?);
        let name_bytes = bytes.get(at + 46..at + 46 + name_length).ok_or_else(|| anyhow!("truncated zip directory"))?;
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        // zip64 extra field: the 64-bit values of exactly those fields that are saturated, in this order.
        let extra_start = at + 46 + name_length;
        let mut field = extra_start;
        while field + 4 <= extra_start + extra_length {
            let id = u16_at(bytes, field)?;
            let length = usize::from(u16_at(bytes, field + 2)?);
            if id == 1 {
                let mut cursor = field + 4;
                for value in [&mut size, &mut compressed, &mut local] {
                    if *value == 0xFFFF_FFFF {
                        *value = u64_at(bytes, cursor)?;
                        cursor += 8;
                    }
                }
            }
            field += 4 + length;
        }
        members.push(Member { name, method, crc, compressed, size, offset: local });
        at = extra_start + extra_length + comment_length;
    }
    Ok(members)
}

fn member_content(bytes: &[u8], member: &Member) -> Result<Vec<u8>> {
    let at = usize::try_from(member.offset)?;
    ensure!(u32_at(bytes, at)? == LOCAL_HEADER, "damaged zip member header");
    let start = at + 30 + usize::from(u16_at(bytes, at + 26)?) + usize::from(u16_at(bytes, at + 28)?);
    let raw = bytes.get(start..start + usize::try_from(member.compressed)?).ok_or_else(|| anyhow!("truncated zip member"))?;
    let size = usize::try_from(member.size)?;
    let content = match member.method {
        0 => raw.to_vec(),
        8 => miniz_oxide::inflate::decompress_to_vec_with_limit(raw, size).map_err(|error| anyhow!("inflate failed: {error}"))?,
        other => bail!("zip compression method {other} is not supported (stored and deflate are)"),
    };
    ensure!(content.len() == size, "zip member has {} bytes, its directory says {}", content.len(), size);
    ensure!(crc32fast::hash(&content) == member.crc, "zip member checksum mismatch");
    Ok(content)
}

/// The value after `'key':` in an `.npy` header dictionary, up to the next top-level comma.
fn header_value<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let marker = format!("'{key}':");
    let start = header.find(&marker).ok_or_else(|| anyhow!("npy header lacks {key}"))? + marker.len();
    let rest = &header[start..];
    let mut depth = 0i32;
    for (position, character) in rest.char_indices() {
        match character {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' | '}' if depth == 0 => return Ok(rest[..position].trim()),
            _ => {}
        }
    }
    bail!("npy header is not terminated")
}

fn parse_npy(bytes: &[u8]) -> Result<Array> {
    ensure!(bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY", "not an .npy array");
    let (header_start, header_length) = match bytes[6] {
        1 => (10, usize::from(u16_at(bytes, 8)?)),
        2 | 3 => (12, usize::try_from(u32_at(bytes, 8)?)?),
        major => bail!("npy format version {major} is not supported"),
    };
    let header = bytes.get(header_start..header_start + header_length).ok_or_else(|| anyhow!("truncated npy header"))?;
    let header = std::str::from_utf8(header)?;
    let descr = header_value(header, "descr")?.trim_matches(|c| c == '\'' || c == '"');
    let fortran = header_value(header, "fortran_order")? == "True";
    let shape_text = header_value(header, "shape")?;
    let shape = shape_text
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<usize>().map_err(|_| anyhow!("bad npy shape {shape_text}")))
        .collect::<Result<Vec<usize>>>()?;
    ensure!(!fortran || shape.len() <= 1, "Fortran-ordered arrays are not supported");
    let count: usize = shape.iter().product();
    let body = &bytes[header_start + header_length..];
    let width = match descr {
        "<f4" | "<i4" => 4,
        "<f8" | "<i8" => 8,
        other => bail!("element type {other} is not supported (little-endian f4, f8, i4, i8 are)"),
    };
    ensure!(body.len() == count * width, "npy body has {} bytes, shape {:?} of {descr} needs {}", body.len(), shape, count * width);
    let data = match descr {
        "<f4" => Data::F32(body.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()),
        "<f8" => Data::F64(body.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect()),
        "<i4" => Data::I32(body.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect()),
        _ => Data::I64(body.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect()),
    };
    Ok(Array { shape, data })
}

fn npy_bytes(array: &Array) -> Vec<u8> {
    let shape = match array.shape.as_slice() {
        [] => "()".to_string(),
        [only] => format!("({only},)"),
        more => format!("({})", more.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", array.descr(), shape);
    // NumPy pads so the data starts at a multiple of 64 bytes; the header ends with a newline.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = Vec::with_capacity(10 + header.len() + array.len() * 8);
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&array.element_bytes());
    bytes
}

/// Writes an archive NumPy can load. `compress` selects deflate (`savez_compressed`) or stored (`savez`).
pub fn write(path: &Path, arrays: &[(&str, &Array)], compress: bool) -> Result<()> {
    let mut out = Vec::new();
    let mut directory = Vec::new();
    for (name, array) in arrays {
        let content = npy_bytes(array);
        let stored = if compress { miniz_oxide::deflate::compress_to_vec(&content, 6) } else { content.clone() };
        let member = format!("{name}.npy");
        ensure!(
            content.len() < u32::MAX as usize && out.len() < u32::MAX as usize && arrays.len() < 0xFFFF,
            "archives of 4 GiB or more are not supported"
        );
        let method: u16 = if compress { 8 } else { 0 };
        let mut fields = Vec::new();
        fields.extend_from_slice(&20u16.to_le_bytes()); // version needed
        fields.extend_from_slice(&0u16.to_le_bytes()); // flags
        fields.extend_from_slice(&method.to_le_bytes());
        fields.extend_from_slice(&0u16.to_le_bytes()); // time
        fields.extend_from_slice(&0x21u16.to_le_bytes()); // date: 1980-01-01
        fields.extend_from_slice(&crc32fast::hash(&content).to_le_bytes());
        fields.extend_from_slice(&(stored.len() as u32).to_le_bytes());
        fields.extend_from_slice(&(content.len() as u32).to_le_bytes());
        fields.extend_from_slice(&(member.len() as u16).to_le_bytes());
        fields.extend_from_slice(&0u16.to_le_bytes()); // extra length
        let offset = out.len() as u32;
        out.extend_from_slice(&LOCAL_HEADER.to_le_bytes());
        out.extend_from_slice(&fields);
        out.extend_from_slice(member.as_bytes());
        out.extend_from_slice(&stored);
        directory.extend_from_slice(&DIRECTORY_ENTRY.to_le_bytes());
        directory.extend_from_slice(&20u16.to_le_bytes()); // version made by
        directory.extend_from_slice(&fields);
        directory.extend_from_slice(&0u16.to_le_bytes()); // comment length
        directory.extend_from_slice(&0u16.to_le_bytes()); // disk
        directory.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
        directory.extend_from_slice(&0u32.to_le_bytes()); // external attributes
        directory.extend_from_slice(&offset.to_le_bytes());
        directory.extend_from_slice(member.as_bytes());
    }
    let directory_offset = out.len() as u32;
    out.extend_from_slice(&directory);
    out.extend_from_slice(&END_OF_DIRECTORY.to_le_bytes());
    out.extend_from_slice(&[0u8; 4]); // disk numbers
    out.extend_from_slice(&(arrays.len() as u16).to_le_bytes());
    out.extend_from_slice(&(arrays.len() as u16).to_le_bytes());
    out.extend_from_slice(&(directory.len() as u32).to_le_bytes());
    out.extend_from_slice(&directory_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment length
    let mut file = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
    file.write_all(&out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dense-native").join(name)
    }

    /// The arrays `tests/fixtures/dense-native/make_fixtures.py` stores in both archives.
    fn check_fixture(npz: &Npz) {
        let index = npz.get("index").unwrap();
        assert_eq!(index.shape, vec![4, 3]);
        assert_eq!(index.data, Data::I32(vec![0, 1, 2, 3, 4, 5, -6, 7, 8, 9, 10, 2147483647]));
        let total = npz.get("total").unwrap();
        assert_eq!(total.shape, vec![4]);
        assert_eq!(total.data, Data::F32(vec![-1.5, 0.0, 0.25, 3.0e-7]));
        assert_eq!(npz.get("shape").unwrap().data, Data::I64(vec![5, 6, 7]));
        assert_eq!(npz.get("shape").unwrap().to_i64().unwrap(), vec![5, 6, 7]);
        let voxel = npz.get("voxel").unwrap();
        assert!(voxel.shape.is_empty() && voxel.is_f32());
        assert_eq!(voxel.scalar().unwrap(), f64::from(0.05f32));
        let down = npz.get("support_down").unwrap();
        assert_eq!(down.data, Data::F64(vec![0.0, -0.6, 0.8]));
        assert!(npz.get("support_height").unwrap().scalar().unwrap().is_nan());
        let grid = npz.get("grid").unwrap();
        assert_eq!(grid.shape, vec![2, 3, 4]);
        assert_eq!(grid.to_f64(), (0..24).map(f64::from).collect::<Vec<f64>>());
        assert_eq!(npz.get("empty").unwrap().shape, vec![0, 3]);
        assert!(npz.get("missing").is_err() && !npz.contains("missing"));
    }

    #[test]
    fn reads_archives_written_by_numpy() {
        check_fixture(&Npz::read(&fixture("arrays-deflated.npz")).unwrap());
        check_fixture(&Npz::read(&fixture("arrays-stored.npz")).unwrap());
    }

    #[test]
    fn written_archives_read_back_identically() {
        let original = Npz::read(&fixture("arrays-deflated.npz")).unwrap();
        let entries: Vec<(&str, &Array)> = original.arrays.iter().map(|(name, array)| (name.as_str(), array)).collect();
        for compress in [false, true] {
            let path = std::env::temp_dir().join(format!("crisp3ds-npz-{}-{}.npz", std::process::id(), compress));
            write(&path, &entries, compress).unwrap();
            let again = Npz::read(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            check_fixture(&again);
            assert_eq!(again.arrays.len(), original.arrays.len());
        }
    }

    #[test]
    fn refuses_what_it_cannot_read() {
        assert!(Npz::from_bytes(b"not a zip archive at all, just some text").is_err());
        let mut damaged = std::fs::read(fixture("arrays-stored.npz")).unwrap();
        // Flip one byte of the first member: the checksum must catch it.
        let at = 30 + usize::from(u16_at(&damaged, 26).unwrap()) + usize::from(u16_at(&damaged, 28).unwrap()) + 100;
        damaged[at] ^= 0x40;
        assert!(Npz::from_bytes(&damaged).is_err());
        let big_endian = b"\x93NUMPY\x01\x00\x46\x00{'descr': '>f4', 'fortran_order': False, 'shape': (1,), }            \n\0\0\0\0";
        assert!(parse_npy(big_endian).is_err());
    }
}
