//! File access of the photos stage. Everything goes through `crate::storage`,
//! so a run can read its photos from and write its scene to the in-memory tree
//! (`mem:` paths on any platform, every path in a browser) as well as to disk.
//! The functions mirror the `std::fs` ones they replace.

use std::io::Result;
use std::path::{Path, PathBuf};

use crate::storage;

pub use storage::{create_dir, create_dir_all, read, read_to_string, remove_dir_all, remove_file, rename, write};

/// Copies a file; returns the number of bytes.
pub fn copy(from: impl AsRef<Path>, to: impl AsRef<Path>) -> Result<u64> {
    let bytes = storage::read(from)?;
    let count = bytes.len() as u64;
    storage::write_owned(to, bytes)?;
    Ok(count)
}

/// Paths of the entries directly inside a directory, sorted (what `std::fs::read_dir` lists).
pub fn list(directory: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    storage::list(directory)
}

/// Size of a file, `None` when there is none.
pub fn file_len(path: impl AsRef<Path>) -> Option<u64> {
    let path = path.as_ref();
    if !storage::is_memory(path) {
        return std::fs::metadata(path).ok().filter(|m| m.is_file()).map(|m| m.len());
    }
    storage::is_file(path).then(|| storage::read(path).map(|b| b.len() as u64).ok()).flatten()
}

/// `std::path::absolute` on disk, unchanged in the in-memory tree.
pub fn absolute(path: impl AsRef<Path>) -> PathBuf {
    storage::absolute(path.as_ref()).unwrap_or_else(|_| path.as_ref().to_path_buf())
}

/// Decodes an image file (PNG or JPEG).
pub fn open_image(path: impl AsRef<Path>) -> anyhow::Result<image::DynamicImage> {
    storage::open_image(path)
}

/// Writes an image as PNG (the encoder and settings of `ImageBuffer::save`).
pub fn save_image<P, C>(image: &image::ImageBuffer<P, C>, path: impl AsRef<Path>) -> anyhow::Result<()>
where
    P: image::Pixel + image::PixelWithColorType,
    [P::Subpixel]: image::EncodableLayout,
    C: std::ops::Deref<Target = [P::Subpixel]>,
{
    let path = path.as_ref();
    let mut bytes = std::io::Cursor::new(Vec::new());
    image.write_to(&mut bytes, image::ImageFormat::Png).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    storage::write_owned(path, bytes.into_inner()).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

/// `exists`, `is_file` and `is_dir` that see the in-memory tree.
pub trait Stored {
    fn stored(&self) -> bool;
    fn stored_file(&self) -> bool;
    fn stored_dir(&self) -> bool;
}

impl Stored for Path {
    fn stored(&self) -> bool {
        storage::exists(self)
    }
    fn stored_file(&self) -> bool {
        storage::is_file(self)
    }
    fn stored_dir(&self) -> bool {
        storage::is_dir(self)
    }
}
