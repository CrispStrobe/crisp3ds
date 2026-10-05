//! Dense 3-D grids in C order (`[x][y][z]`, z fastest) and line-wise passes over them.

use rayon::prelude::*;

/// Extent of a grid along x, y and z.
pub type Dims = [usize; 3];

pub fn count(dims: Dims) -> usize {
    dims[0] * dims[1] * dims[2]
}

/// Linear position of a voxel.
#[inline]
pub fn position(dims: Dims, x: usize, y: usize, z: usize) -> usize {
    (x * dims[1] + y) * dims[2] + z
}

struct Shared<T>(*mut T);
// Used only to hand disjoint sets of elements to different threads (see `for_each_line`).
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Send> Sync for Shared<T> {}

/// Runs `pass` on every line of the grid along `axis`, in place and in parallel.
///
/// `pass` gets the line as a contiguous slice (copied out and back where the grid does not
/// store it contiguously) and a per-thread scratch value from `scratch`.
pub fn for_each_line<T, S, M, P>(data: &mut [T], dims: Dims, axis: usize, scratch: M, pass: P)
where
    T: Copy + Default + Send + Sync,
    M: Fn() -> S + Send + Sync,
    P: Fn(&mut [T], &mut S) + Send + Sync,
{
    let [nx, ny, nz] = dims;
    assert_eq!(data.len(), count(dims));
    if data.is_empty() {
        return;
    }
    match axis {
        2 => data.par_chunks_mut(nz).for_each_init(&scratch, |state, line| pass(line, state)),
        1 => data.par_chunks_mut(ny * nz).for_each_init(
            || (scratch(), vec![T::default(); ny]),
            |(state, line), slab| {
                for z in 0..nz {
                    for (y, value) in line.iter_mut().enumerate() {
                        *value = slab[y * nz + z];
                    }
                    pass(line, state);
                    for (y, value) in line.iter().enumerate() {
                        slab[y * nz + z] = *value;
                    }
                }
            },
        ),
        0 => {
            let shared = Shared(data.as_mut_ptr());
            (0..ny).into_par_iter().for_each_init(
                || (scratch(), vec![T::default(); nx * nz]),
                |(state, plane), y| {
                    let shared = &shared;
                    // The plane of constant y, transposed so that lines along x are contiguous.
                    for x in 0..nx {
                        // SAFETY: rows (x, y, ..) for one y are touched by this iteration only,
                        // and they lie inside `data` (checked by the length assertion above).
                        let row = unsafe { std::slice::from_raw_parts(shared.0.add((x * ny + y) * nz), nz) };
                        for (z, value) in row.iter().enumerate() {
                            plane[z * nx + x] = *value;
                        }
                    }
                    for line in plane.chunks_exact_mut(nx) {
                        pass(line, state);
                    }
                    for x in 0..nx {
                        // SAFETY: as above; no other reference to this row exists now.
                        let row = unsafe { std::slice::from_raw_parts_mut(shared.0.add((x * ny + y) * nz), nz) };
                        for (z, value) in row.iter_mut().enumerate() {
                            *value = plane[z * nx + x];
                        }
                    }
                },
            );
        }
        _ => panic!("a grid has three axes"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_line_of_every_axis_is_visited_once() {
        let dims = [3, 4, 5];
        for axis in 0..3 {
            let mut data: Vec<i32> = (0..60).collect();
            // Replace each element by 1000 * (line length) + its position within the line.
            for_each_line(
                &mut data,
                dims,
                axis,
                || (),
                |line, _| {
                    let step = line[1] - line[0];
                    assert!(line.windows(2).all(|pair| pair[1] - pair[0] == step));
                    let length = line.len() as i32;
                    line.iter_mut().enumerate().for_each(|(i, value)| *value = 1000 * length + i as i32);
                },
            );
            for x in 0..3 {
                for y in 0..4 {
                    for z in 0..5 {
                        let along = [x, y, z][axis] as i32;
                        assert_eq!(data[position(dims, x, y, z)], 1000 * dims[axis] as i32 + along);
                    }
                }
            }
        }
    }
}
