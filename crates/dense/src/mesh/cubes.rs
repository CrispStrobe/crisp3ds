//! Iso-surface extraction at zero: marching cubes with the asymptotic decider on faces.
//!
//! The reference calls scikit-image's `marching_cubes` (Lewiner's method). What the pipeline
//! needs from it is a closed 2-manifold whenever the field permits one. That is obtained here
//! by construction instead of from the 33-case tables:
//!
//! * A cell's surface is described by its boundary: on each of its six faces the contour of
//!   the bilinear interpolant is drawn as segments between the crossings on the face's edges
//!   (marching squares). A face with two diagonally opposite inside corners is ambiguous;
//!   it is resolved by the sign of the interpolant at its saddle point (Nielson and Hamann's
//!   asymptotic decider), which depends on the four corner values of that face only. The two
//!   cells sharing a face therefore draw the same segments on it.
//! * Every crossing lies on two faces of the cell and ends exactly one segment on each, so the
//!   segments form disjoint closed loops; each loop is filled with a triangle fan.
//! * One vertex is created per crossed grid edge and shared by the four cells around it.
//!
//! * Fan diagonals never join two crossings on a common face of the cell (see `Tables`), so
//!   no other cell can contain the same diagonal.
//!
//! Hence every mesh edge is either private to one cell and used by two triangles of one fan,
//! or a face segment used by one triangle in each of the two cells sharing the face: no
//! boundary and no non-manifold edge can arise away from the border of the grid, whatever
//! the field. Segments are directed with the inside to their left as seen from outside the
//! cell, so the two cells traverse a shared segment in opposite directions and the mesh is
//! consistently oriented; triangles are wound so that normals point from negative to
//! positive values. The tests check these properties exhaustively for all 256 corner
//! configurations and all decisions.
//!
//! Compared with Lewiner's tables this omits the tests for tunnels through the interior of a
//! cell, so the genus can differ in rare cells; positions of the vertices are the same linear
//! interpolation along grid edges.

use super::grid::Dims;

/// Corner `c` of the unit cell is at `(c & 1, c >> 1 & 1, c >> 2 & 1)`.
/// Edge `4 * axis + k` runs along `axis` from the corner whose other two coordinates are
/// `k & 1` (axis + 1) and `k >> 1` (axis + 2).
fn edge_between(a: usize, b: usize) -> usize {
    let axis = (a ^ b).trailing_zeros() as usize;
    debug_assert_eq!((a ^ b).count_ones(), 1);
    let base = a & b;
    4 * axis + (base >> ((axis + 1) % 3) & 1) + 2 * (base >> ((axis + 2) % 3) & 1)
}

/// Offsets of the lower end of an edge, and its axis.
fn edge_base(edge: usize) -> ([usize; 3], usize) {
    let axis = edge / 4;
    let mut offset = [0; 3];
    offset[(axis + 1) % 3] = edge & 1;
    offset[(axis + 2) % 3] = edge >> 1 & 1;
    (offset, axis)
}

/// The corners of face `2 * axis + side`, counter-clockwise as seen from outside the cell.
fn face_corners(face: usize) -> [usize; 4] {
    let (axis, side) = (face / 2, face & 1);
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let order = if side == 1 { [(0, 0), (1, 0), (1, 1), (0, 1)] } else { [(0, 0), (0, 1), (1, 1), (1, 0)] };
    order.map(|(a, b)| side << axis | a << u | b << v)
}

/// Directed contour segments (from edge, to edge) on one face. `inside` is the corner bit
/// mask; `joined` connects the two inside corners of an ambiguous face through its middle.
fn face_segments(face: usize, inside: usize, joined: bool) -> Vec<(usize, usize)> {
    let corners = face_corners(face);
    let is_in = |k: usize| inside >> corners[k % 4] & 1 == 1;
    let edge = |k: usize| edge_between(corners[k % 4], corners[(k + 1) % 4]);
    // Walking the face boundary counter-clockwise, a contour starts where the walk leaves the
    // inside region and ends where it enters it again.
    let leaving: Vec<usize> = (0..4).filter(|&k| is_in(k) && !is_in(k + 1)).collect();
    match leaving.len() {
        0 => Vec::new(),
        1 => {
            let entering = (0..4).find(|&k| !is_in(k) && is_in(k + 1)).unwrap();
            vec![(edge(leaving[0]), edge(entering))]
        }
        // Inside corners k and k + 2: cut each corner off, or run around both.
        _ => leaving.iter().map(|&k| (edge(k), if joined { edge(k + 1) } else { edge(k + 3) })).collect(),
    }
}

/// The two faces of the cell an edge lies on.
fn edge_faces(edge: usize) -> [usize; 2] {
    let axis = edge / 4;
    [2 * ((axis + 1) % 3) + (edge & 1), 2 * ((axis + 2) % 3) + (edge >> 1 & 1)]
}

fn share_a_face(a: usize, b: usize) -> bool {
    edge_faces(a).iter().any(|face| edge_faces(b).contains(face))
}

/// First vertex code that names the centre of a ring instead of an edge crossing.
const CENTRE: u8 = 12;

/// Triangles for every corner configuration and every choice on its ambiguous faces.
///
/// A ring of crossings is filled by a fan from one of its vertices, chosen so that no
/// diagonal joins two crossings on a common face of the cell: such a chord lies in that
/// face, and the neighbouring cell may contain the same chord, which would make the edge
/// non-manifold. Where every fan has such a chord the ring is fanned from an extra vertex
/// at its centre instead. Diagonals and centre edges are therefore private to their cell.
pub struct Tables {
    /// Start of the triangles of `configuration * 64 + decisions` in `triangles`.
    start: Vec<u32>,
    /// Vertex codes: below `CENTRE` an edge of the cell, else `CENTRE + n` for the centre
    /// of the entry's n-th centred ring. Triangles are wound against the direction of the
    /// face segments, so that normals point from negative to positive values.
    triangles: Vec<[u8; 3]>,
    /// Start of the centred rings of an entry in `rings`.
    ring_start: Vec<u32>,
    rings: Vec<Vec<u8>>,
    /// Bit `f` set: face `f` of this configuration is ambiguous.
    ambiguous: [u8; 256],
}

impl Tables {
    pub fn new() -> Self {
        let mut tables = Tables {
            start: Vec::with_capacity(256 * 64 + 1),
            triangles: Vec::new(),
            ring_start: Vec::with_capacity(256 * 64 + 1),
            rings: Vec::new(),
            ambiguous: [0; 256],
        };
        for inside in 0..256usize {
            for face in 0..6 {
                let corners = face_corners(face);
                let bits: Vec<usize> = corners.iter().map(|&c| inside >> c & 1).collect();
                if bits[0] == bits[2] && bits[1] == bits[3] && bits[0] != bits[1] {
                    tables.ambiguous[inside] |= 1 << face;
                }
            }
            for decisions in 0..64usize {
                tables.start.push(tables.triangles.len() as u32);
                tables.ring_start.push(tables.rings.len() as u32);
                if decisions & !usize::from(tables.ambiguous[inside]) != 0 {
                    continue; // never looked up
                }
                let mut next = [usize::MAX; 12];
                for face in 0..6 {
                    for (from, to) in face_segments(face, inside, decisions >> face & 1 == 1) {
                        debug_assert_eq!(next[from], usize::MAX);
                        next[from] = to;
                    }
                }
                let mut seen = [false; 12];
                let mut centred = 0u8;
                for first in 0..12 {
                    if next[first] == usize::MAX || seen[first] {
                        continue;
                    }
                    let mut ring = Vec::new();
                    let mut edge = first;
                    while !seen[edge] {
                        seen[edge] = true;
                        ring.push(edge as u8);
                        edge = next[edge];
                    }
                    let n = ring.len();
                    let at = |i: usize| ring[i % n];
                    let fan_is_private = |apex: usize| (2..n - 1).all(|i| !share_a_face(usize::from(at(apex)), usize::from(at(apex + i))));
                    if let Some(apex) = (0..n).find(|&apex| fan_is_private(apex)) {
                        for i in 1..n - 1 {
                            tables.triangles.push([at(apex), at(apex + i + 1), at(apex + i)]);
                        }
                    } else {
                        for i in 0..n {
                            tables.triangles.push([CENTRE + centred, at(i + 1), at(i)]);
                        }
                        tables.rings.push(ring);
                        centred += 1;
                    }
                }
            }
        }
        tables.start.push(tables.triangles.len() as u32);
        tables.ring_start.push(tables.rings.len() as u32);
        tables
    }

    fn triangles(&self, inside: usize, decisions: usize) -> &[[u8; 3]] {
        let entry = inside * 64 + decisions;
        &self.triangles[self.start[entry] as usize..self.start[entry + 1] as usize]
    }

    fn centred_rings(&self, inside: usize, decisions: usize) -> &[Vec<u8>] {
        let entry = inside * 64 + decisions;
        &self.rings[self.ring_start[entry] as usize..self.ring_start[entry + 1] as usize]
    }
}

impl Default for Tables {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether the inside corners of an ambiguous face are connected: the bilinear interpolant
/// is negative at its saddle point. `values` are the face's corners in cyclic order.
#[inline]
fn saddle_is_inside(values: [f64; 4]) -> bool {
    let (even, odd) = (values[0] * values[2], values[1] * values[3]);
    if values[0] < 0.0 {
        even > odd
    } else {
        odd > even
    }
}

/// An indexed triangle mesh in grid coordinates (voxel centres at integers).
#[derive(Debug, Default)]
pub struct GridMesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

const NONE: u32 = u32::MAX;

/// Vertices of the grid edges around the slab of cells being processed, so that the four
/// cells around an edge share one vertex.
struct Slab {
    step: usize,
    /// Samples per layer along z (the row stride of the maps below).
    mz: usize,
    /// Vertex of the edge starting at sample (y, z): along x between the two layers of the
    /// slab, along y and along z within the lower `[0]` and the upper `[1]` layer.
    along_x: Vec<u32>,
    along_y: [Vec<u32>; 2],
    along_z: [Vec<u32>; 2],
    mesh: GridMesh,
}

impl Slab {
    /// The vertex where the surface crosses `edge` of cell `(xi, yi, zi)`, created on first use.
    fn vertex(&mut self, edge: usize, cell: [usize; 3], values: &[f64; 8]) -> u32 {
        let (offset, axis) = edge_base(edge);
        let [_, yi, zi] = cell;
        let slot = match axis {
            0 => &mut self.along_x[(yi + offset[1]) * self.mz + zi + offset[2]],
            1 => &mut self.along_y[offset[0]][yi * self.mz + zi + offset[2]],
            _ => &mut self.along_z[offset[0]][(yi + offset[1]) * self.mz + zi],
        };
        if *slot == NONE {
            let low = offset[0] | offset[1] << 1 | offset[2] << 2;
            let (a, b) = (values[low], values[low | 1 << axis]);
            let mut point: [f64; 3] = std::array::from_fn(|k| ((cell[k] + offset[k]) * self.step) as f64);
            point[axis] += self.step as f64 * (a / (a - b));
            *slot = self.mesh.vertices.len() as u32;
            self.mesh.vertices.push(point.map(|p| p as f32));
        }
        *slot
    }

    fn next_layer(&mut self) {
        self.along_x.fill(NONE);
        self.along_y.swap(0, 1);
        self.along_z.swap(0, 1);
        self.along_y[1].fill(NONE);
        self.along_z[1].fill(NONE);
    }
}

/// The surface `value = 0` with `value < 0` inside. Cells span `step` voxels and start at
/// multiples of `step` below `n - step` on each axis, like scikit-image's `step_size`.
pub fn extract(value: &[f32], dims: Dims, step: usize) -> GridMesh {
    assert!(step >= 1 && value.len() == dims[0] * dims[1] * dims[2]);
    let tables = Tables::new();
    let cells = dims.map(|n| if n > step { (n - step).div_ceil(step) } else { 0 });
    if cells.contains(&0) {
        return GridMesh::default();
    }
    let [_, ny, nz] = dims;
    let (my, mz) = (cells[1] + 1, cells[2] + 1);
    let map = || vec![NONE; my * mz];
    let mut slab = Slab { step, mz, along_x: map(), along_y: [map(), map()], along_z: [map(), map()], mesh: GridMesh::default() };
    let corners_of_face: [[usize; 4]; 6] = std::array::from_fn(face_corners);
    for xi in 0..cells[0] {
        for yi in 0..cells[1] {
            let row = |dx: usize, dy: usize| {
                let start = ((xi + dx) * step * ny + (yi + dy) * step) * nz;
                &value[start..start + nz]
            };
            let rows = [row(0, 0), row(1, 0), row(0, 1), row(1, 1)];
            let column = |z: usize| rows.iter().enumerate().fold(0usize, |bits, (c, r)| bits | usize::from(r[z] < 0.0) << c);
            let mut upper = column(0);
            for zi in 0..cells[2] {
                let lower = upper;
                upper = column((zi + 1) * step);
                let inside = lower | upper << 4;
                if inside == 0 || inside == 255 {
                    continue;
                }
                let values: [f64; 8] = std::array::from_fn(|c| f64::from(rows[c & 3][(zi + (c >> 2)) * step]));
                let mut decisions = 0;
                let ambiguous = tables.ambiguous[inside];
                if ambiguous != 0 {
                    for (face, corners) in corners_of_face.iter().enumerate() {
                        if ambiguous >> face & 1 == 1 && saddle_is_inside(corners.map(|c| values[c])) {
                            decisions |= 1 << face;
                        }
                    }
                }
                let cell = [xi, yi, zi];
                let mut local = [NONE; 16];
                for (n, ring) in tables.centred_rings(inside, decisions).iter().enumerate() {
                    let mut centre = [0f64; 3];
                    for &edge in ring {
                        let vertex = slab.vertex(usize::from(edge), cell, &values);
                        local[usize::from(edge)] = vertex;
                        let point = slab.mesh.vertices[vertex as usize];
                        centre.iter_mut().zip(point).for_each(|(sum, p)| *sum += f64::from(p));
                    }
                    local[usize::from(CENTRE) + n] = slab.mesh.vertices.len() as u32;
                    slab.mesh.vertices.push(centre.map(|sum| (sum / ring.len() as f64) as f32));
                }
                for triangle in tables.triangles(inside, decisions) {
                    let face = triangle.map(|code| {
                        let code = usize::from(code);
                        if local[code] == NONE {
                            local[code] = slab.vertex(code, cell, &values);
                        }
                        local[code]
                    });
                    slab.mesh.faces.push(face);
                }
            }
        }
        slab.next_layer();
    }
    slab.mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Corner positions of a face in the frame of the neighbouring cell across it.
    fn across(face: usize, corner: usize) -> usize {
        corner ^ 1 << (face / 2)
    }

    #[test]
    fn edges_and_faces_are_consistent() {
        let mut seen = [false; 12];
        for a in 0..8usize {
            for axis in 0..3 {
                let b = a ^ 1 << axis;
                let edge = edge_between(a, b);
                assert_eq!(edge, edge_between(b, a));
                let (offset, edge_axis) = edge_base(edge);
                assert_eq!(edge_axis, axis);
                assert_eq!(offset[0] | offset[1] << 1 | offset[2] << 2, a & b);
                seen[edge] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
        for face in 0..6 {
            let corners = face_corners(face);
            let point = |c: usize| [(c & 1) as i32, (c >> 1 & 1) as i32, (c >> 2 & 1) as i32];
            let (p, q, r) = (point(corners[0]), point(corners[1]), point(corners[3]));
            let (u, v) = ([q[0] - p[0], q[1] - p[1], q[2] - p[2]], [r[0] - p[0], r[1] - p[1], r[2] - p[2]]);
            let normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            let mut outward = [0; 3];
            outward[face / 2] = if face & 1 == 1 { 1 } else { -1 };
            assert_eq!(normal, outward, "face {face} is not counter-clockwise from outside");
        }
    }

    /// The property that makes the mesh closed: what a cell draws on a face is the reverse of
    /// what its neighbour draws there, for every configuration of the shared corners.
    #[test]
    fn neighbouring_cells_draw_the_same_segments_reversed() {
        for face in 0..6 {
            let opposite = face ^ 1;
            for inside in 0..256usize {
                // The neighbour sees the shared corners on its opposite face.
                let mut neighbour = 0usize;
                for corner in face_corners(face) {
                    neighbour |= (inside >> corner & 1) << across(face, corner);
                }
                for joined in [false, true] {
                    let mut ours: Vec<(usize, usize)> = face_segments(face, inside, joined);
                    // Edge numbers of the shared edges, expressed in the neighbour's frame.
                    let mut theirs: Vec<(usize, usize)> = face_segments(opposite, neighbour, joined)
                        .into_iter()
                        .map(|(from, to)| {
                            let map = |edge: usize| {
                                let (offset, axis) = edge_base(edge);
                                let low = offset[0] | offset[1] << 1 | offset[2] << 2;
                                edge_between(across(face, low), across(face, low | 1 << axis))
                            };
                            (map(to), map(from))
                        })
                        .collect();
                    ours.sort_unstable();
                    theirs.sort_unstable();
                    assert_eq!(ours, theirs, "face {face}, corners {inside:08b}, joined {joined}");
                }
            }
        }
    }

    /// Within a cell: fan diagonals are used twice in opposite directions, every other
    /// triangle edge is one of the face segments, each exactly once and in its direction.
    #[test]
    fn every_table_entry_is_a_set_of_disks_bounded_by_the_face_segments() {
        let tables = Tables::new();
        let (mut most, mut entries, mut with_centres) = (0, 0, 0);
        let mut unambiguous_with_centres = Vec::new();
        for inside in 0..256usize {
            let ambiguous = usize::from(tables.ambiguous[inside]);
            for decisions in (0..64usize).filter(|d| d & !ambiguous == 0) {
                let triangles = tables.triangles(inside, decisions);
                most = most.max(triangles.len());
                let centred = tables.centred_rings(inside, decisions);
                assert!(triangles.iter().flatten().all(|&code| usize::from(code) < 12 + centred.len()));
                let mut directed: HashMap<(u8, u8), i32> = HashMap::new();
                for t in triangles {
                    for k in 0..3 {
                        *directed.entry((t[k], t[(k + 1) % 3])).or_default() += 1;
                    }
                }
                assert!(directed.values().all(|&n| n == 1), "an edge is traversed twice in one direction");
                // Triangles run against the segments, so reverse the boundary edges.
                let mut boundary: Vec<(usize, usize)> = directed
                    .keys()
                    .filter(|(a, b)| !directed.contains_key(&(*b, *a)))
                    .map(|&(a, b)| (usize::from(b), usize::from(a)))
                    .collect();
                // Interior edges are private to the cell: they end at a ring centre or join
                // crossings that no single face contains.
                for &(a, b) in directed.keys().filter(|(a, b)| directed.contains_key(&(*b, *a))) {
                    assert!(a >= CENTRE || b >= CENTRE || !share_a_face(usize::from(a), usize::from(b)), "chord {a}-{b} lies in a face");
                }
                if !centred.is_empty() {
                    with_centres += 1;
                    if ambiguous == 0 {
                        unambiguous_with_centres.push(inside);
                    }
                }
                entries += 1;
                let mut segments: Vec<(usize, usize)> =
                    (0..6).flat_map(|face| face_segments(face, inside, decisions >> face & 1 == 1)).collect();
                boundary.sort_unstable();
                segments.sort_unstable();
                assert_eq!(boundary, segments, "corners {inside:08b}, decisions {decisions:06b}");
                // Each crossed edge of the cell is used, and only those.
                for edge in 0..12 {
                    let (offset, axis) = edge_base(edge);
                    let low = offset[0] | offset[1] << 1 | offset[2] << 2;
                    let crossed = (inside >> low & 1) != (inside >> (low | 1 << axis) & 1);
                    assert_eq!(crossed, triangles.iter().flatten().any(|&e| usize::from(e) == edge));
                }
            }
            assert!(inside != 0 && inside != 255 || tables.triangles(inside, 0).is_empty());
        }
        assert!(most <= 12);
        // Extra vertices occur only in cells with an ambiguous face (140 of the 656 entries).
        assert!(unambiguous_with_centres.is_empty(), "{unambiguous_with_centres:?}");
        assert_eq!((entries, with_centres), (656, 140));
    }

    #[test]
    fn the_decider_follows_the_saddle_value() {
        // Corners in cyclic order; bilinear saddle value is (ac - bd) / (a + c - b - d).
        for values in [[-1.0, 0.5, -1.0, 0.5], [-0.2, 1.0, -0.3, 2.0], [1.0, -0.5, 1.0, -0.5], [3.0, -0.1, 0.2, -0.4]] {
            let [a, b, c, d] = values;
            let saddle: f64 = (a * c - b * d) / (a + c - b - d);
            assert_eq!(saddle_is_inside(values), saddle < 0.0, "{values:?}");
            assert_eq!(saddle_is_inside([b, c, d, a]), saddle < 0.0, "rotated {values:?}");
            assert_eq!(saddle_is_inside([d, c, b, a]), saddle < 0.0, "mirrored {values:?}");
        }
    }

    pub(crate) fn edge_counts(faces: &[[u32; 3]]) -> (usize, usize, usize) {
        let mut undirected: HashMap<(u32, u32), (u32, i32)> = HashMap::new();
        for f in faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                let entry = undirected.entry((a.min(b), a.max(b))).or_default();
                entry.0 += 1;
                entry.1 += if a < b { 1 } else { -1 };
            }
        }
        let boundary = undirected.values().filter(|e| e.0 == 1).count();
        let nonmanifold = undirected.values().filter(|e| e.0 > 2).count();
        let misoriented = undirected.values().filter(|e| e.0 == 2 && e.1 != 0).count();
        (boundary, nonmanifold, misoriented)
    }

    /// Deterministic noise in [-1, 1): a field with every kind of ambiguous cell.
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn random_fields_give_closed_oriented_manifolds() {
        for (seed, step) in [(1, 1), (2, 1), (3, 2), (4, 3)] {
            let dims = [22, 19, 20];
            let mut value = noise(dims[0] * dims[1] * dims[2], seed);
            // Positive on the border, so that no surface leaves the grid.
            for x in 0..dims[0] {
                for y in 0..dims[1] {
                    for z in 0..dims[2] {
                        let border = [x, y, z].iter().zip(&dims).any(|(&p, &n)| p < step || p + step + 1 > n - 1);
                        let i = (x * dims[1] + y) * dims[2] + z;
                        if border {
                            value[i] = 1.0;
                        } else if seed == 2 && i % 7 == 0 {
                            value[i] = 0.0; // exact zeros put vertices on grid points
                        }
                    }
                }
            }
            let mesh = extract(&value, dims, step);
            assert!(mesh.faces.len() > 100);
            assert_eq!(edge_counts(&mesh.faces), (0, 0, 0), "seed {seed}, step {step}");
            let used: std::collections::HashSet<u32> = mesh.faces.iter().flatten().copied().collect();
            assert_eq!(used.len(), mesh.vertices.len());
        }
    }

    #[test]
    fn a_sphere_has_outward_normals_and_the_right_size() {
        let n = 24;
        let centre = 11.3;
        let mut value = Vec::new();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let r = [x, y, z].iter().map(|&p| (p as f64 - centre) * (p as f64 - centre)).sum::<f64>().sqrt();
                    value.push((r - 8.0) as f32);
                }
            }
        }
        let mesh = extract(&value, [n, n, n], 1);
        assert_eq!(edge_counts(&mesh.faces), (0, 0, 0));
        assert_eq!(mesh.vertices.len() as i64 - mesh.faces.len() as i64 / 2, 2, "Euler characteristic of a sphere");
        let mut volume = 0.0;
        for f in &mesh.faces {
            let [a, b, c] = f.map(|i| mesh.vertices[i as usize].map(f64::from));
            volume += a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        volume /= 6.0;
        let exact = 4.0 / 3.0 * std::f64::consts::PI * 512.0;
        assert!(volume > 0.0 && (volume - exact).abs() < 0.02 * exact, "volume {volume}, exact {exact}");
        for v in &mesh.vertices {
            let r = v.iter().map(|&p| (f64::from(p) - centre) * (f64::from(p) - centre)).sum::<f64>().sqrt();
            assert!((r - 8.0).abs() < 0.05);
        }
        assert!(extract(&value, [n, n, n], 30).faces.is_empty());
    }
}
