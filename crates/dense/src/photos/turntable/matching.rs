//! Matches between the features of two photos: nearest neighbour by
//! descriptor distance in both directions, each passing the ratio test, and
//! both agreeing.

use super::features::{Features, DESCRIPTOR};

/// The photo pairs to match: every photo with its next `span` photos, around
/// the closed turn unless the turn is open.
pub fn ring_pairs(count: usize, span: usize, open: bool) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..count {
        for d in 1..=span.min(count.saturating_sub(1)) {
            if open && i + d >= count {
                continue;
            }
            let j = (i + d) % count;
            if !pairs.contains(&(i, j)) && !pairs.contains(&(j, i)) {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// `(feature in a, feature in b)` for mutual nearest neighbours whose distance is below
/// `ratio` times that of the second nearest, in both directions.
pub fn match_pair(a: &Features, b: &Features, ratio: f32) -> Vec<(u32, u32)> {
    let (na, nb) = (a.points.len(), b.points.len());
    if na < 2 || nb < 2 {
        return Vec::new();
    }
    // Best and second best of every feature of `a` (rows) and of `b` (columns) in one pass.
    let mut row_best = vec![(f32::MAX, f32::MAX, 0u32); na];
    let mut column_best = vec![(f32::MAX, f32::MAX, 0u32); nb];
    for i in 0..na {
        let da = &a.descriptors[i * DESCRIPTOR..(i + 1) * DESCRIPTOR];
        let row = &mut row_best[i];
        for j in 0..nb {
            let db = &b.descriptors[j * DESCRIPTOR..(j + 1) * DESCRIPTOR];
            let mut sum = 0f32;
            for k in 0..DESCRIPTOR {
                let d = da[k] - db[k];
                sum += d * d;
            }
            if sum < row.0 {
                *row = (sum, row.0, j as u32);
            } else if sum < row.1 {
                row.1 = sum;
            }
            let column = &mut column_best[j];
            if sum < column.0 {
                *column = (sum, column.0, i as u32);
            } else if sum < column.1 {
                column.1 = sum;
            }
        }
    }
    let squared = ratio * ratio;
    let mut out = Vec::new();
    for (i, row) in row_best.iter().enumerate() {
        let column = column_best[row.2 as usize];
        if row.0 < squared * row.1 && column.2 as usize == i && column.0 < squared * column.1 {
            out.push((i as u32, row.2));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_close_the_turn_unless_it_is_open() {
        assert_eq!(ring_pairs(5, 2, false), [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 0), (4, 0), (4, 1)]);
        assert_eq!(ring_pairs(5, 2, true), [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]);
        assert_eq!(ring_pairs(3, 4, false).len(), 3);
    }

    #[test]
    fn mutual_ratio_matches() {
        let descriptor =
            |seed: usize| -> Vec<f32> { (0..DESCRIPTOR).map(|k| (((seed * 31 + k * 17) % 97) as f32 / 97.0).powi(2)).collect() };
        let features = |seeds: &[usize]| Features {
            points: seeds.iter().map(|_| [0.0, 0.0]).collect(),
            descriptors: seeds.iter().flat_map(|&s| descriptor(s)).collect(),
        };
        // b holds a's features in another order, one missing and one stranger.
        let (a, b) = (features(&[1, 2, 3, 4, 5]), features(&[3, 1, 60, 5, 2]));
        let mut found = match_pair(&a, &b, 0.8);
        found.sort_unstable();
        assert_eq!(found, [(0, 1), (1, 4), (2, 0), (4, 3)]);
        assert!(match_pair(&a, &features(&[1]), 0.8).is_empty());
    }
}
