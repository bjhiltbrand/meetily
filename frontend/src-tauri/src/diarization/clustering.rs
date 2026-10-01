//! Agglomerative (average-linkage) clustering of speaker embeddings
//! using cosine distance.

/// Default cosine-distance threshold for "auto" speaker count.
/// Larger = fewer speakers. Tuned for the WeSpeaker ResNet34 model.
pub const DEFAULT_THRESHOLD: f32 = 0.7;

/// Clusters whose total speech is shorter than this are merged into the
/// nearest larger cluster when the speaker count is automatic.
pub const MIN_SPEAKER_SECONDS: f64 = 1.5;

pub enum ClusterTarget {
    /// Stop merging when the closest pair is farther apart than this.
    Threshold(f32),
    /// Merge until exactly this many clusters remain.
    NumSpeakers(usize),
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

/// Cluster L2-normalized embeddings. Returns one label per embedding,
/// numbered 0.. in order of first appearance.
pub fn cluster(embeddings: &[Vec<f32>], target: &ClusterTarget) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Full distance matrix (n is the number of transcript segments: hundreds to
    // a few thousand), updated with the Lance-Williams average-linkage formula.
    let mut dist = vec![f32::INFINITY; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = cosine_distance(&embeddings[i], &embeddings[j]);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    let mut sizes = vec![1usize; n];
    let mut active: Vec<usize> = (0..n).collect();
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while active.len() > 1 {
        if let ClusterTarget::NumSpeakers(k) = target {
            if active.len() <= (*k).max(1) {
                break;
            }
        }

        // Find the closest pair among active clusters
        let mut best = (0usize, 0usize, f32::INFINITY);
        for (ai, &i) in active.iter().enumerate() {
            for &j in &active[ai + 1..] {
                let d = dist[i * n + j];
                if d < best.2 {
                    best = (i, j, d);
                }
            }
        }
        let (i, j, d) = best;

        if let ClusterTarget::Threshold(t) = target {
            if d > *t {
                break;
            }
        }

        // Merge j into i
        for &m in &active {
            if m == i || m == j {
                continue;
            }
            let merged = (dist[i * n + m] * sizes[i] as f32 + dist[j * n + m] * sizes[j] as f32)
                / (sizes[i] + sizes[j]) as f32;
            dist[i * n + m] = merged;
            dist[m * n + i] = merged;
        }
        sizes[i] += sizes[j];
        let moved = std::mem::take(&mut members[j]);
        members[i].extend(moved);
        active.retain(|&c| c != j);
    }

    relabel(&active.iter().map(|&c| members[c].clone()).collect::<Vec<_>>(), n)
}

/// Turn cluster membership lists into labels ordered by first appearance.
fn relabel(clusters: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut order: Vec<&Vec<usize>> = clusters.iter().filter(|c| !c.is_empty()).collect();
    order.sort_by_key(|c| *c.iter().min().unwrap());
    let mut labels = vec![0usize; n];
    for (label, c) in order.iter().enumerate() {
        for &m in c.iter() {
            labels[m] = label;
        }
    }
    labels
}

/// Merge clusters with very little total speech into the most similar
/// remaining cluster. `durations` are per-embedding speech durations (s).
pub fn merge_small_clusters(
    embeddings: &[Vec<f32>],
    labels: &[usize],
    durations: &[f64],
    min_seconds: f64,
) -> Vec<usize> {
    let n = labels.len();
    let num_clusters = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if num_clusters <= 1 {
        return labels.to_vec();
    }

    let dim = embeddings[0].len();
    let mut centroids = vec![vec![0.0f32; dim]; num_clusters];
    let mut totals = vec![0.0f64; num_clusters];
    for (idx, &l) in labels.iter().enumerate() {
        for (c, v) in centroids[l].iter_mut().zip(&embeddings[idx]) {
            *c += *v;
        }
        totals[l] += durations[idx];
    }
    centroids.iter_mut().for_each(|c| l2_normalize(c));

    let big: Vec<usize> = (0..num_clusters).filter(|&c| totals[c] >= min_seconds).collect();
    if big.is_empty() {
        return labels.to_vec();
    }

    let mut remap: Vec<usize> = (0..num_clusters).collect();
    for c in 0..num_clusters {
        if totals[c] < min_seconds {
            remap[c] = *big
                .iter()
                .min_by(|&&a, &&b| {
                    cosine_distance(&centroids[c], &centroids[a])
                        .partial_cmp(&cosine_distance(&centroids[c], &centroids[b]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
        }
    }

    // Rebuild membership and relabel by first appearance
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];
    for (idx, &l) in labels.iter().enumerate() {
        clusters[remap[l]].push(idx);
    }
    relabel(&clusters, n)
}

pub fn normalize(v: &mut [f32]) {
    l2_normalize(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(v: &[f32]) -> Vec<f32> {
        let mut e = v.to_vec();
        l2_normalize(&mut e);
        e
    }

    #[test]
    fn separates_two_clear_groups() {
        let e = vec![
            emb(&[1.0, 0.0, 0.1]),
            emb(&[0.0, 1.0, 0.1]),
            emb(&[0.95, 0.05, 0.0]),
            emb(&[0.05, 0.9, 0.0]),
        ];
        assert_eq!(cluster(&e, &ClusterTarget::Threshold(0.5)), vec![0, 1, 0, 1]);
        assert_eq!(cluster(&e, &ClusterTarget::NumSpeakers(2)), vec![0, 1, 0, 1]);
        assert_eq!(cluster(&e, &ClusterTarget::NumSpeakers(1)), vec![0, 0, 0, 0]);
    }

    #[test]
    fn merges_tiny_cluster() {
        let e = vec![emb(&[1.0, 0.0]), emb(&[0.0, 1.0]), emb(&[0.9, 0.3])];
        let labels = vec![0, 1, 2];
        let merged = merge_small_clusters(&e, &labels, &[10.0, 10.0, 0.5], 3.0);
        assert_eq!(merged, vec![0, 1, 0]);
    }
}
