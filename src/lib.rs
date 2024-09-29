use std::iter::zip;
use std::collections::HashMap;
use ndarray::{
    Array, Array1, ArrayView1, Axis, Dim, Dimension, RemoveAxis
};

/// Squared L1 norm between two points
pub fn l1_norm_sq(point1: &ArrayView1<f32>, point2: &ArrayView1<f32>) -> f32 {
    let mut dist = 0.0f32;
    for (val1, val2) in zip(point1.iter(), point2.iter()) {
        dist += val1 - val2;
    }
    dist 
}

/// Squared L2 norm between two points
pub fn l2_norm_sq(point1: &ArrayView1<f32>, point2: &ArrayView1<f32>) -> f32 {
    let mut dist = 0.0f32;
    for (val1, val2) in zip(point1.iter(), point2.iter()) {
        let diff = val1 - val2;
        dist +=  diff * diff;
    }
    dist 
}

// k-Nearest Neighbors classifier
pub fn knn_classifier<D>(
                        mat_x: &Array<f32, D>, 
                        mat_y: &Array1<u32>, 
                        x_test: &Array1<f32>, 
                        num_neighbors: usize,
                        distance_metric: impl Fn(&ArrayView1<f32>, &ArrayView1<f32>) -> f32) -> u32
    where
        D: Dimension<Smaller = Dim<[usize; 1]>> + RemoveAxis {
    let mut distances = Vec::<(usize, f32, u32)>::new();

    // Calculate distances
    for (j, x_j) in mat_x.axis_iter(Axis(0)).enumerate() {
        distances.push((j, distance_metric(&x_test.view(), &x_j.view()), mat_y[j]));
    }

    // Sort distances
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    // Find most common class among top k-labels
    let mut freqs = HashMap::<u32, u32>::new();
    for (_,  _, label) in distances.iter().take(num_neighbors) {
        let count = freqs.entry(*label).or_insert(0u32);
        *count += 1;
    }
    let mut sorted_freqs: Vec<(&u32, &u32)> = freqs
                                        .iter()
                                        .collect();
    sorted_freqs.sort_by(|a, b| b.1.cmp(&a.1));
    *sorted_freqs[0].0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};

    #[test]
    fn test_l1_norm_sq() {
        let row_x: Array1<f32> = array![-2.0, -2.0];
        let row_y: Array1<f32> = array![3.0, 1.0];
 
        assert_eq!(l1_norm_sq(&row_x.view(), &row_y.view()), 8.0f32);
    }

    #[test]
    fn test_l2_norm_sq() {
        let row_x: Array1<f32> = array![-2.0, -2.0];
        let row_y: Array1<f32> = array![3.0, 1.0];
 
        assert_eq!(l2_norm_sq(&row_x.view(), &row_y.view()), 34.0f32);
    }
}
