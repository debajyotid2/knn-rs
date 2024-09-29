use ndarray::{array, Array1, Array2};
use knn_rs::{knn_classifier, l2_norm_sq};

fn main() {
    let mat_x: Array2<f32> = array![
       [-1.0, -1.0],
       [-2.0, -1.0],
       [-3.0, -2.0],
       [1.0, 1.0],
       [2.0, 1.0],
       [3.0, 2.0]
    ];
    let mat_y: Array1<u32> = array![0, 0, 0, 1, 1, 1];
    let k = 2usize;
    
    let x_i: Array1<f32> = array![0.5, 0.0];
    
    let predicted = knn_classifier(&mat_x, &mat_y, &x_i, k, l2_norm_sq);
 
    println!("Predicted label: {}", &predicted);
}
