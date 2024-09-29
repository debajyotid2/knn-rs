# knn.rs

An implementation of classification using [k-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) in [Rust](https://www.rust-lang.org/).

## Algorithm

```
1. Load "training" data ($N$ labeled data points $x_i$, $y_i$).
2. Initialize $k$, the number of neighbors to consider.
3. For a given "test" data point $x_j$, to find its predicted label $y_j$, for $i = 1..N$,   
    3.a. find distance $d(x_i, x_j)$.
    3.b. store distance $d(x_{i}, x_{j})$ in distances array $D_{j}$.
4. Sort $D_j$ by distance values.
5. Get top $k$ rows from sorted array.
6. Find the class that is the most frequent of these rows. This is $y_{j}$.
```

## Example usage

From [example.rs](./examples/example.rs)
```rust
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
```

## How to build

Rust (v1.81 and above) is used to build and run the sources. To pull all dependencies, from the project root directory, run

```
cargo fetch
```
Then to build, run
```
cargo build
```
To run tests, run
```
cargo test
```

## License

[MIT](https://mit-license.org/)
