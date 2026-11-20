//! Task 479's crossover: a fresh Cholesky factorization of a `k x k`
//! matrix against moving a kept factor by one rank-one step
//! (`SpdFactor::updated`, LINPACK's `dchud`), at each order `k`, so the
//! order from which the move is the cheaper can be read on each platform.
//!
//!     cargo run --release -p rheon-core --example factor_crossover -- [rounds]
//!
//! Per order it prints, in ns a call, the best over `rounds` (default 7) of:
//! `fresh`, `SpdFactor::of` of the matrix (which makes a new `k x k` matrix
//! for its factor); `alloc`, making and dropping that matrix alone, so
//! `fresh - alloc` is what a factorization written over a kept factor costs,
//! as the models' working spaces do; `move`, one `updated(c, v)` step, the
//! fresh factorization a model would take every `SpdFactor::MAX_MOVES`
//! moves included at its share; and `move / (fresh - alloc)`, under 1 where
//! the move is the cheaper. The matrix is `c·A + v vᵀ` of a well conditioned
//! `A`, as an exponentially weighted co-moment matrix moves.
use rheon_core::SpdFactor;
use std::hint::black_box;
use std::time::Instant;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// `I + Σ x xᵀ / n` over `n = 4k` rows of `[-1, 1)` noise: positive
/// definite, condition near 1 to 10.
fn matrix(k: usize, s: &mut u64) -> Vec<f64> {
    let n = 4 * k;
    let mut a = vec![0.0; k * k];
    for _ in 0..n {
        let x: Vec<f64> = (0..k).map(|_| lcg(s)).collect();
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] += x[i] * x[j] / n as f64;
            }
        }
    }
    for i in 0..k {
        a[i * k + i] += 1.0;
    }
    a
}

fn best_of<F: FnMut() -> f64>(rounds: usize, mut f: F) -> f64 {
    (0..rounds).map(|_| f()).fold(f64::INFINITY, f64::min)
}

fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(7);
    println!(
        "{:>4} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "k", "fresh", "alloc", "fresh-alloc", "move", "move/(f-a)"
    );
    for k in [2usize, 3, 4, 5, 6, 7, 8, 10, 12, 13, 16, 20, 24, 32, 48, 64] {
        let mut s = 7 + k as u64;
        let a = matrix(k, &mut s);
        let iters = (2_000_000 / (k * k * k)).clamp(2_000, 400_000);
        let fresh = best_of(rounds, || {
            let t = Instant::now();
            for _ in 0..iters {
                black_box(SpdFactor::of(black_box(&a), k));
            }
            t.elapsed().as_secs_f64() / iters as f64 * 1e9
        });
        let alloc = best_of(rounds, || {
            let t = Instant::now();
            for _ in 0..iters {
                black_box(faer::Mat::<f64>::zeros(black_box(k), k));
            }
            t.elapsed().as_secs_f64() / iters as f64 * 1e9
        });
        // Steps of the size an exponentially weighted update takes: the
        // old matrix decayed by `c`, a new row's outer product added at
        // weight `1 - c`.
        let steps: Vec<f64> = (0..64 * k).map(|_| lcg(&mut s)).collect();
        let c = 0.999f64;
        let w = (1.0 - c).sqrt();
        let mv = best_of(rounds, || {
            let mut f = SpdFactor::of(&a, k).expect("positive definite");
            let mut v = vec![0.0; k];
            let t = Instant::now();
            for it in 0..iters {
                let at = (it % 64) * k;
                for (vi, si) in v.iter_mut().zip(&steps[at..at + k]) {
                    *vi = w * si;
                }
                f = if f.can_move() {
                    match f.updated(black_box(c), &mut v) {
                        Some(f) => f,
                        None => SpdFactor::of(&a, k).expect("positive definite"),
                    }
                } else {
                    // At the cap, a model factorizes afresh: here the
                    // starting matrix, which costs what the moved one would.
                    SpdFactor::of(black_box(&a), k).expect("positive definite")
                };
            }
            let dt = t.elapsed().as_secs_f64() / iters as f64 * 1e9;
            black_box(&f);
            dt
        });
        let net = fresh - alloc;
        println!(
            "{k:>4} {fresh:>10.1} {alloc:>10.1} {net:>10.1} {mv:>10.1} {:>12.3}",
            mv / net
        );
    }
}
