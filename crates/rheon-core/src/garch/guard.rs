//! EGARCH's guard, in one place: what keeps the filter that runs
//! contracting.
//!
//! EGARCH's log-variance recursion, inverted to run on the returns, steps
//! `ln s_{t+1}` by the multiplier `μ_t = β − (α |z_t| + γ z_t)/2` of `ln
//! s_t` (`∂ ln s_{t+1}/∂ ln s_t`, the factor the sensitivity recursion
//! carries). The filter forgets its start, and a small error in it dies
//! out, while the mean of `ln |μ_t|` over the rows -- the recursion's
//! Lyapunov exponent -- is below 0. The guard holds:
//!
//! - the box `α ≥ |γ|`, `α ≤ 1`, `0 ≤ β ≤ 1 − 10⁻⁴` (the set the step is
//!   projected onto, [`super::fit::egarch_set`]);
//! - the variance within `arch`'s bounds ([`super::filter`]);
//! - the mean of `ln |μ_t|` below 0, at the parameters as they stand, over
//!   the standardized residuals the filter has met: their distribution is
//!   kept as weighted counts in [`BINS`] bins of `z` over `[−8, 8]` (a
//!   residual past the edge counted in the edge bin), 8 bytes a bin, so the
//!   mean is read at any parameters in `O(BINS)` without the rows. A step
//!   that would take it to 0 or above is cut back along the step to where
//!   it is just below; a step is never refused outright, since a refusal
//!   on a mean of each past row at the parameters of its time ratchets:
//!   once that mean crosses 0, every step one way is refused and every
//!   step the other taken.
//!
//! The guard keeps the running filter contracting, and agrees with `arch`'s
//! fit wherever that fit's own filter contracts; it carries no proof that
//! the estimate it guards is consistent. (Wintenberger's (2013) stable
//! likelihood, which has one, bounds the recursion at the smallest
//! variance it can reach, and so excludes persistent processes, `β` near 1,
//! that `arch` fits and whose filter does contract.)

use super::math;

/// The bins of `z` the guard keeps.
pub(super) const BINS: usize = 64;
const Z_EDGE: f64 = 8.0;
/// The guard's bound on the mean of `ln |μ|`: just below 0.
const BOUND: f64 = -1e-6;

/// The bin a standardized residual falls in.
pub(super) fn bin(z: f64) -> usize {
    let w = 2.0 * Z_EDGE / BINS as f64;
    (((z + Z_EDGE) / w).floor().max(0.0) as usize).min(BINS - 1)
}

/// The centre of bin `b`.
fn centre(b: usize) -> f64 {
    let w = 2.0 * Z_EDGE / BINS as f64;
    -Z_EDGE + (b as f64 + 0.5) * w
}

/// `ln |β − (α |z| + γ z)/2|`, floored where `μ` is 0.
pub(super) fn log_mu(theta: &[f64], z: f64) -> f64 {
    let mu = theta[3] - 0.5 * (theta[1] * z.abs() + theta[2] * z);
    math::ln(mu.abs().max(f64::MIN_POSITIVE))
}

/// The mean of `ln |μ|` at `theta` over the residuals `counts` holds; 0 with
/// none counted.
pub(super) fn mean_log_mu(theta: &[f64], counts: &[f64]) -> f64 {
    let (mut s, mut w) = (0.0, 0.0);
    for (b, &c) in counts.iter().enumerate() {
        if c > 0.0 {
            s += c * log_mu(theta, centre(b));
            w += c;
        }
    }
    if w > 0.0 { s / w } else { 0.0 }
}

/// Whether `theta` keeps the filter contracting on `counts`.
pub(super) fn admits(theta: &[f64], counts: &[f64]) -> bool {
    mean_log_mu(theta, counts) <= BOUND
}

/// The step from `from` to `to` (both in the box) cut back, where `to`
/// breaks the guard, to the furthest point along it that keeps it: by
/// bisection, to `10⁻¹²` of the step. Where `from` breaks it too (the
/// residuals moved under it), `(α, γ, β)` of `to` are shrunk together by
/// the largest factor `t ≤ 1` that keeps it, `ω` held. `μ` is linear in
/// them, so the mean of `ln |μ|` at `t` is `ln t` plus its value at `to`
/// (bins where `μ` is 0 aside): it rises with `t`, and at `t = 0` every
/// `μ` is 0 and the mean is `ln` of the smallest normal number, far below
/// the bound. The box holds along the way. (Lowering `β` alone, as the
/// guard did, is no remedy where `μ < 0`: there it makes `|μ|` larger, and
/// `β = 0` could leave the filter more explosive than `to`; review
/// 2026-11-23, GARCH-3.)
pub(super) fn cut(from: &[f64], to: &[f64], counts: &[f64]) -> Vec<f64> {
    if admits(to, counts) {
        return to.to_vec();
    }
    let at = |t: f64| -> Vec<f64> { from.iter().zip(to).map(|(a, b)| a + t * (b - a)).collect() };
    if admits(from, counts) {
        let (mut lo, mut hi) = (0.0, 1.0);
        while hi - lo > 1e-12 {
            let mid = 0.5 * (lo + hi);
            if admits(&at(mid), counts) {
                lo = mid
            } else {
                hi = mid
            }
        }
        return at(lo);
    }
    let shrunk = |t: f64| -> Vec<f64> { vec![to[0], t * to[1], t * to[2], t * to[3]] };
    let (mut lo, mut hi) = (0.0, 1.0);
    while hi - lo > 1e-12 {
        let mid = 0.5 * (lo + hi);
        if admits(&shrunk(mid), counts) {
            lo = mid
        } else {
            hi = mid
        }
    }
    shrunk(lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mean_is_the_definition_over_the_bins_and_a_cut_keeps_it() {
        // residuals of a standard normal, binned
        let mut counts = vec![0.0; BINS];
        let mut s = 7u64;
        let mut draws = Vec::new();
        for _ in 0..20_000 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            let z = (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos();
            counts[bin(z)] += 1.0;
            draws.push(z);
        }
        // the binned mean is the mean over the draws to within the bins'
        // width: at the study's truth, where μ stays away from 0, closely;
        // near the box's edge, where μ crosses 0 and ln |μ| is singular
        // there, to a tenth
        for (th, tol) in [
            ([0.0, 0.15, -0.08, 0.97], 0.01),
            ([0.0, 0.9, -0.5, 0.2], 0.1),
        ] {
            let exact: f64 =
                draws.iter().map(|&z| log_mu(&th, z)).sum::<f64>() / draws.len() as f64;
            assert!((mean_log_mu(&th, &counts) - exact).abs() < tol, "{th:?}");
        }
        // a persistent truth contracts, within the box, on normal residuals
        let from = [0.0, 0.15, -0.08, 0.97];
        assert!(admits(&from, &counts));
        assert!(admits(&[0.0, 1.0, -1.0, 0.9999], &counts));
        // on residuals of 6, a large news term does not: |μ| = 5.1 at
        // (α, γ) = (1, 1); the step there is cut to the guard's edge
        let mut tail = vec![0.0; BINS];
        tail[bin(6.0)] = 1.0;
        let from = [0.0, 0.1, 0.0, 0.9];
        let to = [0.0, 1.0, 1.0, 0.9];
        assert!(admits(&from, &tail) && !admits(&to, &tail));
        let got = cut(&from, &to, &tail);
        assert!(admits(&got, &tail));
        assert!(got[1] > from[1] && got[1] < to[1], "{got:?}");
        assert!((mean_log_mu(&got, &tail) - BOUND).abs() < 1e-6);
        // from a point that breaks it too, (α, γ, β) shrink together to the
        // guard's edge: here no β alone keeps it (|μ| = 6.125 − β at the
        // bin's centre, 6.125), and the factor is e^{BOUND} / |μ(to)|
        let got = cut(&to, &to, &tail);
        assert!(admits(&got, &tail), "{got:?}");
        assert!((mean_log_mu(&got, &tail) - BOUND).abs() < 1e-9, "{got:?}");
        let t = got[1] / to[1];
        let want = libm::exp(BOUND - mean_log_mu(&to, &tail));
        assert!((t - want).abs() < 1e-11, "{t} against {want}");
        assert!(got[2] == t * to[2] && got[3] == t * to[3] && got[0] == to[0]);
        let near = [0.0, 0.1, 0.1, 0.9];
        let got = cut(&near, &[0.0, 0.3, 0.3, 0.9], &tail);
        assert!(admits(&got, &tail), "{got:?}");
    }
}
