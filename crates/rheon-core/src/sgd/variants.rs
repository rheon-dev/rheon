//! Shared parameter averaging (`sgd`, `pa`) and `sgd`'s elastic-net (`l1`),
//! Bottou `optimal` schedule, and Vowpal Wabbit importance-invariant step.

use serde::{Deserialize, Serialize};

/// Decay-weighted Polyak–Ruppert parameter average shared by `sgd` and `pa`.
///
/// For each target `j`, maintains `W_j' = lam * W_j + w` and
/// `bbar_j' = bbar_j + (w / W_j') * (beta_j' - bbar_j)` in the same
/// coordinates as the live iterate `beta_j`. At `half_life = inf` and unit
/// weights this is the uniform Polyak–Ruppert average of every learned
/// iterate. A zero-weight row only ages `W_j` and never touches `bbar_j`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct AvgState {
    pub(crate) beta: Vec<Vec<f64>>,
    pub(crate) w: Vec<f64>,
}

impl AvgState {
    pub(crate) fn new(beta: &[Vec<f64>]) -> Self {
        Self {
            beta: beta.to_vec(),
            w: vec![0.0; beta.len()],
        }
    }

    pub(crate) fn has_shape(&self, n_targets: usize, k_total: usize) -> bool {
        self.beta.len() == n_targets
            && self.beta.iter().all(|b| b.len() == k_total)
            && self.w.len() == n_targets
            && self.w.iter().all(|w| w.is_finite() && *w >= 0.0)
    }

    /// Age target `j`'s accumulated weight without learning a new iterate.
    #[inline]
    pub(crate) fn age(&mut self, j: usize, lam: f64) {
        self.w[j] *= lam;
    }

    /// Fold target `j`'s new live iterate `live` into the running average at
    /// decay factor `lam` and positive row weight `weight`.
    #[inline]
    pub(crate) fn learn(&mut self, j: usize, lam: f64, weight: f64, live: &[f64]) {
        if !(weight > 0.0 && weight.is_finite()) {
            self.w[j] *= lam;
            return;
        }
        let lam_t = crate::gaps::decay_into(lam, self.w[j], weight);
        let aged = lam_t * self.w[j];
        let w_new = aged + weight;
        if !(w_new > 0.0 && w_new.is_finite()) {
            self.w[j] = aged;
            return;
        }
        if aged == 0.0 {
            self.beta[j].copy_from_slice(live);
        } else {
            let share = weight / w_new;
            for (a, &b) in self.beta[j].iter_mut().zip(live) {
                *a += share * (b - *a);
            }
        }
        self.w[j] = w_new;
    }
}

/// Tsuruoka et al. (2009) cumulative L1 penalty state for `sgd` (`l1 > 0`).
///
/// Tracks `u` (cumulative L1 penalty available) and `q` (cumulative L1
/// truncation already applied to each slope), per target. Under a shared-rate
/// schedule (`constant`, `inv_scaling`, `optimal`), `u[j]` has length 1; under
/// `adagrad`, `u[j]` has length `n_features` (one per slope).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct EnetState {
    pub(crate) u: Vec<Vec<f64>>,
    pub(crate) q: Vec<Vec<f64>>,
}

impl EnetState {
    pub(crate) fn new(n_targets: usize, n_features: usize, per_coord: bool) -> Self {
        let u_len = if per_coord { n_features } else { 1 };
        Self {
            u: vec![vec![0.0; u_len]; n_targets],
            q: vec![vec![0.0; n_features]; n_targets],
        }
    }

    pub(crate) fn has_shape(&self, n_targets: usize, n_features: usize, per_coord: bool) -> bool {
        let u_len = if per_coord { n_features } else { 1 };
        self.u.len() == n_targets
            && self.q.len() == n_targets
            && self
                .u
                .iter()
                .all(|v| v.len() == u_len && v.iter().all(|x| x.is_finite() && *x >= 0.0))
            && self
                .q
                .iter()
                .all(|v| v.len() == n_features && v.iter().all(|x| x.is_finite()))
    }

    /// Reset target `j`'s cumulative L1 penalty state (e.g. after a Huber refit
    /// replaces the iterate).
    pub(crate) fn reset_target(&mut self, j: usize) {
        self.u[j].fill(0.0);
        self.q[j].fill(0.0);
    }
}

/// Optional state carried by `Sgd` when `average`, `l1 > 0`, or
/// `schedule == Optimal` is enabled. `None` when none of those options is set,
/// adding zero bytes to existing serialized states.
///
/// Fields inside `SgdExtras` never skip serialization so positional compact
/// msgpack (`rmp_serde::to_vec`) always round-trips any combination cleanly.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SgdExtras {
    #[serde(default)]
    pub(crate) avg: Option<AvgState>,
    #[serde(default)]
    pub(crate) enet: Option<EnetState>,
    /// Per-target learned step counter for `LearningRate::Optimal`.
    #[serde(default)]
    pub(crate) t_step: Vec<u64>,
}

/// Bottou's `optimal` learning rate schedule matching scikit-learn's
/// `SGDRegressor(learning_rate="optimal")`:
/// `typw = sqrt(1 / sqrt(alpha))`, `t0 = 1 / (alpha * typw)`,
/// `eta(t) = 1 / (alpha * (t0 + t))` for 0-indexed learned step `t`.
#[inline]
pub(crate) fn optimal_rate(alpha: f64, t: u64) -> f64 {
    let typw = (1.0 / alpha.sqrt()).sqrt();
    let t0 = 1.0 / (alpha * typw);
    1.0 / (alpha * (t0 + t as f64))
}

/// Karampatziakis & Langford (2011) importance-weight-invariant squared-loss
/// step factor (Vowpal Wabbit `--invariant`):
///
/// ```text
/// s(h) = (1 - exp(-lr * w * zz)) / zz = -expm1(-lr * w * zz) / zz
/// ```
///
/// Replaces `lr * w` in the squared-loss update so that the updated prediction
/// `p' = y + (p - y) * exp(-lr * w * zz)` strictly contracts toward `y` for any
/// positive `lr`, `w`, and `zz = ||z||^2`, agreeing with plain SGD to first
/// order as `lr * w * zz -> 0`. Uses `libm::expm1` for cross-platform
/// bit-identity.
#[inline]
pub(crate) fn invariant_scale(lr: f64, weight: f64, zz: f64) -> f64 {
    if !(lr > 0.0
        && lr.is_finite()
        && weight > 0.0
        && weight.is_finite()
        && zz > 0.0
        && zz.is_finite())
    {
        return 0.0;
    }
    let x = (lr * weight) * zz;
    if !x.is_finite() {
        1.0 / zz
    } else {
        -libm::expm1(-x) / zz
    }
}

/// Apply Tsuruoka et al.'s cumulative L1 truncation to a single coordinate `z`
/// given cumulative penalty `u` and previously applied truncation `q`.
/// Returns `(new_val, new_q)`.
#[inline]
pub(crate) fn tsuruoka_clip(z: f64, u: f64, q: f64) -> (f64, f64) {
    let w = if z > 0.0 {
        (z - (u + q)).max(0.0)
    } else if z < 0.0 {
        (z + (u - q)).min(0.0)
    } else {
        0.0
    };
    (w, q + (w - z))
}

/// Check and compute a direct (unmapped) variant step for target `j` when any
/// of `l1 > 0`, `schedule == Optimal`, or `invariant` is active.
///
/// Writes candidate slopes into `next_slopes` and candidate `q` into `next_q`
/// (when `l1 > 0`), and returns `Some((intercept_step, next_u_opt))` on
/// success or `None` if any value is non-finite (leaving state untouched).
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_direct_variant(
    beta: &[f64],
    off: usize,
    z: &[f64],
    d: f64,
    weight: f64,
    (l1, l2, clip, ada_eps): (f64, f64, f64, f64),
    lr_row: Option<f64>,
    lr0: f64,
    invariant: bool,
    zz: f64,
    level_step: Option<f64>,
    g2: Option<&[f64]>,
    u_cur: Option<&[f64]>,
    q_cur: Option<&[f64]>,
    next_slopes: &mut [f64],
    next_g2: &mut [f64],
    next_u: &mut [f64],
    next_q: &mut [f64],
) -> Option<f64> {
    let n = z.len();
    // Multiplicative factor on `d * z_i`: under `invariant`, `s(h)` replaces
    // `lr * w`; otherwise `lr * w` (shared rate) or per-coordinate AdaGrad.
    let inv_scale = if invariant {
        let lr = lr_row?;
        Some(invariant_scale(lr, weight, zz))
    } else {
        None
    };

    // Intercept step (unpenalized).
    let b0_step = if off == 1 {
        if let Some(sc) = inv_scale {
            let g = (d * weight).clamp(-clip, clip);
            if !g.is_finite() {
                return None;
            }
            // Match `clip`: if unclipped (`g == d * weight`), use exact `-sc * d`;
            // if clipped, scale consistently by `sc / weight`.
            let s = if g == d * weight {
                -(sc * d)
            } else {
                -(sc * (g / weight))
            };
            if !(beta[0] + level_step.unwrap_or(s)).is_finite() {
                return None;
            }
            level_step.unwrap_or(s)
        } else {
            let g = (d * weight).clamp(-clip, clip);
            let s = match lr_row {
                Some(lr) => {
                    if !g.is_finite() {
                        return None;
                    }
                    -(lr * g)
                }
                None => {
                    let s0 = g2?[0] + g * g;
                    if !s0.is_finite() {
                        return None;
                    }
                    next_g2[0] = s0;
                    let den = s0.sqrt() + ada_eps;
                    if den > 0.0 { -(lr0 / den * g) } else { 0.0 }
                }
            };
            let chosen = level_step.unwrap_or(s);
            if !(beta[0] + chosen).is_finite() {
                return None;
            }
            chosen
        }
    } else {
        0.0
    };

    // Slopes step.
    let shared_u = if l1 > 0.0 {
        if let Some(lr) = lr_row {
            let u0 = u_cur?[0] + lr * l1;
            if !u0.is_finite() {
                return None;
            }
            next_u[0] = u0;
            Some(u0)
        } else {
            None
        }
    } else {
        None
    };

    for i in 0..n {
        let bi = beta[off + i];
        let zi = z[i];
        let new_b = if let Some(sc) = inv_scale {
            let lr = lr_row?;
            let shrunk = bi * (1.0 - l2 * lr).max(0.0);
            let raw_g = d * zi * weight;
            let g = raw_g.clamp(-clip, clip);
            let step = if g == raw_g {
                -(sc * d * zi)
            } else {
                -(sc * (g / weight))
            };
            let cand = shrunk + step;
            if !cand.is_finite() {
                return None;
            }
            cand
        } else if let Some(lr) = lr_row {
            let g = (d * zi * weight).clamp(-clip, clip);
            if !g.is_finite() {
                return None;
            }
            let shrunk = bi * (1.0 - l2 * lr).max(0.0);
            let cand = shrunk - lr * g;
            if !cand.is_finite() {
                return None;
            }
            cand
        } else {
            // AdaGrad with l1 > 0 (and optional l2).
            let g_loss = (d * zi * weight).clamp(-clip, clip);
            let g_tot = (g_loss + l2 * bi).clamp(-clip, clip);
            let s = g2?[off + i] + g_tot * g_tot;
            if !s.is_finite() {
                return None;
            }
            next_g2[off + i] = s;
            let den = s.sqrt() + ada_eps;
            let eta_i = if den > 0.0 { lr0 / den } else { 0.0 };
            let shrunk = bi * (1.0 - l2 * eta_i).max(0.0);
            let cand = shrunk - eta_i * g_loss;
            if !cand.is_finite() {
                return None;
            }
            if l1 > 0.0 {
                let ui = u_cur?[i] + eta_i * l1;
                if !ui.is_finite() {
                    return None;
                }
                next_u[i] = ui;
            }
            cand
        };

        if l1 > 0.0 {
            let ui = shared_u.unwrap_or_else(|| next_u[i]);
            let (w, qn) = tsuruoka_clip(new_b, ui, q_cur?[i]);
            if !w.is_finite() || !qn.is_finite() {
                return None;
            }
            next_slopes[i] = w;
            next_q[i] = qn;
        } else {
            next_slopes[i] = new_b;
        }
    }

    Some(b0_step)
}

/// Check and compute a standardized-held (mapped) variant step for target `j`
/// when any of `l1 > 0`, `schedule == Optimal`, or `invariant` is active.
///
/// Works in standardized slope coordinates `b_std_i = s_i * b_i`, applies the
/// same shrinkage, gradient step, and Tsuruoka L1 truncation as
/// [`step_direct_variant`], then maps back `b_i_new = b_std_i_new / s_i` and
/// shifts the intercept `b_0_new = b_0 + b0_step - sum_i m_i (b_i_new - b_i)`.
/// Writes full candidate coefficients into `next_beta` and returns `true` if
/// every updated value is finite.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_mapped_variant(
    beta: &[f64],
    off: usize,
    zs: &[f64],
    ss: &[f64],
    ms: &[f64],
    d: f64,
    weight: f64,
    (l1, l2, clip, ada_eps): (f64, f64, f64, f64),
    lr_row: Option<f64>,
    lr0: f64,
    invariant: bool,
    zz: f64,
    level_step: Option<f64>,
    g2: Option<&[f64]>,
    u_cur: Option<&[f64]>,
    q_cur: Option<&[f64]>,
    next_beta: &mut [f64],
    next_g2: &mut [f64],
    next_u: &mut [f64],
    next_q: &mut [f64],
) -> bool {
    let n = beta.len() - off;
    let inv_scale = if invariant {
        let Some(lr) = lr_row else {
            return false;
        };
        Some(invariant_scale(lr, weight, zz))
    } else {
        None
    };

    let shared_u = if l1 > 0.0 {
        if let Some(lr) = lr_row {
            let Some(uc) = u_cur else {
                return false;
            };
            let u0 = uc[0] + lr * l1;
            if !u0.is_finite() {
                return false;
            }
            next_u[0] = u0;
            Some(u0)
        } else {
            None
        }
    } else {
        None
    };

    let mut shift = 0.0;
    for i in 0..n {
        let idx = off + i;
        let b_std = ss[idx] * beta[idx];
        let zi = zs[idx];
        let cand_std = if let Some(sc) = inv_scale {
            let Some(lr) = lr_row else {
                return false;
            };
            let shrunk = b_std * (1.0 - l2 * lr).max(0.0);
            let raw_g = d * zi * weight;
            let g = raw_g.clamp(-clip, clip);
            let step = if g == raw_g {
                -(sc * d * zi)
            } else {
                -(sc * (g / weight))
            };
            shrunk + step
        } else if let Some(lr) = lr_row {
            let g = (d * zi * weight).clamp(-clip, clip);
            if !g.is_finite() {
                return false;
            }
            let shrunk = b_std * (1.0 - l2 * lr).max(0.0);
            shrunk - lr * g
        } else {
            let Some(sums) = g2 else {
                return false;
            };
            let g_loss = (d * zi * weight).clamp(-clip, clip);
            let g_tot = (g_loss + l2 * b_std).clamp(-clip, clip);
            let s = sums[idx] + g_tot * g_tot;
            if !s.is_finite() {
                return false;
            }
            next_g2[idx] = s;
            let den = s.sqrt() + ada_eps;
            let eta_i = if den > 0.0 { lr0 / den } else { 0.0 };
            let shrunk = b_std * (1.0 - l2 * eta_i).max(0.0);
            let cand = shrunk - eta_i * g_loss;
            if l1 > 0.0 {
                let Some(uc) = u_cur else {
                    return false;
                };
                let ui = uc[i] + eta_i * l1;
                if !ui.is_finite() {
                    return false;
                }
                next_u[i] = ui;
            }
            cand
        };

        let final_std = if l1 > 0.0 {
            let Some(qc) = q_cur else {
                return false;
            };
            let ui = shared_u.unwrap_or_else(|| next_u[i]);
            let (w, qn) = tsuruoka_clip(cand_std, ui, qc[i]);
            if !w.is_finite() || !qn.is_finite() {
                return false;
            }
            next_q[i] = qn;
            w
        } else {
            cand_std
        };

        let new_bi = final_std / ss[idx];
        if !new_bi.is_finite() {
            return false;
        }
        next_beta[idx] = new_bi;
        shift += ms[idx] * (new_bi - beta[idx]);
    }

    if off == 1 {
        let step0 = if let Some(sc) = inv_scale {
            let g = (d * weight).clamp(-clip, clip);
            if !g.is_finite() {
                return false;
            }
            let s = if g == d * weight {
                -(sc * d)
            } else {
                -(sc * (g / weight))
            };
            level_step.unwrap_or(s)
        } else {
            let g = (d * weight).clamp(-clip, clip);
            let s = match lr_row {
                Some(lr) => {
                    if !g.is_finite() {
                        return false;
                    }
                    -(lr * g)
                }
                None => {
                    let Some(sums) = g2 else {
                        return false;
                    };
                    let s0 = sums[0] + g * g;
                    if !s0.is_finite() {
                        return false;
                    }
                    next_g2[0] = s0;
                    let den = s0.sqrt() + ada_eps;
                    if den > 0.0 { -(lr0 / den * g) } else { 0.0 }
                }
            };
            level_step.unwrap_or(s)
        };
        let new_b0 = beta[0] + step0 - shift;
        if !new_b0.is_finite() {
            return false;
        }
        next_beta[0] = new_b0;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invariant_scale_contracts_and_matches_first_order() {
        // Small step agrees with lr * w.
        let s = invariant_scale(1e-6, 1.0, 2.0);
        assert!((s - 1e-6).abs() < 1e-11);
        // Huge weight never overshoots 1 / zz.
        let big = invariant_scale(0.5, 1e6, 4.0);
        assert!((big - 0.25).abs() < 1e-15);
        // Overflow in lr * w * zz saturates cleanly at 1 / zz.
        let ovf = invariant_scale(1e200, 1e200, 5.0);
        assert_eq!(ovf, 0.2);
    }

    #[test]
    fn optimal_rate_matches_bottou() {
        let r0 = optimal_rate(0.0001, 0);
        assert!((r0 - 10.0).abs() < 1e-14);
        let r1 = optimal_rate(1.0, 1);
        assert!((r1 - 0.5).abs() < 1e-15);
    }

    #[test]
    fn avg_state_uniform_and_zero_weight() {
        let mut avg = AvgState::new(&[vec![0.0, 0.0]]);
        avg.learn(0, 1.0, 1.0, &[1.0, 2.0]);
        assert_eq!(avg.beta[0], vec![1.0, 2.0]);
        avg.learn(0, 1.0, 0.0, &[99.0, 99.0]);
        assert_eq!(avg.beta[0], vec![1.0, 2.0]);
        avg.learn(0, 1.0, 1.0, &[3.0, 6.0]);
        assert_eq!(avg.beta[0], vec![2.0, 4.0]);
    }
}
