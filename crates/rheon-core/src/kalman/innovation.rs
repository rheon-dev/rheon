//! `kalman`'s coefficient prior (`coef_prior`) and out-of-sample Gaussian
//! innovation diagnostics (`emit_loglik`, `emit_mahal`).
//!
//! Kept in a submodule because `kalman.rs` stands near the repository's
//! source-size cap (`tests/test_repo_hygiene.py`). Every transcendental in
//! this module (`libm::log`) calls the `libm` crate compiled from source,
//! never the platform's math library, so `loglik` stays bit-identical across
//! macOS, Linux and Windows.

use super::{Kalman, KalmanCfg, a_number_or_none, forms, shared_noise, transit};
use crate::model::{Extra, OnlineModel, Step};

/// `ln(2 * pi)` in IEEE-754 `f64` (`0x3ffd_67f1_c864_beb4`).
const LN_2_PI: f64 = 1.837_877_066_409_345_3;

impl KalmanCfg {
    /// Whether this filter emits at least one per-row innovation diagnostic
    /// (`emit_loglik` or `emit_mahal`).
    #[inline]
    pub fn emits_innovation(&self) -> bool {
        self.emit_loglik || self.emit_mahal
    }

    /// Number of per-row floats written into [`Extra::Kalman`]:
    /// `n_targets * (emit_loglik + emit_mahal)`.
    #[inline]
    pub fn extra_width(&self) -> usize {
        self.n_targets * (usize::from(self.emit_loglik) + usize::from(self.emit_mahal))
    }
}

/// Build the initial coefficient state `beta` (`n_targets x k_total`) for
/// [`Kalman::new`].
///
/// - Unstandardized (`standardize == false`): copies `coef_prior` directly,
///   except that under a diffuse intercept (`noise_units && obs_var.is_none()
///   && fit_intercept && obs.is_gaussian_identity()`) the intercept slot
///   starts at `0.0` so the diffuse first-observation step takes the level
///   with zero cancellation; without `coef_prior`, inverse/power links start
///   their intercept at `1.0` so `eta > 0`.
/// - Standardized (`standardize == true`): slopes wait until their feature's
///   scale is usable (seeded by [`Kalman::seed_standardized_slot_prior`]), so
///   raw feature levels never corrupt the anchor; a non-diffuse intercept
///   starts at `coef_prior[j][0]` (or `1.0` for inverse/power links without a
///   prior).
pub(super) fn initial_beta(cfg: &KalmanCfg) -> Vec<Vec<f64>> {
    let k = cfg.k_total();
    let m = cfg.n_targets;
    let diffuse = cfg.obs.is_gaussian_identity()
        && cfg.noise_units
        && cfg.obs_var.is_none()
        && cfg.fit_intercept;
    if let Some(prior) = &cfg.coef_prior {
        if !cfg.standardize {
            let mut beta = prior.clone();
            if diffuse {
                for b in &mut beta {
                    b[0] = 0.0;
                }
            }
            return beta;
        }
        let mut beta = vec![vec![0.0; k]; m];
        if cfg.fit_intercept && !diffuse {
            for (b, p) in beta.iter_mut().zip(prior) {
                b[0] = p[0];
            }
        }
        return beta;
    }
    let mut beta = vec![vec![0.0; k]; m];
    if !cfg.obs.link.defined_at_zero() && cfg.fit_intercept {
        for b in &mut beta {
            b[0] = 1.0;
        }
    }
    beta
}

/// Gaussian innovation log-likelihood and Mahalanobis distance for scalar
/// innovation `nu = y - z' b⁻` and innovation variance `s_abs = z' P⁻ z + R/w`
/// (in target `y`'s squared units).
///
/// Returns `(loglik, mahal)`, each `NaN` if `s_abs` is not positive/finite or
/// the result overflows.
#[inline]
pub(super) fn gaussian_innovation(nu: f64, s_abs: f64) -> (f64, f64) {
    if !(s_abs > 0.0 && s_abs.is_finite() && nu.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    let sd = s_abs.sqrt();
    let mahal = a_number_or_none(nu.abs() / sd);
    let nu2 = nu * nu;
    let quad = if nu2.is_finite() {
        nu2 / s_abs
    } else {
        let z = nu / sd;
        z * z
    };
    let ll = a_number_or_none(-0.5 * (LN_2_PI + libm::log(s_abs) + quad));
    (ll, mahal)
}

/// Prepare `out.extra` for [`Kalman`]: returns `Some(&mut [f64])` filled with
/// `NaN` when `extra_width > 0` (reusing an existing [`Extra::Kalman`] vector
/// with zero allocation), or sets `out.extra = None` and returns `None`.
#[inline]
pub(super) fn init_extra(extra: &mut Option<Extra>, width: usize) -> Option<&mut [f64]> {
    if width == 0 {
        *extra = None;
        return None;
    }
    if let Some(Extra::Kalman { values }) = extra {
        values.clear();
        values.resize(width, f64::NAN);
    } else {
        *extra = Some(Extra::Kalman {
            values: vec![f64::NAN; width],
        });
    }
    match extra {
        Some(Extra::Kalman { values }) => Some(values.as_mut_slice()),
        _ => None,
    }
}

impl Kalman {
    /// Seed slope slot `i` (`i >= off`) from `cfg.coef_prior` when it is
    /// first sized under `standardize = true`. Returns `true` if `beta` was
    /// updated.
    ///
    /// In anchor coordinates `c_i = b_i / s^a_i`, so `b_i = prior_i * s^a_i`
    /// places the slope at `prior_i` in the caller's original units. Where
    /// another target had already anchored slot `i` (`map_c[i] != 0`) and the
    /// intercept is not diffuse, shifting `b_0 -= map_c[i] * b_i` keeps the
    /// centered intercept anchored at the current feature mean.
    pub(super) fn seed_standardized_slot_prior(
        &mut self,
        pi: usize,
        i: usize,
        off: usize,
        open: bool,
    ) -> bool {
        let Some(prior) = &self.cfg.coef_prior else {
            return false;
        };
        if i < off {
            return false;
        }
        let scale = self.anchor_scale[i];
        let map_c = self.map_c[i];
        let targets = if self.cfg.share_p {
            0..self.cfg.n_targets
        } else {
            pi..pi + 1
        };
        let mut seeded = false;
        for j in targets {
            let b_i = prior[j][i] * scale;
            if b_i.is_finite() {
                self.beta[j][i] = b_i;
                seeded = true;
                if off == 1 && !open && map_c != 0.0 {
                    let shift = map_c * b_i;
                    if shift.is_finite() {
                        self.beta[j][0] -= shift;
                    }
                }
            }
        }
        seeded
    }

    /// Recompute `self.zb[j] = self.zs . self.beta[j]` for targets belonging
    /// to covariance `pi` after `seed_standardized_slot_prior` initializes
    /// newly-usable slope slots. `out.pred` already holds the pre-sizing
    /// prediction, so out-of-sample `predict` parity is preserved while the
    /// measurement update uses the prior-seeded mean.
    pub(super) fn refresh_zb_for_cov(&mut self, pi: usize) {
        let targets = if self.cfg.share_p {
            0..self.cfg.n_targets
        } else {
            pi..pi + 1
        };
        for j in targets {
            self.zb[j] = self.zs.iter().zip(&self.beta[j]).map(|(z, b)| z * b).sum();
        }
    }

    /// Target `j`'s absolute innovation variance `S_abs` (in target `y`'s
    /// squared units) from pre-row noise only — never from the row's own
    /// squared innovation `e^2` (CLAUDE.md hard rule 2). Returns `NaN` before
    /// target `j` has a positive finite noise variance.
    #[inline]
    pub(super) fn innovation_s_abs(
        &self,
        j: usize,
        zpz: f64,
        s_inn: f64,
        weight: f64,
        shared_s2: f64,
    ) -> f64 {
        if self.cfg.obs_var.is_some() {
            return s_inn;
        }
        let own = self.sig2[j];
        if !(own > 0.0 && own.is_finite()) {
            return f64::NAN;
        }
        if self.units() {
            own * s_inn
        } else if self.cfg.share_p {
            if shared_s2 > 0.0 && shared_s2.is_finite() {
                own * (zpz / shared_s2 + 1.0 / weight)
            } else {
                f64::NAN
            }
        } else {
            s_inn
        }
    }

    /// Record target `j`'s out-of-sample `loglik` and/or `mahal` into
    /// `extra_vals` (block layout: `loglik[0..m]`, then `mahal[0..m]`).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(super) fn record_innovation(
        &self,
        j: usize,
        zpz: f64,
        s_inn: f64,
        nu: f64,
        weight: f64,
        shared_s2: f64,
        extra_vals: &mut [f64],
    ) {
        let s_abs = self.innovation_s_abs(j, zpz, s_inn, weight, shared_s2);
        let (ll, mh) = gaussian_innovation(nu, s_abs);
        let m = self.cfg.n_targets;
        let mut off = 0;
        if self.cfg.emit_loglik {
            extra_vals[off + j] = ll;
            off += m;
        }
        if self.cfg.emit_mahal {
            extra_vals[off + j] = mh;
        }
    }

    /// Out-of-sample read at an explicit observation weight `weight`, filling
    /// `out` with `pred`, `n_eff`, and (when `emit_loglik || emit_mahal`)
    /// [`Extra::Kalman`] bit-identical to [`OnlineModel::step_into`], without
    /// mutating `self`.
    ///
    /// A zero-weight row (`weight <= 0.0`) or a row with missing targets
    /// produces `NaN` in `Extra::Kalman` while still scoring `pred`.
    pub fn predict_with_weight(
        &self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
    ) -> Step {
        let mut out = Step::default();
        self.predict_with_weight_into(x, y, d_clock, weight, &mut out);
        out
    }

    /// Buffer-reusing form of [`Self::predict_with_weight`].
    pub fn predict_with_weight_into(
        &self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        *out = OnlineModel::predict(self, x, d_clock);
        if !self.cfg.emits_innovation() || !crate::model::all_usable(x) {
            return;
        }
        let Some(Extra::Kalman { values: extra_vals }) = out.extra.as_mut() else {
            return;
        };
        let n_eff = out.n_eff;
        let ready = n_eff >= self.cfg.min_weight;
        if !ready || !(crate::model::usable(weight) && weight > 0.0) {
            return;
        }
        let m = self.cfg.n_targets;
        let any_y = (0..m).any(|j| {
            y.get(j)
                .copied()
                .flatten()
                .is_some_and(crate::model::usable)
        });
        if !any_y {
            return;
        }
        if self.factor() {
            self.score_innovations_as::<true>(x, y, d_clock, weight, extra_vals);
        } else {
            self.score_innovations_as::<false>(x, y, d_clock, weight, extra_vals);
        }
    }

    /// Compute `loglik` and/or `mahal` on a read-only copy of the per-target
    /// prior `(b⁻, P⁻)`, matching [`Kalman::step_as`] instruction-for-instruction
    /// so `predict_with_weight` is bit-identical to `step`.
    fn score_innovations_as<const FACTOR: bool>(
        &self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        extra_vals: &mut [f64],
    ) {
        let k = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let off = usize::from(self.cfg.fit_intercept);
        let units = self.units();
        let diffuse = units && self.cfg.fit_intercept;

        let mut zbuf = vec![0.0; k];
        if self.cfg.fit_intercept {
            zbuf[0] = 1.0;
            zbuf[1..].copy_from_slice(x);
        } else {
            zbuf.copy_from_slice(x);
        }
        let mut live: Vec<bool> = (0..k).map(|i| self.is_live_as::<FACTOR>(i)).collect();
        let (mut map_a, mut map_c) = (vec![1.0; k], vec![0.0; k]);
        if self.cfg.standardize {
            self.map_into(&live, &mut map_a, &mut map_c);
        }
        let mut anchor_scale = self.anchor_scale.clone();

        let reverts = self.cfg.reverts();
        let (phi, g, coupled) = if reverts {
            let phi: Vec<f64> = if self.cfg.revert_half_life.len() == 1 {
                vec![self.cfg.phi(0, d_clock); k]
            } else {
                (0..k).map(|i| self.cfg.phi(i, d_clock)).collect()
            };
            let mut g = vec![0.0; k];
            let coupled = self.coupling(&phi, &map_c, &live, &mut g);
            (phi, g, coupled)
        } else {
            (Vec::new(), Vec::new(), false)
        };
        let g_opt = coupled.then_some(&g[..]);

        let mut zs = vec![0.0; k];
        self.anchored_into(&zbuf, &live, &mut zs);

        let mut beta: Vec<Vec<f64>> = self.beta.clone();
        if reverts {
            for b in &mut beta {
                transit(b, &phi, g_opt);
            }
        }
        let mut zb: Vec<f64> = beta
            .iter()
            .map(|b| zs.iter().zip(b).map(|(z, bv)| z * bv).sum())
            .collect();

        let obs = |j: usize| {
            y.get(j)
                .copied()
                .flatten()
                .filter(|v| crate::model::usable(*v))
        };

        let shared_s2 = if self.cfg.share_p {
            shared_noise(&mut Vec::new(), &self.sig2)
        } else {
            f64::NAN
        };
        let mut shared_first: Option<f64> = None;
        let mut noise = vec![0.0; m];
        for (j, slot) in noise.iter_mut().enumerate() {
            *slot = match self.cfg.obs_var {
                Some(v) => v,
                None => {
                    let s2 = if self.cfg.share_p {
                        shared_s2
                    } else {
                        self.sig2[j]
                    };
                    if s2 > 0.0 {
                        s2
                    } else if self.cfg.share_p {
                        *shared_first.get_or_insert_with(|| {
                            let squares = (0..m).filter_map(|t| {
                                let e = obs(t)? - zb[t];
                                Some(e * e).filter(|e2| e2.is_finite())
                            });
                            let mut buf = Vec::new();
                            let sum = super::sum_ascending(&mut buf, squares);
                            if !buf.is_empty() {
                                super::first_noise(sum / buf.len() as f64)
                            } else {
                                0.0
                            }
                        })
                    } else {
                        obs(j).map_or(0.0, |yj| {
                            let e = yj - zb[j];
                            super::first_noise(e * e)
                        })
                    }
                }
            };
        }

        let mut vbuf = vec![0.0; k];
        let mut qbuf = vec![0.0; k];
        let mut xbuf = vec![0.0; k];
        let mut fbuf = vec![0.0; k];
        let mut pz = vec![0.0; k];

        let n_p = self.p.len();
        let mut p_work: Vec<Vec<f64>> = self.p.clone();
        if reverts {
            for p in &mut p_work {
                forms::transit_cov::<FACTOR>(p, &phi, g_opt, &mut vbuf);
            }
        }

        let sized_in = |p: &[f64], slot: usize| -> bool {
            if FACTOR {
                (slot..k).any(|r| p[r * k + slot] != 0.0)
            } else {
                p[slot * k + slot] != 0.0
            }
        };

        for (pi, p) in p_work.iter_mut().enumerate().take(n_p) {
            let informs = if self.cfg.share_p {
                (0..m).any(|j| obs(j).is_some())
            } else {
                obs(pi).is_some()
            };
            if !informs {
                continue;
            }
            let gap = self.elapsed[pi] + d_clock;
            let sigma2 = noise[if self.cfg.share_p { 0 } else { pi }];
            let open_now = units && self.cfg.fit_intercept && !sized_in(p, 0);

            // Process noise (exact same arithmetic as `add_process_noise` / `factor_noise`).
            let add_pn = |p: &mut [f64],
                          qbuf: &mut [f64],
                          xbuf: &mut [f64],
                          map_a: &[f64],
                          map_c: &[f64]| {
                if gap == 0.0 {
                    return;
                }
                self.q_held_into(sigma2, qbuf);
                if FACTOR {
                    for i in 0..k {
                        let qd = qbuf[i] * self.cfg.gap_noise(i, gap);
                        if qd == 0.0 {
                            continue;
                        }
                        if !self.cfg.standardize {
                            if !(open_now && i < off) {
                                xbuf.fill(0.0);
                                xbuf[i] = qd.sqrt();
                                forms::factor_add(p, xbuf);
                            }
                            continue;
                        }
                        if !sized_in(p, i) {
                            continue;
                        }
                        if i < off {
                            xbuf.fill(0.0);
                            xbuf[i] = qd.sqrt();
                            forms::factor_add(p, xbuf);
                            continue;
                        }
                        let a = map_a[i];
                        let sv = qd.sqrt();
                        xbuf.fill(0.0);
                        xbuf[i] = sv / a;
                        if off == 1 {
                            xbuf[0] = -(map_c[i] / a) * sv;
                        }
                        forms::factor_add(p, xbuf);
                    }
                    return;
                }
                let mut corner = 0.0;
                for i in 0..k {
                    let qd = qbuf[i] * self.cfg.gap_noise(i, gap);
                    if !self.cfg.standardize {
                        if !(open_now && i < off) {
                            p[i * k + i] += qd;
                        }
                        continue;
                    }
                    if p[i * k + i] == 0.0 || qd == 0.0 {
                        continue;
                    }
                    if i < off {
                        p[i * k + i] += qd;
                        continue;
                    }
                    let a = map_a[i];
                    if off == 1 {
                        let v = map_c[i] / a;
                        corner += v * v * qd;
                        let o = -v * qd / a;
                        p[i] += o;
                        p[i * k] += o;
                    }
                    p[i * k + i] += qd / (a * a);
                }
                if corner != 0.0 {
                    p[0] += corner;
                }
            };

            if self.cfg.standardize {
                add_pn(p, &mut qbuf, &mut xbuf, &map_a, &map_c);
                let size_noise = if units {
                    Some((1.0, false))
                } else {
                    self.cfg
                        .obs_var
                        .or_else(|| {
                            let mut basis = self.basis[pi].clone();
                            if self.cfg.share_p {
                                let squares = (0..m).filter_map(|j| {
                                    let e = obs(j)? - zb[j];
                                    Some(e * e).filter(|e2| *e2 > 0.0 && e2.is_finite())
                                });
                                let mut buf = Vec::new();
                                let sum = super::sum_ascending(&mut buf, squares);
                                if !buf.is_empty() {
                                    basis.add(sum, buf.len() as f64, weight);
                                }
                            } else if let Some(yj) = obs(pi) {
                                let e = yj - zb[pi];
                                let e2 = e * e;
                                if e2 > 0.0 && e2.is_finite() {
                                    basis.add(e2, 1.0, weight);
                                }
                            }
                            basis.noise()
                        })
                        .map(|r| (r, true))
                };
                if let Some((r_sz, intercept_sz)) = size_noise {
                    let v = self.cfg.p0 * r_sz;
                    if v > 0.0 && v.is_finite() {
                        let mut seeded = false;
                        for i in 0..k {
                            if sized_in(p, i) || !self.current_scale(i).1 {
                                continue;
                            }
                            if i < off {
                                if intercept_sz {
                                    if FACTOR {
                                        xbuf.fill(0.0);
                                        xbuf[0] = v.sqrt();
                                        forms::factor_add(p, &mut xbuf);
                                    } else {
                                        p[0] += v;
                                    }
                                    live[0] = true;
                                }
                                continue;
                            }
                            if !live[i] {
                                anchor_scale[i] = self.current_scale(i).0;
                                map_a[i] = 1.0;
                                map_c[i] = 0.0;
                            }
                            let a = map_a[i];
                            let open_i = units && self.cfg.fit_intercept && !sized_in(p, 0);
                            let cc = (off == 1 && !open_i).then(|| map_c[i] / a);
                            if FACTOR {
                                let sv = v.sqrt();
                                xbuf.fill(0.0);
                                xbuf[i] = sv / a;
                                if let Some(c) = cc {
                                    xbuf[0] = -c * sv;
                                }
                                forms::factor_add(p, &mut xbuf);
                            } else {
                                if let Some(c) = cc {
                                    p[0] += c * c * v;
                                    let o = -c * v / a;
                                    p[i] += o;
                                    p[i * k] += o;
                                }
                                p[i * k + i] += v / (a * a);
                            }
                            live[i] = true;
                            if let Some(prior) = &self.cfg.coef_prior {
                                let targets = if self.cfg.share_p { 0..m } else { pi..pi + 1 };
                                let sc = anchor_scale[i];
                                let mc = map_c[i];
                                for j in targets {
                                    let b_i = prior[j][i] * sc;
                                    if b_i.is_finite() {
                                        beta[j][i] = b_i;
                                        seeded = true;
                                        if off == 1 && !open_i && mc != 0.0 {
                                            let shift = mc * b_i;
                                            if shift.is_finite() {
                                                beta[j][0] -= shift;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if seeded {
                            let targets = if self.cfg.share_p { 0..m } else { pi..pi + 1 };
                            for j in targets {
                                zb[j] = zs.iter().zip(&beta[j]).map(|(z, bv)| z * bv).sum();
                            }
                        }
                    }
                }
            } else if units {
                add_pn(p, &mut qbuf, &mut xbuf, &map_a, &map_c);
            } else if forms::unsized_of::<FACTOR>(p, k) {
                let v = self.cfg.p0 * sigma2;
                if v > 0.0 && v.is_finite() {
                    let diag = if FACTOR { v.sqrt() } else { v };
                    p.fill(0.0);
                    for i in 0..k {
                        p[i * k + i] = diag;
                    }
                }
            } else {
                add_pn(p, &mut qbuf, &mut xbuf, &map_a, &map_c);
            }
        }

        let open_shared = diffuse && self.cfg.share_p && !sized_in(&p_work[0], 0);
        let mut shared_pz = false;
        let mut zpz_kept = 0.0;
        for j in 0..m {
            let pi = if self.cfg.share_p { 0 } else { j };
            let Some(yj) = obs(j) else {
                continue;
            };
            let open = open_shared || (diffuse && !self.cfg.share_p && !sized_in(&p_work[pi], 0));
            if open || (diffuse && self.wj[j] == 0.0) || forms::unsized_of::<FACTOR>(&p_work[pi], k)
            {
                continue;
            }
            if !shared_pz {
                zpz_kept = if FACTOR {
                    forms::factor_f(&p_work[pi], &zs, &mut fbuf);
                    forms::factor_cf(&p_work[pi], &fbuf, &mut pz);
                    fbuf.iter().map(|v| v * v).sum()
                } else {
                    super::matvec(&p_work[pi], &zs, &mut pz);
                    zs.iter().zip(&pz).map(|(z, pv)| z * pv).sum()
                };
                shared_pz = self.cfg.share_p;
            }
            let r_inn = if units { 1.0 } else { noise[j] };
            let s_inn = zpz_kept + r_inn / weight;
            let err = yj - zb[j];
            if r_inn > 0.0 && s_inn > 0.0 && s_inn.is_finite() && err.is_finite() {
                self.record_innovation(j, zpz_kept, s_inn, err, weight, shared_s2, extra_vals);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// Every transcendental in `kalman/innovation.rs` calls the `libm` crate,
    /// never the platform's math library, so `loglik` stays bit-identical
    /// across macOS, Linux and Windows.
    #[test]
    fn no_platform_math_library_is_called() {
        let src = include_str!("innovation.rs");
        let code: String = src
            .split("#[cfg(test)]")
            .next()
            .unwrap()
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let platform = [
            ".ln(", ".ln_1p(", ".log(", ".log2(", ".log10(", ".exp(", ".exp2(", ".exp_m1(",
            ".powf(", ".powi(", ".sin(", ".cos(", ".tan(", ".sinh(", ".cosh(", ".tanh(", ".asin(",
            ".acos(", ".atan(", ".atan2(", ".cbrt(", ".hypot(", "f64::ln", "f64::exp", "f64::pow",
        ];
        for call in platform {
            assert!(
                !code.contains(call),
                "kalman/innovation.rs calls `{call}`: route it through `libm`"
            );
        }
    }
}
