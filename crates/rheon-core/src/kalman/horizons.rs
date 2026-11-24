//! `kalman`'s forecasts at horizons (task 559), spans of the clock `Δ ≥ 0`
//! past the row read. Each is what the filter would make of a row with this
//! row's features at the row's clock plus `Δ`, with no row between:
//!
//! ```text
//! b(Δ) = T(Δ) b,   T(Δ) = Phi(Δ) in the anchor's coordinates          (transit, Phi = diag(2^(−Δ/r_i)))
//! f(Δ) = zᵀ b(Δ)                                                      f(0) = pred
//! F(Δ) = ∫_0^Δ f(u) du = zᵀ T̄(Δ) b,   T̄ the transition with each 2^(−u/r_i) integrated:
//!        r_i/ln 2 · (1 − 2^(−Δ/r_i)),  Δ at r_i = inf
//! V(Δ) = the prediction's variance from P(Δ) = T(Δ) P T(Δ)ᵀ + Q g(D + Δ)²
//! ```
//!
//! with `b` and `P` the row's after its transition, before its noise and its
//! update, and `D` the clock since the covariance's last observation, this
//! row's step included: the noise an observation `D + Δ` after the last
//! would take, carried into the anchor's coordinates as
//! `Kalman::covariance_now` carries it, and scaled to the prediction's
//! variance as `pred_var` is. The transition is linear in its factors, so
//! its integral is the same map at the integrated factors. Under the
//! default random walk the transition is the identity: the forecast does not
//! move with `Δ`, its integral is `Δ` times it, and only the variance grows.
//! The features are held at the row's: the forecast is the regression's at
//! them. Each target's three slots a horizon follow the predictions.

use super::{Kalman, KalmanCfg, a_number_or_none, forms, shared_noise, transit};

/// The most horizons a `kalman` spec may ask for.
pub const KALMAN_MAX_HORIZONS: usize = 64;

/// The horizons' checks, for [`KalmanCfg::validate`].
pub(super) fn check(cfg: &KalmanCfg, no: &mut crate::Refusals) {
    let hs = &cfg.horizons;
    if hs.len() > KALMAN_MAX_HORIZONS {
        no.refuse(format!(
            "kalman: forecast_spans may hold at most {KALMAN_MAX_HORIZONS} spans, but it holds {}. Each \
             adds three fields a target to every row. Ask for {KALMAN_MAX_HORIZONS} or fewer.",
            hs.len()
        ));
    }
    if let Some(h) = hs.iter().find(|h| !(h.is_finite() && **h >= 0.0)) {
        no.refuse(format!(
            "kalman: each of forecast_spans must be a finite span of the clock of 0 or more, but one \
             is {h}. A horizon is the clock past the row read: 0 is the row's own prediction. \
             Give a span of 0 or more."
        ));
    }
    for (i, h) in hs.iter().enumerate() {
        if hs[..i].contains(h) {
            no.refuse(format!(
                "kalman: forecast_spans holds {h} twice. Each horizon's fields are named by it, so it \
                 may be asked for once. Leave out the second."
            ));
            break;
        }
    }
}

impl Kalman {
    /// A prediction's variance for target `j` from `quad = zᵀ P z`, in the
    /// noise's units or not: `quad + R` with `obs_var`, `σ²_j (quad + 1)` in
    /// units of the noise, `σ²_j (quad/σ̄² + 1)` under `share_p` with the
    /// noise in the gain, `quad + σ²_j` otherwise; NaN before there is a
    /// noise.
    pub(super) fn variance_of(&self, j: usize, quad: f64) -> f64 {
        match self.cfg.obs_var {
            Some(r) => quad + r,
            // `P` in units of the noise: the target's own noise scales it,
            // shared or not (the module doc).
            None if self.cfg.noise_units => {
                let own = self.sig2[j];
                if own > 0.0 {
                    own * (quad + 1.0)
                } else {
                    f64::NAN
                }
            }
            None if self.cfg.share_p => {
                let mean = shared_noise(&mut Vec::new(), &self.sig2);
                let own = self.sig2[j];
                if own > 0.0 && mean > 0.0 {
                    own * (quad / mean + 1.0)
                } else {
                    f64::NAN
                }
            }
            None => {
                let own = self.sig2[j];
                if own > 0.0 { quad + own } else { f64::NAN }
            }
        }
    }

    /// The forecasts `step` writes after the predictions `pred` (one a
    /// target): from the state as the prediction reads it, after the row's
    /// transition, before its noise and update. Empty with no horizons.
    pub(super) fn ahead_of_the_step(&self, pred: &[f64]) -> Vec<f64> {
        let per = 3 * self.cfg.horizons.len();
        if per == 0 {
            return Vec::new();
        }
        let (m, k) = (self.cfg.n_targets, self.cfg.k_total());
        let mut ahead = vec![f64::NAN; m * per];
        let live: Vec<bool> = (0..k).map(|i| self.is_live(i)).collect();
        let (mut a, mut c) = (vec![1.0; k], vec![0.0; k]);
        self.map_into(&live, &mut a, &mut c);
        for j in 0..m {
            if !pred[j].is_finite() {
                continue;
            }
            let pi = if self.cfg.share_p { 0 } else { j };
            let p = self.cov(pi);
            self.forecasts_into(
                j,
                &self.zs,
                (&self.beta[j], &p, self.elapsed[pi]),
                (&live, &a, &c),
                &mut ahead[j * per..(j + 1) * per],
            );
        }
        ahead
    }

    /// The forecasts `predict` reads after its predictions `pred`, the
    /// state carried through the row's transition (`phi`, `g`) as `step`
    /// carries it, and the row's clock step added to each covariance's
    /// clock since its last observation: the numbers `step` writes.
    pub(super) fn ahead_of_the_read(
        &self,
        pred: &[f64],
        zs: &[f64],
        reverts: bool,
        phi: &[f64],
        g: Option<&[f64]>,
        d_clock: f64,
    ) -> Vec<f64> {
        let per = 3 * self.cfg.horizons.len();
        let (m, k) = (self.cfg.n_targets, self.cfg.k_total());
        let live: Vec<bool> = (0..k).map(|i| self.is_live(i)).collect();
        let (mut a, mut c) = (vec![1.0; k], vec![0.0; k]);
        self.map_into(&live, &mut a, &mut c);
        let mut v = vec![0.0; k];
        let mut ahead = vec![f64::NAN; m * per];
        for j in 0..m {
            if !pred[j].is_finite() {
                continue;
            }
            let pi = if self.cfg.share_p { 0 } else { j };
            let (b, p) = if reverts {
                let mut b = self.beta[j].clone();
                transit(&mut b, phi, g);
                let mut p = self.p[pi].clone();
                if self.factor() {
                    forms::transit_cov::<true>(&mut p, phi, g, &mut v);
                    p = forms::factor_product(&p, k);
                } else {
                    forms::transit_cov::<false>(&mut p, phi, g, &mut v);
                }
                (b, p)
            } else {
                (self.beta[j].clone(), self.cov(pi).into_owned())
            };
            self.forecasts_into(
                j,
                zs,
                (&b, &p, self.elapsed[pi] + d_clock),
                (&live, &a, &c),
                &mut ahead[j * per..(j + 1) * per],
            );
        }
        ahead
    }

    /// Target `j`'s forecasts at the horizons into `out`, three slots a
    /// horizon (the module doc): `zs` the row in the anchor's coordinates,
    /// `state` the row's `b` and `P` (dense) after its transition and the
    /// clock since the covariance's last observation, this row's step
    /// included, and `map` the live slots and the map `(a, c)` from the
    /// anchor.
    fn forecasts_into(
        &self,
        j: usize,
        zs: &[f64],
        state: (&[f64], &[f64], f64),
        map: (&[bool], &[f64], &[f64]),
        out: &mut [f64],
    ) {
        let (b, p, gap) = state;
        let (live, a, c) = map;
        let k = self.cfg.k_total();
        let pi = if self.cfg.share_p { 0 } else { j };
        let dot = |b: &[f64]| zs.iter().zip(b).map(|(z, b)| z * b).sum::<f64>();
        let now = dot(b);
        let reverts = self.cfg.reverts();
        let noise = self.readiness_noise(pi);
        let not_sized = self.is_unsized(pi);
        let mut q = vec![0.0; k];
        if !noise.is_nan() {
            self.q_held_into(noise, &mut q);
        }
        let off = usize::from(self.cfg.fit_intercept);
        let (mut phi, mut bar, mut g) = (vec![1.0; k], vec![0.0; k], vec![0.0; k]);
        let mut v = vec![0.0; k];
        for (h, &dt) in self.cfg.horizons.iter().enumerate() {
            let at = 3 * h;
            let mut pt = p.to_vec();
            if reverts {
                for i in 0..k {
                    phi[i] = self.cfg.phi(i, dt);
                    let r = self.cfg.revert_of(i);
                    bar[i] = if r.is_infinite() {
                        dt
                    } else {
                        let rate = std::f64::consts::LN_2 / r;
                        -(-rate * dt).exp_m1() / rate
                    };
                }
                let coupled = self.coupling(&phi, c, live, &mut g);
                let mut bt = b.to_vec();
                transit(&mut bt, &phi, coupled.then_some(&g[..]));
                out[at] = a_number_or_none(dot(&bt));
                forms::transit_cov::<false>(&mut pt, &phi, coupled.then_some(&g[..]), &mut v);
                let coupled = self.coupling(&bar, c, live, &mut g);
                let mut bi = b.to_vec();
                transit(&mut bi, &bar, coupled.then_some(&g[..]));
                out[at + 1] = a_number_or_none(dot(&bi));
            } else {
                out[at] = a_number_or_none(now);
                out[at + 1] = a_number_or_none(dt * now);
            }
            // the process noise an observation `gap + Δ` after the last would
            // take, carried into the anchor's coordinates
            let d = gap + dt;
            if d != 0.0 && !noise.is_nan() && !not_sized {
                for i in 0..k {
                    let qd = q[i] * self.cfg.gap_noise(i, d);
                    if qd == 0.0 || (self.cfg.standardize && pt[i * k + i] == 0.0) {
                        continue;
                    }
                    if !self.cfg.standardize || i < off {
                        pt[i * k + i] += qd;
                        continue;
                    }
                    if off == 1 {
                        let w = c[i] / a[i];
                        pt[0] += w * w * qd;
                        let o = -w * qd / a[i];
                        pt[i] += o;
                        pt[i * k] += o;
                    }
                    pt[i * k + i] += qd / (a[i] * a[i]);
                }
            }
            let mut quad = 0.0;
            for i in 0..k {
                let row = i * k;
                let mut acc = 0.0;
                for jj in 0..k {
                    acc += pt[row + jj] * zs[jj];
                }
                quad += zs[i] * acc;
            }
            out[at + 2] = self.variance_of(j, quad);
        }
    }
}
