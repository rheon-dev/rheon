//! Hansen's (1994) skewed t for `dist_fit`, standardized to mean 0 and
//! variance 1 as `arch`'s `SkewStudent` writes it, with the tail stepped on
//! `η = 1/ν` and the skew `λ`:
//!
//! ```text
//! c = Γ((ν+1)/2) / (√(π (ν−2)) Γ(ν/2)),   a = 4 λ c (ν−2)/(ν−1),   b = √(1 + 3λ² − a²)
//! x = b z + a,   q = 1 − λ for x < 0 and 1 + λ otherwise,   m = x/q
//! log f(z) = ln b + ln c − ((ν+1)/2) ln(1 + m²/(ν−2))
//! ```
//!
//! `λ = 0` is Student's t of variance 1, and `η = 0` (`ν = ∞`) a two-piece
//! normal, the normal itself at `λ = 0`. `x` is a two-piece t: the t of
//! variance 1 with density `g`, scaled by `1 − λ` left of 0 and `1 + λ`
//! right of it, so `P(x < 0) = (1 − λ)/2`; `a` is its mean and `b` its
//! standard deviation. In `η`, `ln c = log_norm(η) − ½ ln(1 − 2η)` with the
//! t's constant `log_norm` ([`student_t::log_norm`]), and `g(m) = t_ν(m/κ)/κ`
//! with `κ = √(1 − 2η)`, so every term near `η = 0` is the t's series form.
//!
//! **Scores and observed information.** On each piece the density is the
//! t's in scipy's form, at the kink `ξ = μ − σ a/b` and the piece's scale
//! `S = σ κ q/b`, times `q`, so both are the t's (`student_t`) taken to
//! `θ = (μ, ln σ, η, λ)` by the chain rule through `a(η, λ)` and `b(η, λ)`,
//! whose derivatives are closed ([`Shape::row`] has the forms).
//!
//! **Expected information**, closed. In `(ξ, ln s, η, λ)`, with `E` over the
//! t of variance 1:
//!
//! ```text
//! I_ξξ = E[ψ²]/(s² (1−λ²)) = (1+η)/((1+3η)(1−2η) s² (1−λ²))
//! I_ξλ = E[|m| ψ²]/(s (1−λ²)) = 4c (1+η)/((1+3η) s (1−λ²))
//! I_λλ = E[m² ψ²]/(1−λ²) = 3(1+η)/((1+3η)(1−λ²))
//! ```
//!
//! and the `(ln s, η)` block is the t's own, moved from scipy's scale to the
//! standard deviation (`ln s_scipy = ln s + ½ ln(1 − 2η)`); every other
//! entry is 0, by the symmetry of `g` on each piece. In `θ`, `Jᵀ I J` with
//! `J` the Jacobian of `(ξ, ln s, η, λ)` in `θ`. At the normal, `(η, λ) = 0`,
//! it is positive definite: Hansen's skew is regular at `λ = 0`.
//!
//! **Distribution function, quantile and shortfall**, from the t's on each
//! piece, `G(m) = T_ν(m/κ)`:
//!
//! ```text
//! P(Z ≤ z) = (1−λ) G(x/(1−λ))           for x < 0
//!          = 1 − (1+λ) G(−x/(1+λ))      otherwise
//! z_p = (x_p − a)/b,  x_p = (1−λ) G⁻¹(p/(1−λ)) for p < (1−λ)/2,  −(1+λ) G⁻¹((1−p)/(1+λ)) otherwise
//! E[Z | Z ≤ z_α] = (E[x 1{x ≤ x_α}]/α − a)/b
//! E[x 1{x ≤ x_α}] = (1−λ)² P(x_α/(1−λ))   for x_α < 0,   a + (1+λ)² P(−x_α/(1+λ)) otherwise
//! ```
//!
//! with `P(m₀) = ∫_{−∞}^{m₀} m g(m) dm = −κ (1 + η τ²)/(1 − η) t_ν(τ)` at
//! `τ = m₀/κ`, the t's lower partial moment.

use super::student_t;

/// The largest `η`, `ν = 2.05`, `arch`'s lower bound on `ν`: the variance
/// exists for `ν > 2`.
pub(crate) const ETA_MAX: f64 = 1.0 / 2.05;
/// `|λ| ≤ 0.99`, `arch`'s bound short of 1.
pub(crate) const LAM_MAX: f64 = 0.99;

/// The constants of the density at one shape.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Shape {
    pub(crate) eta: f64,
    pub(crate) lam: f64,
    /// `1 − 2η`.
    v: f64,
    /// `√(1 − 2η)`.
    kappa: f64,
    c: f64,
    /// `c (1 − 2η)/(1 − η)`, so `a = 4 λ φ` and `E|m| = 2φ`.
    phi: f64,
    pub(crate) a: f64,
    pub(crate) b: f64,
    /// `ln b − ½ ln(1 − 2η)`.
    ln_bk: f64,
}

impl Shape {
    pub(crate) fn new(eta: f64, lam: f64) -> Self {
        let v = 1.0 - 2.0 * eta;
        let ln_v = libm::log1p(-2.0 * eta);
        let c = libm::exp(student_t::log_norm(eta) - 0.5 * ln_v);
        let phi = c * v / (1.0 - eta);
        let a = 4.0 * lam * phi;
        let b = (1.0 + 3.0 * lam * lam - a * a).sqrt();
        Self {
            eta,
            lam,
            v,
            kappa: v.sqrt(),
            c,
            phi,
            a,
            b,
            ln_bk: libm::log(b) - 0.5 * ln_v,
        }
    }

    /// `(x, q, sg)` of a standardized `z`.
    fn piece(&self, z: f64) -> (f64, f64, f64) {
        let x = self.b * z + self.a;
        let sg = if x < 0.0 { -1.0 } else { 1.0 };
        (x, 1.0 + sg * self.lam, sg)
    }

    /// `log f(z)` of the standardized `z`.
    pub(crate) fn log_density(&self, z: f64) -> f64 {
        let (x, q, _) = self.piece(z);
        self.ln_bk + student_t::log_density(x / q / self.kappa, self.eta)
    }

    /// The scores and the observed information `−∂² log f` of one row's
    /// standardized `z` in `(μ, ln σ, η, λ)` at `σ = 1` (the `μ` entries
    /// divide by `σ` once for each `μ` at another scale), the information
    /// packed as [`Shape::info`]. On the row's piece, `log f` is the t's in
    /// scipy's form at `φ = (ξ, ln S, η)`, `S = σ κ q/b` the piece's scale,
    /// plus `ln q`:
    ///
    /// ```text
    /// log f = ℓ_t((y − ξ)/S; η) − ln S + ln q,   ξ = μ − σ a/b,   ln S = ln σ − ln b + ½ ln(1−2η) + ln q
    /// ∇_θ log f = Jᵀ ∇_φ ℓ + (0, 0, 0, sg/q)
    /// −∇²_θ log f = Jᵀ (−∇²_φ ℓ) J − Σ_k ∂ℓ/∂φ_k ∇²_θ φ_k + 1/q² at (λ, λ)
    /// ```
    ///
    /// with `J = ∂φ/∂θ`, the t's scores and observed information
    /// ([`student_t::scores`], [`student_t::observed_info`]) at `(y − ξ)/S`,
    /// and `∇²_θ φ` from the second derivatives of `a/b` and `ln b`.
    pub(crate) fn row(&self, z: f64, d: &Derivs) -> ([f64; 4], [f64; 10]) {
        let (x, q, sg) = self.piece(z);
        let s_cap = self.kappa * q / self.b;
        let tau = x / q / self.kappa;
        let [sl, sln, seta] = student_t::scores(tau, self.eta);
        let o = student_t::observed_info(tau, self.eta);
        let v = self.v;
        let sx = sl / s_cap;
        let j = [
            [1.0, -self.a / self.b, -d.r_eta, -d.r_lam],
            [0.0, 1.0, -d.lnb_eta - 1.0 / v, -d.lnb_lam + sg / q],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let mut g = [0.0f64; 4];
        for (k, gk) in g.iter_mut().enumerate() {
            *gk = sx * j[0][k] + sln * j[1][k] + seta * j[2][k];
        }
        g[3] += sg / q;
        let om = [
            [o[0] / (s_cap * s_cap), o[1] / s_cap, o[2] / s_cap],
            [o[1] / s_cap, o[3], o[4]],
            [o[2] / s_cap, o[4], o[5]],
        ];
        // −Σ_k ∂ℓ/∂φ_k ∇²φ_k, nonzero on (ln σ, η, λ) alone
        let mut curv = [[0.0f64; 4]; 4];
        curv[1][1] = sx * (self.a / self.b);
        curv[1][2] = sx * d.r_eta;
        curv[1][3] = sx * d.r_lam;
        curv[2][2] = sx * d.r_ee + sln * (d.lnb_ee + 2.0 / (v * v));
        curv[2][3] = sx * d.r_el + sln * d.lnb_el;
        curv[3][3] = sx * d.r_ll + sln * (d.lnb_ll + 1.0 / (q * q)) + 1.0 / (q * q);
        let mut out = [0.0f64; 10];
        let mut at = 0;
        for r in 0..4 {
            for c in r..4 {
                let mut acc = curv[r][c];
                for (a, ja) in om.iter().zip(&j) {
                    for (ob, jb) in a.iter().zip(&j) {
                        acc += ja[r] * ob * jb[c];
                    }
                }
                out[at] = acc;
                at += 1;
            }
        }
        (g, out)
    }

    /// The first and second derivatives of `a/b` and `ln b` in the shape,
    /// with `φ = c (1−2η)/(1−η)`, `a = 4λφ`, `b² = 1 + 3λ² − a²` and
    /// `(ln φ)_η = A(η) − 1/(1−2η) + 1/(1−η)`, `A` the t's
    /// [`student_t::score_const`] and `A′` its derivative.
    pub(crate) fn derivs(&self) -> Derivs {
        let (eta, lam, a, b) = (self.eta, self.lam, self.a, self.b);
        let (v, u) = (self.v, 1.0 - eta);
        let dl = student_t::score_const(eta) - 1.0 / v + 1.0 / u;
        let dl2 = student_t::score_const_deriv(eta) - 2.0 / (v * v) + 1.0 / (u * u);
        let phi_e = self.phi * dl;
        let phi_ee = self.phi * (dl * dl + dl2);
        let (a_e, a_l) = (4.0 * lam * phi_e, 4.0 * self.phi);
        let (a_ee, a_el) = (4.0 * lam * phi_ee, 4.0 * phi_e);
        // b² and its derivatives
        let b2 = b * b;
        let (b2_e, b2_l) = (-2.0 * a * a_e, 6.0 * lam - 2.0 * a * a_l);
        let b2_ee = -2.0 * (a_e * a_e + a * a_ee);
        let b2_el = -2.0 * (a_l * a_e + a * a_el);
        let b2_ll = 6.0 - 2.0 * a_l * a_l;
        let lnb_1 = |x: f64| x / (2.0 * b2);
        let lnb_2 = |xy: f64, x: f64, y: f64| xy / (2.0 * b2) - x * y / (2.0 * b2 * b2);
        let b3 = b2 * b;
        let r_1 = |ax: f64, bx: f64| ax / b - a * bx / (2.0 * b3);
        let r_2 = |axy: f64, ax: f64, ay: f64, bxy: f64, bx: f64, by: f64| {
            axy / b - (ax * by + ay * bx) / (2.0 * b3) - a * bxy / (2.0 * b3)
                + 3.0 * a * bx * by / (4.0 * b3 * b2)
        };
        Derivs {
            r_eta: r_1(a_e, b2_e),
            r_lam: r_1(a_l, b2_l),
            lnb_eta: lnb_1(b2_e),
            lnb_lam: lnb_1(b2_l),
            r_ee: r_2(a_ee, a_e, a_e, b2_ee, b2_e, b2_e),
            r_el: r_2(a_el, a_e, a_l, b2_el, b2_e, b2_l),
            r_ll: r_2(0.0, a_l, a_l, b2_ll, b2_l, b2_l),
            lnb_ee: lnb_2(b2_ee, b2_e, b2_e),
            lnb_el: lnb_2(b2_el, b2_e, b2_l),
            lnb_ll: lnb_2(b2_ll, b2_l, b2_l),
        }
    }

    /// The expected information per unit weight at `σ = 1` in `(μ, ln σ,
    /// η, λ)`, packed `[00, 01, 02, 03, 11, 12, 13, 22, 23, 33]` (the `μ`
    /// entries divide by `σ²` and `σ` at another scale).
    pub(crate) fn info(&self, d: &Derivs) -> [f64; 10] {
        let (eta, b) = (self.eta, self.b);
        let one_l2 = 1.0 - self.lam * self.lam;
        let e3 = (1.0 + eta) / (1.0 + 3.0 * eta);
        let [_, i1, i2, i3] = student_t::expected_info(eta);
        let v = self.v;
        // (ξ, ln s, η, λ) at s = 1/b
        let mut n = [[0.0f64; 4]; 4];
        n[0][0] = e3 / v * b * b / one_l2;
        n[0][3] = 4.0 * self.c * e3 * b / one_l2;
        n[3][0] = n[0][3];
        n[3][3] = 3.0 * e3 / one_l2;
        n[1][1] = i1;
        n[1][2] = i2 - i1 / v;
        n[2][1] = n[1][2];
        n[2][2] = i3 - 2.0 * i2 / v + i1 / (v * v);
        let j = [
            [1.0, -self.a / b, -d.r_eta, -d.r_lam],
            [0.0, 1.0, -d.lnb_eta, -d.lnb_lam],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        // n J, then Jᵀ (n J)
        let nj = n.map(|nr| [0, 1, 2, 3].map(|c| (0..4).map(|k| nr[k] * j[k][c]).sum::<f64>()));
        let mut out = [0.0f64; 10];
        let pairs = (0..4).flat_map(|r| (r..4).map(move |c| (r, c)));
        for (o, (r, c)) in out.iter_mut().zip(pairs) {
            *o = (0..4).map(|k| j[k][r] * nj[k][c]).sum();
        }
        out
    }

    /// The t of variance 1's distribution function.
    fn g_cdf(&self, m: f64) -> f64 {
        student_t::cdf(m / self.kappa, self.eta)
    }

    /// Its quantile.
    fn g_ppf(&self, p: f64) -> f64 {
        self.kappa * student_t::ppf(p, self.eta)
    }

    /// Its lower partial moment `∫_{−∞}^{m₀} m g(m) dm`.
    fn g_partial(&self, m0: f64) -> f64 {
        let tau = m0 / self.kappa;
        let eta = self.eta;
        -self.kappa * (1.0 + eta * tau * tau) / (1.0 - eta)
            * libm::exp(student_t::log_density(tau, eta))
    }

    /// `P(Z ≤ z)`.
    pub(crate) fn cdf(&self, z: f64) -> f64 {
        let x = self.b * z + self.a;
        if x < 0.0 {
            let q = 1.0 - self.lam;
            q * self.g_cdf(x / q)
        } else {
            let q = 1.0 + self.lam;
            1.0 - q * self.g_cdf(-x / q)
        }
    }

    /// The `z` with `P(Z ≤ z) = p`, `p` in `(0, 1)`.
    pub(crate) fn ppf(&self, p: f64) -> f64 {
        let (lo, hi) = (1.0 - self.lam, 1.0 + self.lam);
        let x = if p < 0.5 * lo {
            lo * self.g_ppf(p / lo)
        } else {
            -hi * self.g_ppf((1.0 - p) / hi)
        };
        (x - self.a) / self.b
    }

    /// The lower tail's expected shortfall `E[Z | Z ≤ z_α]`.
    pub(crate) fn lower_es(&self, alpha: f64) -> f64 {
        let x = self.b * self.ppf(alpha) + self.a;
        let (lo, hi) = (1.0 - self.lam, 1.0 + self.lam);
        let below = if x < 0.0 {
            lo * lo * self.g_partial(x / lo)
        } else {
            self.a + hi * hi * self.g_partial(-x / hi)
        };
        (below / alpha - self.a) / self.b
    }

    /// The skewness and excess kurtosis of `Z`, for `η < 1/4` (where the
    /// fourth moment exists): from the two-piece form's raw moments
    /// `E[x²] = 1 + 3λ²`, `E[x³] = 4λ(1 + λ²) E|m|³`, `E[x⁴] = (1 + 10λ² +
    /// 5λ⁴) E[m⁴]`, with `E|m|³ = 4c (1−2η)²/((1−η)(1−3η))` and `E[m⁴] =
    /// 3(1−2η)/(1−4η)`. At `λ = 0` the excess kurtosis is `6η/(1 − 4η)`,
    /// the moment start's.
    #[cfg(test)]
    pub(crate) fn skew_kurt(&self) -> (f64, f64) {
        let (eta, l2, a) = (self.eta, self.lam * self.lam, self.a);
        let m3 = 4.0 * self.c * self.v * self.v / ((1.0 - eta) * (1.0 - 3.0 * eta));
        let m4 = 3.0 * self.v / (1.0 - 4.0 * eta);
        let x2 = 1.0 + 3.0 * l2;
        let x3 = 4.0 * self.lam * (1.0 + l2) * m3;
        let x4 = m4 * (1.0 + l2 * (10.0 + 5.0 * l2));
        let b2 = self.b * self.b;
        let a2 = a * a;
        let skew = (x3 - 3.0 * a * x2 + 2.0 * a2 * a) / (b2 * self.b);
        let kurt = (x4 - 4.0 * a * x3 + 6.0 * a2 * x2 - 3.0 * a2 * a2) / (b2 * b2) - 3.0;
        (skew, kurt)
    }
}

/// The first and second derivatives of `a/b` (`r`) and `ln b` in `η` and
/// `λ`, at one shape.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Derivs {
    r_eta: f64,
    r_lam: f64,
    lnb_eta: f64,
    lnb_lam: f64,
    r_ee: f64,
    r_el: f64,
    r_ll: f64,
    lnb_ee: f64,
    lnb_el: f64,
    lnb_ll: f64,
}

#[cfg(test)]
mod tests;
