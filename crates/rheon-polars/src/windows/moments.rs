//! The moments operator, `ewm_moments` (docs/PLAN.md task 554): a term of
//! exponentially weighted moments of one or more of a row's inputs, each
//! perhaps lagged, kept by `rheon_core::EwmMoments` in each group and read
//! once a stamp's rows are all in, as a backward window closed on the right
//! is. Each row weighs the decayed mass of the interval its value is held
//! over, the weight `ewm_mean` and `ewm_var` give it, and goes into the
//! model as a row of that weight; the queue of its kernel carries nothing
//! of it.
use super::*;

/// A moments operator's term (`ewm_moments`, task 554): the columns it
/// reads and the model that keeps its moments, one term over the kernel's
/// half-life and window, on the clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MomentsDef {
    /// The operator it is the term of, a [`Stat::Moments`] one.
    pub op: usize,
    /// Each column the term's legs read, as an index into a row's values: a
    /// leg's column is its place in this list.
    pub inputs: Vec<usize>,
    /// The model's settings: the one term, the kernel's decay, a clock (the
    /// row count without a clock column), the kernel's window, no cap but
    /// the window's own rows.
    pub cfg: rheon_core::EwmMomentsCfg,
}

impl MomentsDef {
    /// The settings a moments operator runs on its kernel: the term
    /// `term`, the kernel's half-life as its decay and its window as a
    /// window of the clock.
    pub fn cfg_of(
        k: &KernelDef,
        term: rheon_core::MomentTerm,
        lag_tolerance: Option<f64>,
    ) -> rheon_core::EwmMomentsCfg {
        let mut cfg =
            rheon_core::EwmMomentsCfg::new(rheon_core::Decay::Halflife(k.half_life), vec![term]);
        cfg.clocked = true;
        cfg.lag_tolerance = lag_tolerance;
        cfg.window_clock = k.window_size;
        // A window operator holds the rows of its window, as every other
        // operator's queue does, whatever their number.
        cfg.max_state_bytes = Some(u64::MAX);
        cfg
    }
}

/// A group's moments for one moments operator: the model, and the raw
/// clock of the last row it learned since the group's windows started over
/// (the policy time, as a float clock, without a clock column).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct MomentsRun {
    #[serde(deserialize_with = "reloaded_moments")]
    pub(super) model: Box<rheon_core::EwmMoments>,
    pub(super) last: Option<ClockValue>,
    /// The row's values of the term's columns: a buffer, never state.
    #[serde(skip)]
    x: Vec<f64>,
}

/// A saved model, its derived parts formed again as a state file's are.
fn reloaded_moments<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Box<rheon_core::EwmMoments>, D::Error> {
    let m = rheon_core::EwmMoments::deserialize(d)?;
    m.reloaded().map(Box::new).map_err(serde::de::Error::custom)
}

impl MomentsRun {
    pub(super) fn new(def: &MomentsDef) -> Self {
        Self {
            model: Box::new(
                rheon_core::EwmMoments::new(def.cfg.clone()).expect("checked as the core is built"),
            ),
            last: None,
            x: Vec::with_capacity(def.inputs.len()),
        }
    }

    /// Start over: no row learned.
    pub(super) fn restart(&mut self) {
        let cfg = self.model.cfg().clone();
        *self.model = rheon_core::EwmMoments::new(cfg).expect("its settings ran before");
        self.last = None;
    }

    /// The term read at the stamp of the clock value `at`; NaN where it is
    /// withheld.
    pub(super) fn read(&self, at: ClockValue) -> f64 {
        if self.last.is_none() {
            return f64::NAN;
        }
        self.model.read_at(stamp_of(at))[0].unwrap_or(f64::NAN)
    }
}

/// A clock value as the moments' model places a row: its raw value, which
/// a window's edge and a lag are measured from by one subtraction, as the
/// other operators' edges are.
fn stamp_of(v: ClockValue) -> Stamp {
    match v {
        ClockValue::F64(x) => Stamp::Raw(x, 0.0),
        ClockValue::Ns(n) => Stamp::Ns(i128::from(n)),
        ClockValue::I64(i) => Stamp::Int(i128::from(i), 0.0),
    }
}

/// Whether `moments` holds one term for each moments operator of `ops`, in
/// operator order, each reading inputs a row has and each a model can run.
pub(super) fn check_moments(ops: &[OpDef], moments: &[MomentsDef]) -> Result<(), String> {
    let want: Vec<usize> = (0..ops.len())
        .filter(|&o| ops[o].stat == Stat::Moments)
        .collect();
    let have: Vec<usize> = moments.iter().map(|m| m.op).collect();
    if want != have {
        return Err(format!(
            "ewm_moments: the moments operators are {want:?}, but terms are given for \
             {have:?}. Give one term for each, in operator order."
        ));
    }
    for m in moments {
        if m.inputs.first() != Some(&ops[m.op].input) {
            return Err(format!(
                "operator {}: its first input is not its term's first column. Give the term's \
                 columns in the order of its inputs.",
                m.op
            ));
        }
        if m.cfg.terms.len() != 1 || m.cfg.n_columns() > m.inputs.len() {
            return Err(format!(
                "operator {}: its term reads columns it has no input for. Give an input for \
                 each column the term reads.",
                m.op
            ));
        }
        m.cfg.validate()?;
    }
    Ok(())
}

/// A row into a moments operator's model, at the raw clock value `now`:
/// learned where every column its term reads has a value, under the
/// decayed mass of the interval its value is held over -- from the last row
/// it learned, from before the stretch for the stretch's first, and never
/// longer than the window -- as `ewm_mean` and `ewm_var` weigh a value. A
/// row at a repeated stamp holds no interval and moves nothing.
pub(super) fn learn_moments(
    k: &KernelDef,
    ki: usize,
    ops: &[OpDef],
    defs: &[MomentsDef],
    g: &mut Group,
    values: &[f64],
    now: ClockValue,
) {
    for (mi, def) in defs.iter().enumerate() {
        if ops[def.op].kernel == ki {
            let memo = &mut g.queues[ki].decay;
            learn_one(k, def, &mut g.moments[mi], memo, values, now);
        }
    }
}

fn learn_one(
    k: &KernelDef,
    def: &MomentsDef,
    run: &mut MomentsRun,
    memo: &mut DecayMemo,
    values: &[f64],
    now: ClockValue,
) {
    run.x.clear();
    for &i in &def.inputs {
        let v = values[i];
        if !usable(v) {
            return;
        }
        run.x.push(v);
    }
    let (held, step) = match run.last {
        None => (k.window_size.unwrap_or(f64::INFINITY), 0.0),
        Some(last) => {
            let e = elapsed(now, last);
            (k.window_size.map_or(e, |w| e.min(w)), e)
        }
    };
    let m = k.mass_held_in(held, memo);
    if m > 0.0 {
        run.model.learn_at(&run.x, step, m, stamp_of(now));
        run.last = Some(now);
    }
}

/// Each moments operator of kernel `ki` (its operators `members`) read for
/// group `g` at its stamp, the policy time `tau` without a clock column,
/// into the stamp's values: what `flush_in` writes to the stamp's rows.
pub(super) fn read_moments(
    ki: usize,
    ops: &[OpDef],
    members: &[usize],
    defs: &[MomentsDef],
    g: &mut Group,
    tau: f64,
) {
    let at = g.stamp.unwrap_or(ClockValue::F64(tau));
    for (mi, def) in defs.iter().enumerate() {
        if ops[def.op].kernel == ki {
            let j = members
                .iter()
                .position(|&o| o == def.op)
                .expect("a kernel's member");
            g.stamp_values[ki][j] = g.moments[mi].read(at);
        }
    }
}

/// Whether moments operator `o` can run on its kernel: looking back, closed
/// on the right, and with a window under no decay.
pub(super) fn check_op(o: usize, op: &OpDef, k: &KernelDef) -> Result<(), String> {
    if op.stat != Stat::Moments {
        return Ok(());
    }
    if k.half_life.is_infinite() && k.window_size.is_none() {
        return Err(format!(
            "operator {o}: moments under half_life = inf need a window_size, since a value \
             held from before the first row would have no finite weight. Give a window_size, \
             or a finite half_life."
        ));
    }
    if k.direction == Direction::Forward || k.closed != Closed::Right {
        return Err(format!(
            "operator {o}: moments read only the rows up to each row, its own stamp's \
             included. Set closed to \"right\", looking back."
        ));
    }
    Ok(())
}

/// Whether a core has no moments operator: then its state writes none.
pub(super) fn none(m: &Arc<Vec<MomentsDef>>) -> bool {
    m.is_empty()
}

impl Windows {
    /// [`Self::new`] with the terms of its moments operators, one for each
    /// [`Stat::Moments`] operator, in operator order.
    ///
    /// # Errors
    ///
    /// As [`Self::new`]; and a moments operator without its one term, or a
    /// term its model refuses.
    pub fn with_moments(
        kernels: Vec<KernelDef>,
        ops: Vec<OpDef>,
        moments: Vec<MomentsDef>,
        clock_cfg: ClockCfg,
    ) -> Result<Self, String> {
        Self::with_cross_moments(kernels, ops, Vec::new(), moments, clock_cfg)
    }

    /// [`Self::with_moments`] with cross-group operators beside the window
    /// operators, after them in output order.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn with_cross_moments(
        kernels: Vec<KernelDef>,
        ops: Vec<OpDef>,
        cross: Vec<CrossDef>,
        moments: Vec<MomentsDef>,
        clock_cfg: ClockCfg,
    ) -> Result<Self, String> {
        check_moments(&ops, &moments).map_err(|e| defect(&e))?;
        let mut core = Self::build(kernels, ops, cross, clock_cfg).map_err(|e| defect(&e))?;
        core.moments = Arc::new(moments);
        Ok(core)
    }

    /// The moments operators' terms, in operator order.
    pub fn moments(&self) -> &[MomentsDef] {
        &self.moments
    }
}

#[cfg(test)]
mod tests {
    //! The operator against the model it runs, fed by hand: each row at the
    //! decayed mass of its held interval, read after it at its own stamp,
    //! as a row of weight 0 at that stamp reads the model.
    use rheon_core::{MomentKind, MomentLeg, MomentTerm, OnlineModel};

    use super::*;

    fn kernel(half_life: f64, window: Option<f64>) -> KernelDef {
        KernelDef {
            direction: Direction::Backward,
            half_life,
            window_size: window,
            window_ns: None,
            closed: Closed::Right,
        }
    }

    fn op() -> OpDef {
        OpDef {
            kernel: 0,
            stat: Stat::Moments,
            input: 0,
            min_samples: 1,
            partial: Partial::Keep,
            bias: false,
            quantile: None,
        }
    }

    fn clock() -> ClockCfg {
        ClockCfg {
            gap_cap: 1e9,
            ..ClockCfg::default()
        }
    }

    /// Clock steps of 0 (a repeated stamp) to 3, and values a gamma-like
    /// skewed column at a level of 100.
    fn rows(n: usize) -> Vec<(f64, f64)> {
        let mut s = 0x5eed_u64;
        let mut next = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut t = 0.0;
        (0..n)
            .map(|i| {
                if i > 0 && next() > 0.1 {
                    t += 0.1 + 2.9 * next();
                }
                let u: f64 = next();
                (t, 100.0 - (1.0 - u).ln() * (1.0 + next()))
            })
            .collect()
    }

    fn run(k: &KernelDef, term: MomentTerm, tol: Option<f64>, rows: &[(f64, f64)]) -> Vec<f64> {
        let def = MomentsDef {
            op: 0,
            inputs: vec![0],
            cfg: MomentsDef::cfg_of(k, term, tol),
        };
        let mut core = Windows::with_moments(vec![k.clone()], vec![op()], vec![def], clock())
            .expect("a moments core");
        let g = core.group(None);
        let mut out = Vec::new();
        for &(t, x) in rows {
            let row = RowIn {
                group: g,
                clock: Some(ClockValue::F64(t)),
                session: None,
                values: &[x],
                accept: true,
                lists: &[],
            };
            core.push(&row).expect("a step forward");
            out.extend(core.drain().values[0].iter().copied());
        }
        out.extend(core.finish().values[0].iter().copied());
        out
    }

    /// The model fed the rows by hand: a row at a new stamp at its held
    /// mass, `d` after the last, and read at its stamp once the stamp's rows
    /// are in; a row at a repeated stamp weighs nothing.
    fn by_hand(k: &KernelDef, term: MomentTerm, tol: Option<f64>, rows: &[(f64, f64)]) -> Vec<f64> {
        let mut m = rheon_core::EwmMoments::new(MomentsDef::cfg_of(k, term, tol)).unwrap();
        let mut memo = DecayMemo::default();
        let mut last: Option<f64> = None;
        let mut at = Vec::with_capacity(rows.len());
        for &(t, x) in rows {
            let (held, d) = match last {
                None => (k.window_size.unwrap_or(f64::INFINITY), 0.0),
                Some(p) if p == t => (0.0, 0.0),
                Some(p) => (k.window_size.map_or(t - p, |w| (t - p).min(w)), t - p),
            };
            let w = k.mass_held_in(held, &mut memo);
            if w > 0.0 {
                m.stamp_next(Stamp::Raw(t, 0.0));
                m.step(&[x], &[], d, w);
                last = Some(t);
            }
            at.push(m.clone());
        }
        // A stamp's rows all read the model as the stamp's last row left it.
        (0..rows.len())
            .map(|i| {
                let j = (i..rows.len())
                    .take_while(|&j| rows[j].0 == rows[i].0)
                    .last()
                    .unwrap();
                let s = at[j].predict_at(&[rows[j].1], 0.0, Stamp::Raw(rows[j].0, 0.0));
                s.pred[0]
            })
            .collect()
    }

    fn term(powers: Vec<(MomentLeg, u32)>, kind: MomentKind, bias: bool) -> MomentTerm {
        MomentTerm {
            name: "ewm_moments".into(),
            powers,
            kind,
            bias,
        }
    }

    #[test]
    fn the_operator_is_its_model_fed_the_held_masses_to_the_bit() {
        let rows = rows(400);
        let skew = || term(vec![(MomentLeg::of(0), 3)], MomentKind::Standardized, true);
        let kurt = || term(vec![(MomentLeg::of(0), 4)], MomentKind::Standardized, false);
        let ac = || {
            term(
                vec![(MomentLeg::of(0), 1), (MomentLeg::lagged(0, 2.0), 1)],
                MomentKind::Central,
                true,
            )
        };
        let cases: Vec<(KernelDef, MomentTerm, Option<f64>)> = vec![
            (kernel(20.0, None), skew(), None),
            (kernel(20.0, None), kurt(), None),
            (kernel(8.0, Some(30.0)), skew(), None),
            (kernel(f64::INFINITY, Some(25.0)), kurt(), None),
            (kernel(20.0, None), ac(), Some(0.5)),
            (kernel(12.0, Some(40.0)), ac(), None),
        ];
        for (k, t, tol) in cases {
            let got = run(&k, t.clone(), tol, &rows);
            let want = by_hand(&k, t, tol, &rows);
            assert_eq!(got.len(), rows.len());
            assert!(got.iter().filter(|v| v.is_finite()).count() > 300, "{k:?}");
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{k:?}, row {i}: {a} {b}");
            }
        }
    }

    #[test]
    fn a_core_refuses_moments_it_cannot_run() {
        let k = kernel(5.0, None);
        let t = term(vec![(MomentLeg::of(0), 3)], MomentKind::Central, true);
        let def = MomentsDef {
            op: 0,
            inputs: vec![0],
            cfg: MomentsDef::cfg_of(&k, t, None),
        };
        // No term, a term for no moments operator, a term past its inputs.
        assert!(Windows::with_moments(vec![k.clone()], vec![op()], vec![], clock()).is_err());
        let mean = OpDef {
            stat: Stat::Mean,
            ..op()
        };
        let one = || vec![def.clone()];
        assert!(Windows::with_moments(vec![k.clone()], vec![mean], one(), clock()).is_err());
        let wide = MomentsDef {
            inputs: vec![],
            ..def.clone()
        };
        assert!(Windows::with_moments(vec![k.clone()], vec![op()], vec![wide], clock()).is_err());
        // Closed on the left, looking ahead, and no decay without a window.
        for k in [
            KernelDef {
                closed: Closed::Left,
                ..k.clone()
            },
            KernelDef {
                direction: Direction::Forward,
                window_size: Some(3.0),
                ..k.clone()
            },
            kernel(f64::INFINITY, None),
        ] {
            assert!(Windows::with_moments(vec![k], vec![op()], one(), clock()).is_err());
        }
        assert!(Windows::with_moments(vec![k], vec![op()], one(), clock()).is_ok());
    }
}
