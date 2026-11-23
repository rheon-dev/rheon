//! `ClockCfg` with its session gap tagged. The core type's own derive
//! writes `Some(SessionGap::Reset)` as a bare unit, which msgpack reads back
//! as `None`: a saved `session_gap = "reset"` would resume as no session
//! gap at all. The bank never saves a `ClockCfg` -- it rebuilds one from
//! its specs -- so this state is the first to need the form.

use rheon_core::{ClockCfg, OnClockReset, SessionGap};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize)]
struct Form {
    gap_cap: f64,
    on_clock_reset: OnClockReset,
    session_gap: GapForm,
    min_backwards_jump: f64,
}

#[derive(Serialize, Deserialize)]
enum GapForm {
    Off,
    Gap(f64),
    Reset,
}

pub fn serialize<S: Serializer>(c: &ClockCfg, s: S) -> Result<S::Ok, S::Error> {
    Form {
        gap_cap: c.gap_cap,
        on_clock_reset: c.on_clock_reset,
        session_gap: match c.session_gap {
            None => GapForm::Off,
            Some(SessionGap::Gap(g)) => GapForm::Gap(g),
            Some(SessionGap::Reset) => GapForm::Reset,
        },
        min_backwards_jump: c.min_backwards_jump,
    }
    .serialize(s)
}

pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ClockCfg, D::Error> {
    let f = Form::deserialize(d)?;
    Ok(ClockCfg {
        gap_cap: f.gap_cap,
        on_clock_reset: f.on_clock_reset,
        session_gap: match f.session_gap {
            GapForm::Off => None,
            GapForm::Gap(g) => Some(SessionGap::Gap(g)),
            GapForm::Reset => Some(SessionGap::Reset),
        },
        min_backwards_jump: f.min_backwards_jump,
    })
}
