//! A merge's state file, in the envelope every state file shares
//! (`envelope.rs`), kind `"merge"`:
//!
//! ```text
//! magic        "rheon-state"
//! kind         "merge"
//! version      1
//! clock        the clock column, which a load must match
//! by           the group column or nil, which a load must match
//! clock_dtype  the clock's dtype, as a message names it, which a load must match
//! names        each input's name, in the call's order, which a load must match
//! inputs       per input: last (its last row taken: {Int: i64} | {Float: f64} | nil),
//!              restarts (how many times it started again), and held (the rows
//!              held at a tie, as Arrow IPC; absent when none)
//! groups       [(key, last clock)]: the merged rows' last clock in each group,
//!              sorted by key; the one key "" without `by`
//! carry        under `carry` alone (PLAN 556): columns (the columns carried,
//!              which a load must match) and groups ([(key, per column: nil
//!              or {restarts, at, value})], sorted by key: each group's last
//!              non-null value of each column, the clock it was read at and
//!              its input's restarts then; the value as `refresh_time`'s held
//!              row keeps a value, a `Cell`)
//! checksum     the xxh3-64 of every byte before it
//! ```
//!
//! A temporal clock's stamp is its physical integer in the column's own
//! unit, so a state resumes on the clock dtype it was saved with alone.

use std::io::Cursor;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use super::{
    Block, Carried, Merge, MergeConfig, Mode, WHO, cell_as, cells_series, dtype_text, float_key,
    float_of,
};
use crate::bank::GroupKey;
use crate::refresh::{Bin, Cell};

/// The layout's version.
pub(crate) const MERGE_VERSION: u32 = 1;

/// A clock value as the state keeps it: an integer clock's, or a temporal
/// one's physical integer, exactly; a float clock's as itself.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
enum Stamp {
    Int(i64),
    Float(f64),
}

/// One input's state.
#[derive(Debug, Serialize, Deserialize)]
struct InputState {
    last: Option<Stamp>,
    restarts: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    held: Option<Bin>,
}

/// A carried column's last value in one group, as the state keeps it.
#[derive(Debug, Serialize, Deserialize)]
struct CarriedState {
    restarts: u64,
    at: Stamp,
    value: Cell,
}

/// The values `carry` holds (PLAN 556): the columns, and each group's.
#[derive(Debug, Serialize, Deserialize)]
struct CarryState {
    columns: Vec<String>,
    groups: Vec<(GroupKey, Vec<Option<CarriedState>>)>,
}

#[derive(Serialize, Deserialize)]
struct MergeFile {
    magic: String,
    kind: String,
    version: u32,
    clock: String,
    by: Option<String>,
    clock_dtype: String,
    names: Vec<String>,
    inputs: Vec<InputState>,
    groups: Vec<(GroupKey, Stamp)>,
    /// Under `carry` alone, so a merge without it writes the bytes it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    carry: Option<CarryState>,
    /// The xxh3-64 of every byte before it (`checksum.rs`), last; read from
    /// the file's last bytes, never by serde.
    #[serde(skip_deserializing, skip_serializing_if = "Option::is_none")]
    checksum: Option<crate::checksum::Trailer>,
}

const START_OVER: &str = "Load an undamaged copy of the state file, or leave out load_state to \
                          start over.";

fn damaged(what: &str) -> String {
    format!("{WHO}: the state is damaged: {what}. {START_OVER}")
}

impl Merge {
    fn stamp(&self, key: i64) -> Stamp {
        match self.mode {
            Mode::Float => Stamp::Float(float_of(key)),
            Mode::Int | Mode::Temporal { .. } => Stamp::Int(key),
        }
    }

    fn key_of(&self, s: Stamp) -> Option<i64> {
        match (self.mode, s) {
            (Mode::Float, Stamp::Float(x)) if x.is_finite() => Some(float_key(x)),
            (Mode::Int | Mode::Temporal { .. }, Stamp::Int(k)) => Some(k),
            _ => None,
        }
    }

    /// The state as bytes: each input's last row taken and its restarts,
    /// the rows held at a tie, and the merged rows' last clock in each
    /// group. Versioned msgpack in the envelope, one state one file.
    ///
    /// # Errors
    ///
    /// The held rows cannot be written as Arrow IPC.
    pub fn save_bytes(&self) -> Result<Vec<u8>, String> {
        let mut inputs = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            // At the end of a run that holds, every row still queued is
            // held; under a slice, only the rows a loaded state held.
            let held_blocks: Vec<&Block> = input
                .queue
                .iter()
                .filter(|b| self.held_end || b.first_row.is_none())
                .collect();
            let taken = match (self.held_end, input.queue.back()) {
                (true, Some(b)) => Some(b.last()).map(|(r, k)| (k, r)),
                _ => input.taken,
            };
            let held = if held_blocks.is_empty() {
                None
            } else {
                let mut rows = DataFrame::empty_with_schema(&input.schema);
                for b in held_blocks {
                    rows.vstack_mut(&b.df.slice(b.pos as i64, b.len() - b.pos))
                        .map_err(|e| e.to_string())?;
                }
                // One chunk and its strings compacted, so the bytes are the
                // rows' alone, however the chunks fell.
                rows.rechunk_mut();
                crate::windows_frame::compact_views(&mut rows).map_err(|e| e.to_string())?;
                let mut out = Vec::new();
                IpcWriter::new(&mut out)
                    .finish(&mut rows)
                    .map_err(|e| e.to_string())?;
                Some(Bin(out))
            };
            inputs.push(InputState {
                last: taken.map(|(k, _)| self.stamp(k)),
                restarts: taken.map_or(0, |(_, r)| r),
                held,
            });
        }
        let mut groups: Vec<(GroupKey, Stamp)> = self
            .group_keys
            .iter()
            .zip(&self.group_last)
            .filter_map(|(k, last)| last.map(|l| (GroupKey(k.clone()), self.stamp(l))))
            .collect();
        groups.sort_by(|a, b| a.0.cmp(&b.0));
        let carry = if self.carry.is_empty() {
            None
        } else {
            let mut held: Vec<(GroupKey, Vec<Option<CarriedState>>)> = Vec::new();
            for (gid, key) in self.group_keys.iter().enumerate() {
                let mut row: Vec<Option<CarriedState>> = Vec::with_capacity(self.carry.len());
                for c in &self.carry {
                    row.push(match c.slots[gid] {
                        None => None,
                        Some(v) => Some(CarriedState {
                            restarts: v.restarts,
                            at: self.stamp(v.key),
                            value: Cell::of(&c.store.slice(i64::from(v.at), 1).into_column())
                                .map_err(|e| e.to_string())?,
                        }),
                    });
                }
                if row.iter().any(Option::is_some) {
                    held.push((GroupKey(key.clone()), row));
                }
            }
            held.sort_by(|a, b| a.0.cmp(&b.0));
            Some(CarryState {
                columns: self.carry.iter().map(|c| c.name.clone()).collect(),
                groups: held,
            })
        };
        let file = MergeFile {
            magic: crate::envelope::MAGIC.into(),
            kind: crate::envelope::Kind::Merge.tag().into(),
            version: MERGE_VERSION,
            clock: self.cfg.clock.clone(),
            by: self.cfg.by.clone(),
            clock_dtype: dtype_text(&self.clock_dtype),
            names: self.inputs.iter().map(|i| i.name.clone()).collect(),
            inputs,
            groups,
            carry,
            checksum: Some(crate::checksum::Trailer::Patched),
        };
        let mut out = rmp_serde::to_vec_named(&file).map_err(|e| e.to_string())?;
        crate::checksum::seal(&mut out);
        Ok(out)
    }

    /// [`Self::save_bytes`] to `path`, replacing it whole or not at all.
    ///
    /// # Errors
    ///
    /// The state cannot be written.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = self.save_bytes().map_err(std::io::Error::other)?;
        crate::atomic::write(path, &bytes)
    }

    /// A merge that goes on from a saved state: the inputs, the clock and
    /// `by` must be the ones it was saved with. Rows are counted from the
    /// start of the new inputs.
    ///
    /// # Errors
    ///
    /// What [`Self::new`] refuses; bytes that are not a merge's state, a
    /// version this build does not read, other inputs, clock or `by`, or a
    /// state that does not hold together (a damaged one).
    pub fn load_bytes(
        bytes: &[u8],
        names: Vec<String>,
        schemas: Vec<Schema>,
        cfg: MergeConfig,
    ) -> PolarsResult<Self> {
        let mut m = Self::new(names, schemas, cfg)?;
        m.restore(bytes)
            .map_err(|e| polars_err!(ComputeError: "{e}"))?;
        Ok(m)
    }

    fn restore(&mut self, bytes: &[u8]) -> Result<(), String> {
        use crate::envelope::{Kind, Opened, Refused};
        let opened = crate::envelope::open(bytes, Kind::Merge).map_err(|r| match r {
            Refused::Damaged | Refused::Unsealed => {
                format!("{WHO}: the state is {}. {START_OVER}", r.what())
            }
            Refused::OtherKind(Kind::Checkpoint) => format!(
                "{WHO}: the state is a pipeline checkpoint, not a merge's state. Read its stages \
                 with rheon.polars.stream.load_checkpoint and pass the merge's stage as \
                 load_state."
            ),
            Refused::OtherKind(_) | Refused::UnknownKind(_) => format!(
                "{WHO}: the state is {}, not a merge's. Pass a state file a merge saved, or leave \
                 out load_state to start over.",
                r.what()
            ),
        })?;
        match opened {
            Opened::Enveloped {
                version: MERGE_VERSION,
            } => {}
            Opened::Enveloped { version } => {
                return Err(format!(
                    "{WHO}: the state's version is {version}, but this version of rheon reads \
                     version {MERGE_VERSION}. Load the state with the version of rheon that \
                     saved it, or leave out load_state to start over."
                ));
            }
            Opened::Legacy { .. } => {
                return Err(format!(
                    "{WHO}: the state is not a rheon state. Pass a state file a merge saved, or \
                     leave out load_state to start over."
                ));
            }
        }
        let file: MergeFile = rmp_serde::from_slice(bytes).map_err(|e| {
            format!(
                "{WHO}: the state cannot be read ({e}), so it is damaged. Pass a state file a \
                 merge saved, or leave out load_state to start over."
            )
        })?;
        let names: Vec<&str> = self.inputs.iter().map(|i| i.name.as_str()).collect();
        if file.names != names {
            return Err(format!(
                "{WHO}: the state was saved merging the inputs {:?}, but this call merges {:?}. \
                 Give the inputs the state was saved with, named and in the order it was saved \
                 with, or leave out load_state to start over.",
                file.names, names
            ));
        }
        if file.clock != self.cfg.clock || file.by != self.cfg.by {
            return Err(format!(
                "{WHO}: the state was saved with clock = {:?} and by = {}, but this call gives \
                 {:?} and {}. Give the clock and by the state was saved with, or leave out \
                 load_state to start over.",
                file.clock,
                file.by.as_ref().map_or("None".into(), |b| format!("{b:?}")),
                self.cfg.clock,
                self.cfg
                    .by
                    .as_ref()
                    .map_or("None".into(), |b| format!("{b:?}")),
            ));
        }
        let dtype = dtype_text(&self.clock_dtype);
        if file.clock_dtype != dtype {
            return Err(format!(
                "{WHO}: the state was saved on a clock of dtype {}, but this call's clock \
                 column {:?} is {dtype}. Cast the clock to the dtype the state was saved on, or \
                 leave out load_state to start over.",
                file.clock_dtype, self.cfg.clock
            ));
        }
        let carried: Vec<String> = self.carry.iter().map(|c| c.name.clone()).collect();
        let saved: &[String] = file.carry.as_ref().map_or(&[], |c| &c.columns);
        if saved != carried.as_slice() {
            return Err(format!(
                "{WHO}: the state was saved carrying {saved:?}, but this call carries \
                 {carried:?}. Give the carry the state was saved with, or leave out load_state \
                 to start over."
            ));
        }
        if file.inputs.len() != self.inputs.len() {
            return Err(damaged("it holds another number of inputs than it names"));
        }
        // The groups first, so their ids are the state's order.
        for (key, stamp) in &file.groups {
            let k = self
                .key_of(*stamp)
                .ok_or_else(|| damaged("a group's clock is not of the state's clock"))?;
            let gid = match (&self.cfg.by, &key.0) {
                (None, Some(s)) if s.is_empty() && file.groups.len() == 1 => 0,
                (None, _) => return Err(damaged("an ungrouped state holds a group")),
                (Some(_), Some(s)) => {
                    if self.gid_of.contains_key(s) {
                        return Err(damaged("a group is held twice"));
                    }
                    let id = self.new_group(Some(s.clone()));
                    self.gid_of.insert(s.clone(), id);
                    id
                }
                (Some(_), None) => {
                    if self.null_gid.is_some() {
                        return Err(damaged("a group is held twice"));
                    }
                    let id = self.new_group(None);
                    self.null_gid = Some(id);
                    id
                }
            };
            self.group_last[gid as usize] = Some(k);
        }
        if let Some(carry) = file.carry {
            self.restore_carry(carry)?;
        }
        for (i, st) in file.inputs.into_iter().enumerate() {
            let taken = match st.last {
                None if st.held.is_some() || st.restarts != 0 => {
                    return Err(damaged(
                        "an input holds rows, or restarts, but no last clock",
                    ));
                }
                None => None,
                Some(s) => Some((
                    self.key_of(s)
                        .ok_or_else(|| damaged("an input's clock is not of the state's clock"))?,
                    st.restarts,
                )),
            };
            self.inputs[i].taken = taken;
            self.inputs[i].read = taken;
            let Some(Bin(held)) = st.held else { continue };
            let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                IpcReader::new(Cursor::new(held)).finish()
            }));
            let df = match read {
                Ok(Ok(df)) => df,
                Ok(Err(e)) => return Err(damaged(&format!("its held rows cannot be read ({e})"))),
                Err(_) => return Err(damaged("its held rows cannot be read")),
            };
            let name = &self.inputs[i].name;
            if df.schema().as_ref() != &self.inputs[i].schema {
                return Err(format!(
                    "{WHO}: the state holds rows of input {name:?} with the columns {:?}, but \
                     this call's input {name:?} has the columns {:?}. Give each input the \
                     columns it was saved with, or leave out load_state to start over.",
                    df.schema(),
                    self.inputs[i].schema
                ));
            }
            let (k, restarts) = taken.expect("checked above");
            // Every held row is at the input's last clock: a tie.
            let (keys, _) = self
                .keys(i, &df)
                .map_err(|e| damaged(&format!("the rows it holds have invalid clocks ({e})")))?;
            if df.height() == 0 || keys.iter().any(|&x| x != k) {
                return Err(damaged(
                    "the rows it holds are not at their input's last clock",
                ));
            }
            let gids = self
                .gids(&df)
                .map_err(|e| damaged(&format!("the rows it holds have unreadable groups ({e})")))?;
            self.inputs[i].queue.push_back(Block {
                df,
                keys,
                gids,
                bumps: Vec::new(),
                first_row: None,
                pos: 0,
                restarts,
                next_bump: 0,
            });
            self.heap.push(std::cmp::Reverse((restarts, k, i)));
            self.max_read = self.max_read.max(Some((restarts, k)));
        }
        Ok(())
    }
}

impl Merge {
    /// The values a state carried, into each group's slots: every group a
    /// value is carried in has merged rows, so is among the state's groups.
    fn restore_carry(&mut self, carry: CarryState) -> Result<(), String> {
        let mut prev: Option<&GroupKey> = None;
        for (key, row) in &carry.groups {
            if prev.is_some_and(|p| p >= key) {
                return Err(damaged("its carried values are not one a group, in order"));
            }
            prev = Some(key);
            let gid = match (&self.cfg.by, &key.0) {
                (None, Some(s)) if s.is_empty() => Some(0),
                (None, _) => None,
                (Some(_), Some(s)) => self.gid_of.get(s).copied(),
                (Some(_), None) => self.null_gid,
            }
            .filter(|&g| self.group_last[g as usize].is_some())
            .ok_or_else(|| damaged("it carries a value in a group with no merged rows"))?;
            if row.len() != self.carry.len() {
                return Err(damaged(
                    "a group carries values of another number of columns",
                ));
            }
            for (ci, v) in row.iter().enumerate() {
                let Some(v) = v else { continue };
                let k = self.key_of(v.at).ok_or_else(|| {
                    damaged("a carried value's clock is not of the state's clock")
                })?;
                let col = &self.carry[ci];
                if matches!(v.value, Cell::Null) {
                    return Err(damaged("it carries a null"));
                }
                // Read back as the column's dtype now, not on the row that needs it.
                cell_as(&v.value, &col.name, &col.dtype).map(drop).map_err(|_| {
                    format!(
                        "{WHO}: the state carries a value of column {:?} that this call's {:?}, \
                         of dtype {}, cannot hold. Give each input the columns it was saved \
                         with, or leave out load_state to start over.",
                        col.name,
                        col.name,
                        dtype_text(&col.dtype)
                    )
                })?;
                let col = &mut self.carry[ci];
                col.slots[gid as usize] = Some(Carried {
                    restarts: v.restarts,
                    key: k,
                    at: col.store.len() as IdxSize,
                });
                let one = cells_series(std::slice::from_ref(&v.value), &col.name, &col.dtype)
                    .map_err(|e| e.to_string())?;
                col.store.append(&one).map_err(|e| e.to_string())?;
            }
        }
        for c in &mut self.carry {
            c.store = c.store.rechunk();
        }
        Ok(())
    }
}

#[cfg(test)]
mod carry_state_tests {
    //! A carried state that does not hold together is refused as damaged,
    //! each way it can fail to (PLAN 556).

    use super::super::tests::{cfg, drive, merge_of, names};
    use super::*;

    fn input() -> DataFrame {
        df!("t" => [1.0, 2.0], "g" => ["a", "b"], "x" => [Some(5i64), None]).unwrap()
    }

    fn carry_cfg() -> MergeConfig {
        let mut c = cfg("t");
        c.by = Some("g".into());
        c.carry = vec!["x".into()];
        c
    }

    fn saved() -> MergeFile {
        let a = input();
        let mut m = merge_of(std::slice::from_ref(&a), carry_cfg());
        let _ = drive(&mut m, &[vec![a]], None, false);
        rmp_serde::from_slice(&m.save_bytes().unwrap()).unwrap()
    }

    fn load(mut f: MergeFile) -> String {
        f.checksum = Some(crate::checksum::Trailer::Patched);
        let mut bytes = rmp_serde::to_vec_named(&f).unwrap();
        crate::checksum::seal(&mut bytes);
        Merge::load_bytes(
            &bytes,
            names(1),
            vec![input().schema().as_ref().clone()],
            carry_cfg(),
        )
        .err()
        .map_or_else(String::new, |e| e.to_string())
    }

    #[test]
    fn the_saved_state_loads() {
        let f = saved();
        let carry = f.carry.as_ref().unwrap();
        assert_eq!(carry.columns, ["x"]);
        assert_eq!(carry.groups.len(), 1, "group b has no value yet");
        assert_eq!(load(f), "");
    }

    #[test]
    fn each_way_a_carried_state_falls_apart_is_damaged() {
        let mut f = saved();
        let row = f.carry.as_mut().unwrap().groups.pop().unwrap();
        f.carry.as_mut().unwrap().groups = vec![
            (GroupKey(Some("a".into())), vec![None]),
            (GroupKey(Some("a".into())), vec![None]),
        ];
        assert!(load(f).contains("not one a group, in order"));
        let mut f = saved();
        f.carry.as_mut().unwrap().groups[0].0 = GroupKey(Some("zz".into()));
        assert!(load(f).contains("a group with no merged rows"));
        let mut f = saved();
        f.carry.as_mut().unwrap().groups[0].1.push(None);
        assert!(load(f).contains("another number of columns"));
        let mut f = saved();
        f.carry.as_mut().unwrap().groups[0].1[0]
            .as_mut()
            .unwrap()
            .at = Stamp::Int(1);
        assert!(load(f).contains("a carried value's clock is not of the state's clock"));
        let mut f = saved();
        f.carry.as_mut().unwrap().groups[0].1[0]
            .as_mut()
            .unwrap()
            .value = Cell::Null;
        assert!(load(f).contains("it carries a null"));
        // A group without a value of any column is legal, and loads.
        let mut f = saved();
        f.carry.as_mut().unwrap().groups = vec![(row.0, vec![None])];
        assert_eq!(load(f), "");
    }

    /// Review 2026-11-23, MERGE-4 (2, 3): a crafted state with `restarts` at
    /// `u64::MAX` must not wrap/panic on the next restart, and corrupted held
    /// rows whose clocks are null or out of order must report `"the state is
    /// damaged"`, not an input-order error.
    #[test]
    fn crafted_restarts_overflow_and_corrupted_held_rows_are_refused_cleanly() {
        let a = df!("t" => [1.0, 2.0], "x" => [10i64, 20]).unwrap();
        let b = df!("t" => [2.0], "x" => [30i64]).unwrap();
        let mut c = cfg("t");
        c.restart = Some(super::super::Restart::Float(0.0));
        c.restart_text = Some("0".into());
        let mut m = merge_of(&[a.clone(), b.clone()], c.clone());
        let _ = drive(&mut m, &[vec![a.clone()], vec![b.clone()]], None, true);
        let mut file: MergeFile = rmp_serde::from_slice(&m.save_bytes().unwrap()).unwrap();
        // (2) restarts at u64::MAX must refuse on a step back, not panic/wrap.
        file.inputs[0].restarts = u64::MAX;
        file.inputs[0].held = None;
        file.inputs[1].held = None;
        file.checksum = Some(crate::checksum::Trailer::Patched);
        let mut bytes = rmp_serde::to_vec_named(&file).unwrap();
        crate::checksum::seal(&mut bytes);
        let mut loaded = Merge::load_bytes(
            &bytes,
            names(2),
            vec![a.schema().as_ref().clone(), b.schema().as_ref().clone()],
            c.clone(),
        )
        .unwrap();
        let step_back = df!("t" => [1.0], "x" => [99i64]).unwrap();
        let err = loaded.push(0, step_back).unwrap_err().to_string();
        assert!(err.contains("overflowing"), "{err}");

        // (3) corrupted held rows with a null clock or step back report damaged state.
        for bad_df in [
            df!("t" => [None::<f64>], "x" => [1i64]).unwrap(),
            df!("t" => [1.0], "x" => [1i64]).unwrap(),
        ] {
            let mut f: MergeFile = rmp_serde::from_slice(&m.save_bytes().unwrap()).unwrap();
            let mut ipc = Vec::new();
            let mut copy = bad_df;
            IpcWriter::new(&mut ipc).finish(&mut copy).unwrap();
            f.inputs[0].held = Some(Bin(ipc));
            f.checksum = Some(crate::checksum::Trailer::Patched);
            let mut raw = rmp_serde::to_vec_named(&f).unwrap();
            crate::checksum::seal(&mut raw);
            let err = Merge::load_bytes(
                &raw,
                names(2),
                vec![a.schema().as_ref().clone(), b.schema().as_ref().clone()],
                cfg("t"),
            )
            .err()
            .expect("corrupted held rows must be refused")
            .to_string();
            assert!(err.contains("the state is damaged"), "{err}");
        }
    }
}
