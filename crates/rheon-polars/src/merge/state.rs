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
//! checksum     the xxh3-64 of every byte before it
//! ```
//!
//! A temporal clock's stamp is its physical integer in the column's own
//! unit, so a state resumes on the clock dtype it was saved with alone.

use std::io::Cursor;

use polars::prelude::*;
use serde::{Deserialize, Serialize};

use super::{Block, Merge, MergeConfig, Mode, WHO, dtype_text, float_key, float_of};
use crate::bank::GroupKey;
use crate::refresh::Bin;

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
            let (keys, _) = self.keys(i, &df).map_err(|e| e.to_string())?;
            if df.height() == 0 || keys.iter().any(|&x| x != k) {
                return Err(damaged(
                    "the rows it holds are not at their input's last clock",
                ));
            }
            let gids = self.gids(&df).map_err(|e| e.to_string())?;
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
