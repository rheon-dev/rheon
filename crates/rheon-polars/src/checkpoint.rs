//! A pipeline checkpoint: the states of a run's stages, one file, written
//! in full or not at all.
//!
//! A run that chains stateful stages -- a `refresh_time` grid, a window
//! run, a model bank -- keeps a state for each. Saved as separate files,
//! a crash between two saves leaves stages that stopped at different rows,
//! and a stream processor that commits its input's offsets after them has
//! no one moment to resume from. A checkpoint is the states as one file in
//! the envelope every state file shares (`envelope.rs`):
//!
//! ```text
//! magic    "rheon-state"
//! kind     "checkpoint"
//! version  1
//! stages   [{name: <text>, state: <bytes>}, ...]   in the order given
//! checksum the xxh3-64 of every byte before it
//! ```
//!
//! Each stage's `state` is that stage's own state file, byte for byte, as
//! its kind's `save_bytes` wrote it, so each loads back on its own through
//! its kind's loader, every version that loader reads included: a stage
//! is read back by name and handed to the kind that saved it. A kind added
//! later -- the state of a merge of streams, say -- is a stage like any
//! other, with no change to this layout. A checkpoint is not a stage.
//!
//! The file is written to a temporary beside the destination, synced, and
//! renamed over it, and the directory synced after (`atomic.rs`'s
//! `commit_durable`): a reader sees the old checkpoint or the new one,
//! never a part, and on Unix once the save returns the new one survives a
//! power loss. Windows has no call that syncs a directory: there the rename
//! is durable once NTFS writes its log, and a power loss before that brings
//! back the old checkpoint, whole. On Windows a rename that another
//! process's open handle refuses is tried again for about a second before
//! the save fails, leaving the old checkpoint as it was.

use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::{Deserialize, Serialize, Serializer};

use crate::envelope::{self, Kind};

/// The checkpoint's layout version.
pub const CHECKPOINT_VERSION: u32 = 1;

/// One stage of a checkpoint, as [`load_checkpoint`] reads it back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    /// The name it was saved under.
    pub name: String,
    /// What its state is, read from the state's own head.
    pub kind: Kind,
    /// The state file, byte for byte, which its kind's loader takes.
    pub state: Vec<u8>,
}

/// Bytes written as a msgpack binary, not an array of integers.
struct Bin<'a>(&'a [u8]);

impl Serialize for Bin<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(self.0)
    }
}

/// A msgpack binary read back.
struct ByteBuf(Vec<u8>);

impl<'de> Deserialize<'de> for ByteBuf {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ByteBuf;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a state's bytes")
            }
            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<ByteBuf, E> {
                Ok(ByteBuf(v.to_vec()))
            }
            fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<ByteBuf, E> {
                Ok(ByteBuf(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<ByteBuf, A::Error> {
                // At most 4 KiB reserved ahead of the bytes read (review
                // 2026-11-23, STATE-2).
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(4096));
                while let Some(b) = seq.next_element::<u8>()? {
                    out.push(b);
                }
                Ok(ByteBuf(out))
            }
        }
        d.deserialize_bytes(V)
    }
}

#[derive(Serialize)]
struct StageOut<'a> {
    name: &'a str,
    state: Bin<'a>,
}

#[derive(Serialize)]
struct FileOut<'a> {
    magic: &'a str,
    kind: &'a str,
    version: u32,
    stages: Vec<StageOut<'a>>,
    checksum: crate::checksum::Trailer,
}

#[derive(Deserialize)]
struct StageIn {
    name: String,
    state: ByteBuf,
}

#[derive(Deserialize)]
struct FileIn {
    stages: Vec<StageIn>,
}

const WHO: &str = "checkpoint";

/// A writer that keeps the first error it is handed, other than an
/// interruption `write_all` retries.
struct Kept<'a, W: Write> {
    inner: &'a mut W,
    error: Option<io::Error>,
}

impl<W: Write> Kept<'_, W> {
    fn keep(&mut self, e: &io::Error) {
        if e.kind() != io::ErrorKind::Interrupted && self.error.is_none() {
            self.error = Some(match e.raw_os_error() {
                Some(code) => io::Error::from_raw_os_error(code),
                None => io::Error::new(e.kind(), e.to_string()),
            });
        }
    }
}

impl<W: Write> Write for Kept<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf).inspect_err(|e| self.keep(e))
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush().inspect_err(|e| self.keep(e))
    }
}

/// The stages checked before a byte is written: at least one, each named,
/// no name twice, and each a state file of a kind this build reads (a
/// checkpoint is not one) whose checksum holds. Their kinds, in order.
fn check_stages(stages: &[(&str, &[u8])]) -> Result<Vec<Kind>, String> {
    if stages.is_empty() {
        return Err(format!(
            "{WHO}: no stage is given, so there is nothing to save. Give the state of each \
             stage of the run, by name."
        ));
    }
    let mut seen = std::collections::HashSet::new();
    let mut kinds = Vec::with_capacity(stages.len());
    for (name, state) in stages {
        if name.is_empty() {
            return Err(format!(
                "{WHO}: a stage's name is empty. Name each stage, such as \"bank\" or \
                 \"windows\", to read it back by."
            ));
        }
        if !seen.insert(*name) {
            return Err(format!(
                "{WHO}: the stage {name:?} is given twice. Give each stage once, under a name \
                 of its own."
            ));
        }
        let kind = envelope::kind_of(state).map_err(|why| {
            format!(
                "{WHO}: the stage {name:?} is not a state this version of rheon saves: {why}. \
                 Give the bytes a bank's save_bytes, or a run's save_state file, holds."
            )
        })?;
        if kind == Kind::Checkpoint {
            return Err(format!(
                "{WHO}: the stage {name:?} is itself a pipeline checkpoint. Give the states of \
                 its stages instead, each under a name of its own."
            ));
        }
        kinds.push(kind);
    }
    Ok(kinds)
}

fn file_out<'a>(
    stages: &'a [(&'a str, &'a [u8])],
    checksum: crate::checksum::Trailer,
) -> FileOut<'a> {
    FileOut {
        magic: envelope::MAGIC,
        kind: Kind::Checkpoint.tag(),
        version: CHECKPOINT_VERSION,
        stages: stages
            .iter()
            .map(|(name, state)| StageOut {
                name,
                state: Bin(state),
            })
            .collect(),
        checksum,
    }
}

/// The stages as one checkpoint, in memory: each `(name, state)`, its state
/// a state file as its kind's `save_bytes` wrote it.
///
/// # Errors
///
/// No stage, an empty or repeated name, or a state that is not one this
/// build reads, or is damaged.
pub fn checkpoint_bytes(stages: &[(&str, &[u8])]) -> Result<Vec<u8>, String> {
    check_stages(stages)?;
    let mut out = rmp_serde::to_vec_named(&file_out(stages, crate::checksum::Trailer::Patched))
        .map_err(|e| e.to_string())?;
    crate::checksum::seal(&mut out);
    Ok(out)
}

/// The stages as one checkpoint at `path`, replacing it whole or not at
/// all, and durably: written to a temporary beside it, hashed as it is
/// written, synced, renamed over it, and the directory synced. The stages'
/// bytes are written as they are given, with no copy of the whole file.
///
/// # Errors
///
/// What [`checkpoint_bytes`] refuses, as `InvalidInput`; or the
/// filesystem's own error, the old file left as it was.
pub fn save_checkpoint(path: &Path, stages: &[(&str, &[u8])]) -> io::Result<()> {
    check_stages(stages).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let (file, pending) = crate::atomic::AtomicFile::create(path)?;
    let mut buffered = BufWriter::with_capacity(1 << 20, file);
    let mut kept = Kept {
        inner: &mut buffered,
        error: None,
    };
    let (mut hashing, trailer) = crate::checksum::Hashing::new(&mut kept);
    let encoded = rmp_serde::encode::write_named(&mut hashing, &file_out(stages, trailer))
        .map_err(|e| e.to_string())
        .and_then(|()| hashing.flush().map_err(|e| e.to_string()));
    drop(hashing);
    // A failed write keeps the filesystem's own error, its kind and OS
    // code, which the encoder's message would lose.
    if let Err(msg) = encoded {
        return Err(kept.error.take().unwrap_or_else(|| io::Error::other(msg)));
    }
    // Flushed, then closed, so the commit reopens exactly one handle to sync.
    buffered
        .into_inner()
        .map_err(io::IntoInnerError::into_error)?;
    pending.commit_durable()
}

/// The stages of a checkpoint, in the order they were saved.
///
/// # Errors
///
/// Bytes that are not a checkpoint (another state's are named for what
/// they are), a damaged file, or a version this build does not read.
pub fn load_checkpoint(bytes: &[u8]) -> Result<Vec<Stage>, String> {
    const REMEDY: &str = "Load an undamaged copy of the checkpoint, or save it again.";
    let opened = envelope::open(bytes, Kind::Checkpoint).map_err(|r| {
        use envelope::Refused;
        match r {
            Refused::Damaged | Refused::Unsealed => {
                format!("{WHO}: the file is {}. {REMEDY}", r.what())
            }
            Refused::OtherKind(_) | Refused::UnknownKind(_) => format!(
                "{WHO}: the file is {}, not a pipeline checkpoint. Load it with the call that \
                 saved it, or pass a file save_checkpoint wrote.",
                r.what()
            ),
        }
    })?;
    let envelope::Opened::Enveloped { version } = opened else {
        return Err(format!(
            "{WHO}: the file is not a pipeline checkpoint. Pass a file save_checkpoint wrote."
        ));
    };
    if version > CHECKPOINT_VERSION {
        return Err(format!(
            "{WHO}: the file's version is {version}, newer than the {CHECKPOINT_VERSION} this \
             version of rheon reads. Upgrade rheon to load it."
        ));
    }
    if version < CHECKPOINT_VERSION {
        return Err(format!(
            "{WHO}: the file is damaged (its version {version} was never written). Load an \
             undamaged copy of the checkpoint, or save it again."
        ));
    }
    let file: FileIn = rmp_serde::from_slice(bytes).map_err(|e| {
        format!(
            "{WHO}: the file is damaged (it cannot be read: {e}). Load an undamaged copy of \
                 the checkpoint, or save it again."
        )
    })?;
    let views: Vec<(&str, &[u8])> = file
        .stages
        .iter()
        .map(|s| (s.name.as_str(), s.state.0.as_slice()))
        .collect();
    // Written only by a save that checked them, so a refusal is damage.
    let kinds = check_stages(&views).map_err(|e| {
        format!(
            "{WHO}: the file is damaged ({}). {REMEDY}",
            e.trim_start_matches("checkpoint: ")
        )
    })?;
    Ok(file
        .stages
        .into_iter()
        .zip(kinds)
        .map(|(s, kind)| Stage {
            name: s.name,
            kind,
            state: s.state.0,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct Fake<'a> {
        magic: &'a str,
        kind: &'a str,
        version: u32,
        body: u32,
        checksum: crate::checksum::Trailer,
    }

    fn state(kind: Kind, body: u32) -> Vec<u8> {
        let mut out = rmp_serde::to_vec_named(&Fake {
            magic: envelope::MAGIC,
            kind: kind.tag(),
            version: 1,
            body,
            checksum: crate::checksum::Trailer::Patched,
        })
        .unwrap();
        crate::checksum::seal(&mut out);
        out
    }

    /// Saved and loaded, every stage comes back by name, in order, its
    /// bytes whole; the file in memory and the file on disk are one.
    #[test]
    fn a_checkpoint_gives_back_its_stages() {
        let (a, b, c) = (
            state(Kind::RefreshTime, 1),
            state(Kind::Windows, 2),
            state(Kind::Bank, 3),
        );
        let stages: [(&str, &[u8]); 3] = [("grid", &a), ("windows", &b), ("bank", &c)];
        let bytes = checkpoint_bytes(&stages).unwrap();
        assert_eq!(envelope::head(&bytes).unwrap().kind, "checkpoint");
        let got = load_checkpoint(&bytes).unwrap();
        let want = [
            ("grid", Kind::RefreshTime, &a),
            ("windows", Kind::Windows, &b),
            ("bank", Kind::Bank, &c),
        ];
        assert_eq!(got.len(), 3);
        for (s, (name, kind, state)) in got.iter().zip(want) {
            assert_eq!((s.name.as_str(), s.kind, &s.state), (name, kind, state));
        }
        let dir = std::env::temp_dir().join(format!("rheon-checkpoint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("run.checkpoint");
        save_checkpoint(&path, &stages).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        // Replaced whole, and no temporary left beside it.
        save_checkpoint(&path, &stages[..1]).unwrap();
        assert_eq!(
            load_checkpoint(&std::fs::read(&path).unwrap())
                .unwrap()
                .len(),
            1
        );
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A refused save leaves the old checkpoint as it was.
    #[test]
    fn a_refused_save_touches_nothing() {
        let a = state(Kind::Bank, 1);
        let dir =
            std::env::temp_dir().join(format!("rheon-checkpoint-refused-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("run.checkpoint");
        save_checkpoint(&path, &[("bank", &a)]).unwrap();
        let before = std::fs::read(&path).unwrap();
        let mut broken = a.clone();
        broken[10] ^= 1;
        type Stages<'a> = &'a [(&'a str, &'a [u8])];
        let cases: [(Stages, &str); 5] = [
            (&[], "no stage is given"),
            (&[("", &a)], "name is empty"),
            (&[("bank", &a), ("bank", &a)], "is given twice"),
            (&[("bank", &broken)], "damaged"),
            (&[("inner", &before)], "itself a pipeline checkpoint"),
        ];
        for (stages, says) in cases {
            let err = save_checkpoint(&path, stages).unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
            assert!(err.to_string().contains(says), "{err}");
            assert!(checkpoint_bytes(stages).unwrap_err().contains(says));
        }
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A damaged checkpoint, another kind's state, an unknown stage kind
    /// and a newer version are each refused by name.
    #[test]
    fn a_checkpoint_is_refused_for_what_it_is() {
        let a = state(Kind::Bank, 1);
        let bytes = checkpoint_bytes(&[("bank", &a)]).unwrap();
        for i in 0..bytes.len() {
            let mut b = bytes.clone();
            b[i] ^= 0x40;
            assert!(load_checkpoint(&b).is_err(), "byte {i}");
        }
        let err = load_checkpoint(&a).unwrap_err();
        assert!(
            err.contains("a model bank's state, not a pipeline checkpoint"),
            "{err}"
        );
        let future = state(Kind::Checkpoint, 0);
        let mut v: rmpv::Value = rmpv::decode::read_value(&mut &future[..]).unwrap();
        if let rmpv::Value::Map(m) = &mut v {
            m[2].1 = 2.into();
        }
        let mut newer = Vec::new();
        rmpv::encode::write_value(&mut newer, &v).unwrap();
        let err = load_checkpoint(&crate::reseal(&newer)).unwrap_err();
        assert!(err.contains("newer than the 1"), "{err}");
    }

    /// Review 2026-11-23, STATE-1: a stage cut short after its head, with
    /// a flipped byte inside its checksum's key, or with a byte after its
    /// checksum, ends with no checksum at all (`Seal::Absent`), and was taken
    /// as a stage: the save replaced the good checkpoint with it. A stage
    /// from a file before the envelope is refused too: a stage is what this
    /// build's `save_bytes` writes, and such a file is loaded and saved
    /// again first.
    #[test]
    fn a_stage_without_its_checksum_is_refused_and_the_old_checkpoint_kept() {
        let a = state(Kind::Bank, 1);
        let dir =
            std::env::temp_dir().join(format!("rheon-checkpoint-unsealed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("run.checkpoint");
        save_checkpoint(&path, &[("bank", &a)]).unwrap();
        let before = std::fs::read(&path).unwrap();
        let n = a.len();
        let mut flipped = a.clone();
        flipped[n - 12] ^= 1; // inside "checksum", the trailer's key
        let mut trailing = a.clone();
        trailing.push(0xc0);
        #[derive(Serialize)]
        struct Old<'a> {
            magic: &'a str,
            version: u32,
        }
        let legacy = rmp_serde::to_vec_named(&Old {
            magic: "polars-online-bank",
            version: 2,
        })
        .unwrap();
        for (what, bytes) in [
            ("cut after its head", &a[..40]),
            ("cut before its checksum", &a[..n - 19]),
            ("a flip in its checksum's key", &flipped[..]),
            ("a byte after its checksum", &trailing[..]),
            ("a file before the envelope", &legacy[..]),
        ] {
            // Not a checksum that fails: none at all.
            assert!(
                crate::checksum::check(bytes) == crate::checksum::Seal::Absent,
                "{what}"
            );
            let err =
                save_checkpoint(&path, &[("bank", bytes)]).expect_err(&format!("{what}: saved"));
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{what}");
            assert!(checkpoint_bytes(&[("bank", bytes)]).is_err(), "{what}");
        }
        assert_eq!(std::fs::read(&path).unwrap(), before);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
