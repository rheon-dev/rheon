//! The checksum a bank's and a window run's state file ends with.
//!
//! A state file is one msgpack map. From bank schema 71 and windows state
//! version 11 its last entry is `checksum`, an eight-byte binary holding the
//! xxh3-64 of every byte of the file before those eight, little-endian. So a
//! sealed file ends with the nineteen bytes
//!
//! ```text
//! a8 "checksum" c4 08 <digest, 8 bytes>
//! ```
//!
//! and the check reads only those and hashes the rest: it runs before
//! anything is decoded. A damaged byte that still decodes would otherwise
//! load and change the output in silence, and some damage decodes into a
//! state that panics at the next chunk, or, in the Arrow IPC of the rows a
//! window run holds, asks for an allocation no machine has, which aborts the
//! process before any refusal can be said.
//!
//! The entry is last, not in the header with the versions, so that a bank
//! written to a file one group at a time ([`Hashing`]) needs no second pass
//! and no copy of the file to compute it: the digest is of what has gone
//! through the writer by the time the encoder reaches the last field.
//!
//! xxh3 is the hash polars already links (`polars-core` hashes with it), so
//! no library is added; it is the same function on every platform, and runs
//! at memory speed, so the check costs a read of the file.

use std::cell::RefCell;
use std::io::{self, Write};
use std::rc::Rc;

use serde::{Serialize, Serializer};
use xxhash_rust::xxh3::{Xxh3, xxh3_64};

/// What a sealed file holds before its digest: the key `checksum` as a
/// msgpack string of eight bytes, then the marker of a binary of eight.
const TAIL: &[u8] = b"\xa8checksum\xc4\x08";

/// The digest's width in bytes.
const DIGEST: usize = 8;

/// Whether `bytes` end with a checksum, and whether it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Seal {
    /// The file does not end with a checksum entry: a file of a schema
    /// before it, or one whose last bytes are damaged or cut.
    Absent,
    /// The checksum matches every byte before it.
    Holds,
    /// The file ends with a checksum entry that does not match the bytes
    /// before it: the file is damaged.
    Broken,
}

/// The seal of `bytes`, read from their last nineteen bytes and a hash of the
/// rest; nothing is decoded.
pub(crate) fn check(bytes: &[u8]) -> Seal {
    let n = bytes.len();
    if n < TAIL.len() + DIGEST || &bytes[n - DIGEST - TAIL.len()..n - DIGEST] != TAIL {
        return Seal::Absent;
    }
    let (body, digest) = bytes.split_at(n - DIGEST);
    let mut want = [0u8; DIGEST];
    want.copy_from_slice(digest);
    if xxh3_64(body) == u64::from_le_bytes(want) {
        Seal::Holds
    } else {
        Seal::Broken
    }
}

/// The bytes [`Hashing`] holds before it hashes them and hands them on, a
/// block at a time: a state file is written in many writes of a few bytes
/// (a Polars IPC buffer is a msgpack array of one-byte integers), and the
/// hasher's own cost per call was the save's (measured: a bank's save to a
/// file at 1,000 groups took 27% longer hashing each write as it came, 4%
/// in blocks).
const BLOCK: usize = 1 << 16;

/// What a [`Hashing`] writer and its [`Trailer`] share: the hash of the
/// blocks handed on, and the bytes since.
pub(crate) struct Pending {
    hasher: Xxh3,
    buf: Vec<u8>,
}

/// A writer that hashes what goes through it, for the [`Trailer`] that
/// shares its hasher. It holds up to a block: [`Write::flush`] hands the
/// rest on, and a caller flushes after the last field.
pub(crate) struct Hashing<W: Write> {
    inner: W,
    pending: Rc<RefCell<Pending>>,
}

impl<W: Write> Hashing<W> {
    /// `inner`, hashed, and the trailer to put last in the map written
    /// through it.
    pub(crate) fn new(inner: W) -> (Self, Trailer) {
        let pending = Rc::new(RefCell::new(Pending {
            hasher: Xxh3::new(),
            buf: Vec::with_capacity(BLOCK),
        }));
        (
            Self {
                inner,
                pending: Rc::clone(&pending),
            },
            Trailer::Streamed(pending),
        )
    }
}

/// The state a [`Hashing`] writer shares, which nothing else holds while a
/// write runs: the trailer reads it only between the encoder's writes.
fn held(pending: &RefCell<Pending>) -> io::Result<std::cell::RefMut<'_, Pending>> {
    pending
        .try_borrow_mut()
        .map_err(|_| io::Error::other("the checksum's state is borrowed during a write"))
}

impl<W: Write> Write for Hashing<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut p = held(&self.pending)?;
        p.buf.extend_from_slice(buf);
        if p.buf.len() >= BLOCK {
            let Pending { hasher, buf: block } = &mut *p;
            hasher.update(block);
            self.inner.write_all(block)?;
            block.clear();
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut p = held(&self.pending)?;
        let Pending { hasher, buf } = &mut *p;
        hasher.update(buf);
        self.inner.write_all(buf)?;
        buf.clear();
        self.inner.flush()
    }
}

/// The `checksum` field's value: the digest of every byte written through
/// its [`Hashing`] writer so far, with the binary marker the encoder writes
/// before it. Correct only as the last field of a map encoded by
/// `rmp_serde` through that writer, which is the one place it is put.
///
/// [`Trailer::Patched`] writes eight zero bytes instead, for an encoding
/// into memory, which [`seal`] then patches with the digest of the bytes
/// before them: one pass of the hash over the finished file, and no cost
/// on each of the encoder's writes (measured: a bank's `save_bytes` at
/// 1,000 groups, 4.2 MB, took 5.5 ms before the checksum, 9.4 ms through
/// [`Hashing`], whose cost is a few cycles on each of the millions of
/// writes of a byte or two).
pub(crate) enum Trailer {
    Streamed(Rc<RefCell<Pending>>),
    Patched,
}

impl Serialize for Trailer {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let digest = match self {
            // The encoder has written the key; the marker comes next, and
            // is covered, so the digest is of every byte before its own
            // eight. The borrow ends before the encoder writes through the
            // writer.
            Self::Streamed(pending) => {
                let p = pending.borrow();
                let mut h = p.hasher.clone();
                h.update(&p.buf);
                h.update(&TAIL[TAIL.len() - 2..]);
                h.digest()
            }
            Self::Patched => 0,
        };
        s.serialize_bytes(&digest.to_le_bytes())
    }
}

/// `out`, a map encoded with [`Trailer::Patched`] as its last field, given
/// the digest of every byte before its last eight.
pub(crate) fn seal(out: &mut [u8]) {
    let n = out.len();
    assert!(
        n >= TAIL.len() + DIGEST && out[n - DIGEST - TAIL.len()..n - DIGEST] == *TAIL,
        "seal: the bytes do not end with a checksum entry"
    );
    let digest = xxh3_64(&out[..n - DIGEST]);
    out[n - DIGEST..].copy_from_slice(&digest.to_le_bytes());
}

/// `bytes`, a sealed file of a map, with its checksum computed again: for a
/// test that edits a file's values on purpose and wants the checks behind
/// the checksum to see the edit. Bytes without a checksum entry are given
/// one, with the map's length raised by one.
#[doc(hidden)]
pub fn reseal(bytes: &[u8]) -> Vec<u8> {
    let n = bytes.len();
    let mut out = if n >= TAIL.len() + DIGEST && &bytes[n - DIGEST - TAIL.len()..n - DIGEST] == TAIL
    {
        bytes[..n - DIGEST].to_vec()
    } else {
        let mut out = bytes.to_vec();
        grow_map(&mut out);
        out.extend_from_slice(TAIL);
        out
    };
    let digest = xxh3_64(&out);
    out.extend_from_slice(&digest.to_le_bytes());
    out
}

/// Raise by one the length of the map `bytes` begin with, as a test's
/// re-encoded file needs before a field is appended.
fn grow_map(bytes: &mut Vec<u8>) {
    match bytes.first().copied() {
        Some(m @ 0x80..=0x8e) => bytes[0] = m + 1,
        Some(0x8f) => {
            bytes[0] = 0xde;
            bytes.splice(1..1, 16u16.to_be_bytes());
        }
        Some(0xde) => {
            let len = u16::from_be_bytes([bytes[1], bytes[2]]) + 1;
            bytes[1..3].copy_from_slice(&len.to_be_bytes());
        }
        Some(0xdf) => {
            let len = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) + 1;
            bytes[1..5].copy_from_slice(&len.to_be_bytes());
        }
        _ => panic!("reseal: the bytes are not a msgpack map"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct File<'a> {
        magic: &'a str,
        body: Vec<f64>,
        checksum: Trailer,
    }

    fn sealed() -> Vec<u8> {
        let mut out = Vec::new();
        let (mut w, checksum) = Hashing::new(&mut out);
        let file = File {
            magic: "m",
            body: vec![1.5, -0.0, f64::NAN],
            checksum,
        };
        rmp_serde::encode::write_named(&mut w, &file).unwrap();
        w.flush().unwrap();
        drop(w);
        out
    }

    /// The writer's digest is the one the check computes: the trailer's
    /// marker is the one the encoder writes, and the digest covers every
    /// byte before its own eight.
    #[test]
    fn a_sealed_file_holds_its_seal() {
        let bytes = sealed();
        assert_eq!(check(&bytes), Seal::Holds);
        let n = bytes.len();
        assert_eq!(&bytes[n - 8 - TAIL.len()..n - 8], TAIL);
        let digest = u64::from_le_bytes(bytes[n - 8..].try_into().unwrap());
        assert_eq!(digest, xxh3_64(&bytes[..n - 8]));
        assert_eq!(reseal(&bytes), bytes);
    }

    /// The patched trailer, sealed after, is the streamed one's bytes.
    #[test]
    fn a_patched_file_is_the_streamed_files_bytes() {
        let mut out = rmp_serde::to_vec_named(&File {
            magic: "m",
            body: vec![1.5, -0.0, f64::NAN],
            checksum: Trailer::Patched,
        })
        .unwrap();
        assert_eq!(check(&out), Seal::Broken);
        seal(&mut out);
        assert_eq!(out, sealed());
    }

    /// Every byte changed, by every bit, breaks the seal or removes it.
    #[test]
    fn every_byte_changed_breaks_the_seal() {
        let bytes = sealed();
        for i in 0..bytes.len() {
            for bit in 0..8 {
                let mut b = bytes.clone();
                b[i] ^= 1 << bit;
                assert_ne!(check(&b), Seal::Holds, "byte {i}, bit {bit}");
            }
        }
        for cut in 0..bytes.len() {
            assert_ne!(check(&bytes[..cut]), Seal::Holds, "cut at {cut}");
        }
    }

    /// A map written without the trailer is given one by [`reseal`], with
    /// its length raised, at each width of map header.
    #[test]
    fn reseal_seals_a_file_written_without_one() {
        for fields in [1usize, 14, 15, 16, 300] {
            let map: std::collections::BTreeMap<String, u8> =
                (0..fields).map(|i| (format!("k{i:03}"), 1)).collect();
            let bytes = rmp_serde::to_vec(&map).unwrap();
            let sealed = reseal(&bytes);
            assert_eq!(check(&sealed), Seal::Holds);
            let v: rmpv::Value = rmpv::decode::read_value(&mut &sealed[..]).unwrap();
            let m = v.as_map().unwrap();
            assert_eq!(m.len(), fields + 1);
            assert_eq!(m.last().unwrap().0.as_str(), Some("checksum"));
        }
    }
}
