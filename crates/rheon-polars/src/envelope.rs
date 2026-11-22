//! The envelope every state file of this library is written in.
//!
//! A state file is one msgpack map. Its first three entries are the
//! envelope's head, in this order:
//!
//! ```text
//! magic    "rheon-state"
//! kind     "bank" | "windows" | "refresh_time" | "checkpoint"
//! version  the kind's own layout version
//! ```
//!
//! and its last entry is `checksum`, the xxh3-64 of every byte before it
//! (`checksum.rs`). The entries between are the kind's body, laid out as
//! its `version` says. So one reader tells every file apart: a bank's state
//! handed to a window run, or a pipeline checkpoint handed to a bank, is
//! named for what it is rather than failing on a field it lacks, and a
//! file of a newer version is refused by its number before its body is
//! decoded.
//!
//! The head is read by hand from the file's first bytes ([`head`]), which
//! costs nothing however large the file: serde would walk the whole map to
//! find three keys, since a map's keys can come in any order. A writer puts
//! them first because each kind's struct declares them first.
//!
//! A kind's versions are its own: a bank's is the models' schema
//! (`rheon_core::SCHEMA_VERSION`), which moves whenever a model's state
//! does; a window run's and a `refresh_time` sampler's move with their own
//! layouts alone. A new kind -- the state of a merge of streams, say -- is
//! a new name here and a body of its own, and a pipeline checkpoint carries
//! it as a stage with no change to the checkpoint's layout.
//!
//! Files written before the envelope began with a magic string of their
//! kind's own (`polars-online-bank`, `polars-online windows`,
//! `polars-online refresh_time`) and a version beside it; each kind keeps
//! the loader that reads them ([`legacy_kind`] names their kind).

/// The envelope's magic string: the value of every state file's first entry.
pub(crate) const MAGIC: &str = "rheon-state";

/// What a state file holds, as its `kind` entry names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A model bank: its specs and every group's models.
    Bank,
    /// A window run (`with_windows`): its windows' sums and the rows held.
    Windows,
    /// A `refresh_time` sampler: each group's grid part-way.
    RefreshTime,
    /// A pipeline checkpoint: the states of several stages, one file.
    Checkpoint,
}

impl Kind {
    /// Every kind this build reads.
    pub const ALL: [Self; 4] = [
        Self::Bank,
        Self::Windows,
        Self::RefreshTime,
        Self::Checkpoint,
    ];

    /// The `kind` entry's value.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Bank => "bank",
            Self::Windows => "windows",
            Self::RefreshTime => "refresh_time",
            Self::Checkpoint => "checkpoint",
        }
    }

    /// The kind a `kind` entry names, if this build reads it.
    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.tag() == tag)
    }

    /// What the file is, in a message's words.
    pub const fn noun(self) -> &'static str {
        match self {
            Self::Bank => "a model bank's state",
            Self::Windows => "a with_windows run's state",
            Self::RefreshTime => "a refresh_time run's state",
            Self::Checkpoint => "a pipeline checkpoint",
        }
    }
}

/// The envelope's head, as [`head`] reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Head<'a> {
    /// The `kind` entry, as written: a kind this build may not know.
    pub kind: &'a str,
    pub version: u32,
}

/// A cursor over the first bytes of a msgpack map.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn byte(&mut self) -> Option<u8> {
        let b = *self.bytes.get(self.at)?;
        self.at += 1;
        Some(b)
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(n)?;
        let out = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(out)
    }

    fn be(&mut self, n: usize) -> Option<u64> {
        Some(
            self.take(n)?
                .iter()
                .fold(0u64, |acc, &b| (acc << 8) | u64::from(b)),
        )
    }

    /// A map's length, from its marker.
    fn map_len(&mut self) -> Option<u64> {
        match self.byte()? {
            m @ 0x80..=0x8f => Some(u64::from(m & 0x0f)),
            0xde => self.be(2),
            0xdf => self.be(4),
            _ => None,
        }
    }

    /// A string, as UTF-8.
    fn str(&mut self) -> Option<&'a str> {
        let len = match self.byte()? {
            m @ 0xa0..=0xbf => u64::from(m & 0x1f),
            0xd9 => self.be(1)?,
            0xda => self.be(2)?,
            0xdb => self.be(4)?,
            _ => return None,
        };
        std::str::from_utf8(self.take(usize::try_from(len).ok()?)?).ok()
    }

    /// A non-negative integer that fits a `u32`.
    fn u32(&mut self) -> Option<u32> {
        let v = match self.byte()? {
            m @ 0x00..=0x7f => u64::from(m),
            0xcc => self.be(1)?,
            0xcd => self.be(2)?,
            0xce => self.be(4)?,
            0xcf => self.be(8)?,
            _ => return None,
        };
        u32::try_from(v).ok()
    }

    /// The next entry's key, which must be `key`.
    fn key(&mut self, key: &str) -> Option<()> {
        (self.str()? == key).then_some(())
    }
}

/// The envelope's head, read from the first bytes of `bytes`, or `None`
/// when they do not begin with one: a file written before the envelope, or
/// bytes that are no state file. Nothing else is read.
pub(crate) fn head(bytes: &[u8]) -> Option<Head<'_>> {
    let mut c = Cursor { bytes, at: 0 };
    // The head's three entries and the checksum, at least.
    if c.map_len()? < 4 {
        return None;
    }
    c.key("magic")?;
    if c.str()? != MAGIC {
        return None;
    }
    c.key("kind")?;
    let kind = c.str()?;
    c.key("version")?;
    let version = c.u32()?;
    Some(Head { kind, version })
}

/// The kind of a file written before the envelope, by the magic string its
/// first entry held, or `None` for bytes that hold none of them.
pub(crate) fn legacy_kind(bytes: &[u8]) -> Option<Kind> {
    let mut c = Cursor { bytes, at: 0 };
    c.map_len()?;
    c.key("magic")?;
    match c.str()? {
        "polars-online-bank" => Some(Kind::Bank),
        "polars-online windows" => Some(Kind::Windows),
        "polars-online refresh_time" => Some(Kind::RefreshTime),
        _ => None,
    }
}

/// The kind of any state file this build reads, enveloped or not, or why
/// it is none: for a caller that takes a state of any kind, as a pipeline
/// checkpoint does. The file's checksum is checked first.
pub(crate) fn kind_of(bytes: &[u8]) -> Result<Kind, String> {
    if crate::checksum::check(bytes) == crate::checksum::Seal::Broken {
        return Err("it is damaged (its bytes do not match the checksum it was saved with)".into());
    }
    match head(bytes) {
        Some(h) => Kind::from_tag(h.kind).ok_or_else(|| {
            format!(
                "it is a state of the kind {:?}, which this version of rheon does not read",
                h.kind
            )
        }),
        None => legacy_kind(bytes).ok_or_else(|| "it is not a rheon state".to_string()),
    }
}

/// How a loader of `want` opens `bytes`, before decoding any of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Opened {
    /// An enveloped file of the kind wanted, sealed, of this version.
    Enveloped { version: u32 },
    /// A file written before the envelope, which the kind's own loader
    /// reads; `sealed` when it ends with a checksum that holds.
    Legacy { sealed: bool },
}

/// Why [`open`] refused `bytes`, for each loader to say in its own words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Refused {
    /// The checksum the file ends with does not match its bytes.
    Damaged,
    /// An enveloped file without its checksum: cut short, or damaged.
    Unsealed,
    /// An enveloped file of another kind, which this build reads.
    OtherKind(Kind),
    /// An enveloped file of a kind this build does not know.
    UnknownKind(String),
}

impl Refused {
    /// The refusal as a sentence's middle: "the state file is ...".
    pub(crate) fn what(&self) -> String {
        match self {
            Self::Damaged => {
                "damaged (its bytes do not match the checksum it was saved with)".to_string()
            }
            Self::Unsealed => "damaged (the checksum it ends with is missing or cut)".to_string(),
            Self::OtherKind(k) => k.noun().to_string(),
            Self::UnknownKind(tag) => format!(
                "a state of the kind {tag:?}, which this version of rheon does not read: a \
                 newer version wrote it"
            ),
        }
    }
}

/// Open `bytes` as a state of `want`: the checksum checked, then the head.
/// The version is the caller's to check, against its own range.
pub(crate) fn open(bytes: &[u8], want: Kind) -> Result<Opened, Refused> {
    let seal = crate::checksum::check(bytes);
    if seal == crate::checksum::Seal::Broken {
        return Err(Refused::Damaged);
    }
    let sealed = seal == crate::checksum::Seal::Holds;
    let Some(h) = head(bytes) else {
        return Ok(Opened::Legacy { sealed });
    };
    match Kind::from_tag(h.kind) {
        Some(k) if k == want => {}
        Some(k) => return Err(Refused::OtherKind(k)),
        None => return Err(Refused::UnknownKind(h.kind.to_string())),
    }
    if !sealed {
        return Err(Refused::Unsealed);
    }
    Ok(Opened::Enveloped { version: h.version })
}

/// Files as a build before the envelope wrote them, made from a current
/// one, for the tests of each kind's loader of them.
#[cfg(test)]
pub(crate) mod legacy {
    /// `bytes`, an enveloped file, rewritten in the layout before the
    /// envelope: the head's `magic` set to `magic`, its `kind` gone, its
    /// `version` renamed `version_key` and set to `version`, each key of
    /// `renames` renamed, and the checksum kept only if `sealed`.
    pub(crate) fn rewrite(
        bytes: &[u8],
        magic: &str,
        version_key: &str,
        version: u32,
        renames: &[(&str, &str)],
        sealed: bool,
    ) -> Vec<u8> {
        let mut v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let rmpv::Value::Map(m) = &mut v else {
            panic!("a state file is a map")
        };
        m.retain(|(k, _)| !matches!(k.as_str(), Some("kind" | "checksum")));
        for (k, val) in m.iter_mut() {
            match k.as_str() {
                Some("magic") => *val = magic.into(),
                Some("version") => {
                    *k = version_key.into();
                    *val = version.into();
                }
                Some(name) => {
                    if let Some((_, to)) = renames.iter().find(|(from, _)| *from == name) {
                        *k = (*to).into();
                    }
                }
                None => {}
            }
        }
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &v).unwrap();
        if sealed { crate::reseal(&out) } else { out }
    }

    /// A bank file of `schema` (71 or earlier), sealed from 71.
    pub(crate) fn bank(bytes: &[u8], schema: u32) -> Vec<u8> {
        rewrite(
            bytes,
            "polars-online-bank",
            "schema_version",
            schema,
            &[],
            schema >= 71,
        )
    }

    /// A window run's state of `version` (11 or earlier), sealed from 11.
    pub(crate) fn windows(bytes: &[u8], version: u32) -> Vec<u8> {
        rewrite(
            bytes,
            "polars-online windows",
            "version",
            version,
            &[],
            version >= 11,
        )
    }

    /// A `refresh_time` state of `version` (1 or 2), never sealed.
    pub(crate) fn refresh(bytes: &[u8], version: u32) -> Vec<u8> {
        rewrite(
            bytes,
            "polars-online refresh_time",
            "version",
            version,
            &[("clock_kind", "kind")],
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct File<'a> {
        magic: &'a str,
        kind: &'a str,
        version: u32,
        body: Vec<f64>,
        checksum: crate::checksum::Trailer,
    }

    fn sealed(kind: &str, version: u32) -> Vec<u8> {
        let mut out = rmp_serde::to_vec_named(&File {
            magic: MAGIC,
            kind,
            version,
            body: vec![1.5, -0.0],
            checksum: crate::checksum::Trailer::Patched,
        })
        .unwrap();
        crate::checksum::seal(&mut out);
        out
    }

    /// The head is read at every width msgpack writes a version in.
    #[test]
    fn the_head_is_read_at_every_width_of_its_version() {
        for version in [0, 1, 12, 127, 128, 255, 256, 65_535, 65_536, u32::MAX] {
            let bytes = sealed("bank", version);
            assert_eq!(
                head(&bytes),
                Some(Head {
                    kind: "bank",
                    version
                }),
                "{version}"
            );
            assert_eq!(open(&bytes, Kind::Bank), Ok(Opened::Enveloped { version }));
        }
    }

    /// A map of any width, and a kind's name of any length, is read.
    #[test]
    fn the_head_is_read_in_a_map_of_any_width() {
        for fields in [0usize, 20, 70_000] {
            let mut map: Vec<(String, rmpv::Value)> = vec![
                ("magic".into(), MAGIC.into()),
                ("kind".into(), "x".repeat(40).into()),
                ("version".into(), 3.into()),
            ];
            map.extend((0..fields).map(|i| (format!("f{i}"), rmpv::Value::from(i as u64))));
            map.push(("checksum".into(), rmpv::Value::Binary(vec![0; 8])));
            let v = rmpv::Value::Map(map.into_iter().map(|(k, v)| (k.into(), v)).collect());
            let mut bytes = Vec::new();
            rmpv::encode::write_value(&mut bytes, &v).unwrap();
            let h = head(&bytes).unwrap();
            assert_eq!((h.kind.len(), h.version), (40, 3), "{fields}");
        }
    }

    /// Each kind is refused by a loader of another, by name; an unknown
    /// kind by its tag; a broken seal and a missing one as damage.
    #[test]
    fn a_file_of_another_kind_is_named() {
        for want in Kind::ALL {
            for got in Kind::ALL {
                let r = open(&sealed(got.tag(), 1), want);
                if got == want {
                    assert_eq!(r, Ok(Opened::Enveloped { version: 1 }));
                } else {
                    assert_eq!(r, Err(Refused::OtherKind(got)));
                }
            }
            assert_eq!(Kind::from_tag(want.tag()), Some(want));
        }
        assert_eq!(
            open(&sealed("merge", 1), Kind::Bank),
            Err(Refused::UnknownKind("merge".into()))
        );
        let mut broken = sealed("bank", 1);
        let n = broken.len();
        broken[n - 3] ^= 1;
        assert_eq!(open(&broken, Kind::Bank), Err(Refused::Damaged));
        let mut unsealed = sealed("bank", 1);
        unsealed.truncate(unsealed.len() - 19);
        unsealed[0] -= 1;
        assert_eq!(open(&unsealed, Kind::Bank), Err(Refused::Unsealed));
    }

    /// Bytes that do not begin with the head are not enveloped, wherever
    /// they part from it; the files before the envelope are named by their
    /// magic.
    #[test]
    fn bytes_without_the_head_are_legacy_or_nothing() {
        let bytes = sealed("windows", 12);
        for cut in 0..bytes.len().min(40) {
            assert_eq!(head(&bytes[..cut]), None, "cut at {cut}");
        }
        let mut other = bytes.clone();
        other[8] ^= 1; // inside the magic
        assert_eq!(head(&other), None);
        for (magic, kind) in [
            ("polars-online-bank", Kind::Bank),
            ("polars-online windows", Kind::Windows),
            ("polars-online refresh_time", Kind::RefreshTime),
        ] {
            #[derive(Serialize)]
            struct Old<'a> {
                magic: &'a str,
                version: u32,
            }
            let old = rmp_serde::to_vec_named(&Old { magic, version: 2 }).unwrap();
            assert_eq!(head(&old), None);
            assert_eq!(legacy_kind(&old), Some(kind));
            assert_eq!(open(&old, kind), Ok(Opened::Legacy { sealed: false }));
            assert_eq!(kind_of(&old), Ok(kind));
        }
        assert_eq!(legacy_kind(&bytes), None);
        assert_eq!(kind_of(&bytes), Ok(Kind::Windows));
        assert!(kind_of(b"\x81\xa5magic\xa3abc").is_err());
        assert!(kind_of(b"not a state").is_err());
    }
}
