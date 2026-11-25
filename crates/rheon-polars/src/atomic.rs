//! Replace a file, or leave it exactly as it was.
//!
//! `fs::write` truncates the destination and then writes into it, so an
//! interrupted write -- a kill, a full disk, a quota -- leaves a truncated
//! file *and* destroys what was there. For a state file in a `--load-state` loop
//! that is the difference between "this save failed, keep going" and "start
//! the stream over": measured, with the write cut a third of the way through,
//! the file is unloadable ("failed to fill whole buffer") and the previous
//! state is gone.
//!
//! So write a temporary sibling, then rename it over the destination. Rust's
//! `fs::rename` "renames a file or directory to a new name, replacing the
//! original file if `to` already exists" on both target platforms (`rename`
//! on Unix, `MoveFileExW` on Windows), and a rename either happens or does
//! not, so a reader sees the old file or the new one and never a half of
//! either.
//!
//! Three limits, said here rather than found (review 2026-09-12, S12). On
//! Windows the rename fails while another process holds the destination
//! open without `FILE_SHARE_DELETE` -- a Python `open(path, "rb")`, a
//! viewer, an indexer -- and the save reports that error once the rename
//! has been tried for about a second (`rename_over`); Rust's own `File`
//! shares delete, so two writers in this library never meet it. The rename
//! is not flushed to its directory, so on POSIX a power loss just after it
//! can leave the old file: "old or new" holds, and "new" is durable once the
//! directory is, which `AtomicFile::commit_durable` waits for (a pipeline
//! checkpoint's save). And a process killed between
//! creating the temporary and renaming it leaves `.name.tmpPID-seq` beside
//! the destination, which nothing removes.
//!
//! What a rename does not carry over is said too. The new file is the
//! temporary, so it has the owner and group a new file gets, not the old
//! file's. Its mode is the old file's, copied onto the temporary before
//! the rename on Unix (a no-op elsewhere): created with the process's
//! default, a state file made private came back world-readable, 600
//! becoming 644 under a umask of 022 (review round 4, PA4). A destination
//! that is not there yet takes the default, as `fs::write` gives a new file.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Numbers the temporaries this process creates, so that two writers of one
/// destination in one process never share one. The pid alone was the name
/// once, and two threads saving the same state file at the same moment --
/// which a plan used twice in one query does, `lf.rheon.fit_predict(..,
/// save_state=)` under a self-join or `collect_all` -- both created and
/// wrote the same temporary, and the rename published whichever mixture
/// resulted.
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// The temporary for `dest`: a sibling, so the rename stays on one
/// filesystem (`fs::rename` does not cross mount points), named by pid and
/// sequence number, so no two writers -- processes or threads -- share one.
fn temp_name(dest: &Path, seq: u64) -> io::Result<PathBuf> {
    // The path is not in the message: every caller names it itself, and
    // one that did too read it twice.
    let not_a_file = || io::Error::new(io::ErrorKind::InvalidInput, "not a file path");
    // `out/` names a folder, though `file_name` reads `out` from it: the
    // temporary was written beside it and the rename over it failed at the
    // end, after a run's every row (review 2026-11-23, CLI-5).
    if names_a_folder(dest) {
        return Err(not_a_file());
    }
    let name = dest.file_name().ok_or_else(not_a_file)?;
    Ok(dest.with_file_name(format!(
        ".{}.tmp{}-{seq}",
        name.to_string_lossy(),
        std::process::id()
    )))
}

/// Whether `path` ends in a separator, which names a folder whether or not
/// one is there.
pub(crate) fn names_a_folder(path: &Path) -> bool {
    path.as_os_str()
        .as_encoded_bytes()
        .last()
        .is_some_and(|&b| std::path::is_separator(char::from(b)))
}

/// How many taken names [`AtomicFile::create`] passes over before it gives
/// up with the filesystem's `AlreadyExists`: more than any process leaves
/// by accident, so only a directory filled on purpose reaches it.
const TEMP_TRIES: u32 = 1024;

/// `tmp`, created new, with `dest`'s permission bits where it has some.
#[cfg(unix)]
fn create_new(tmp: &Path, dest: &Path) -> io::Result<File> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    if let Ok(m) = fs::metadata(dest) {
        options.mode((m.permissions().mode() & 0o777) | 0o200);
    }
    options.open(tmp)
}

#[cfg(not(unix))]
fn create_new(tmp: &Path, _dest: &Path) -> io::Result<File> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(tmp)
}

/// A file being written next to `dest`, renamed over it by [`Self::commit`].
///
/// Dropped without committing -- an error on the way, an early `?` -- the
/// temporary is removed and the destination is untouched.
pub(crate) struct AtomicFile {
    dest: PathBuf,
    tmp: PathBuf,
    /// Cleared by `commit`, so `Drop` knows whether to clean up.
    pending: bool,
}

impl AtomicFile {
    /// Create the temporary and hand back the handle to write into. The
    /// caller owns the handle (a `ParquetWriter` consumes one), so `commit`
    /// takes the paths rather than the file.
    ///
    /// The temporary is created new (`O_EXCL`): a name already taken -- a
    /// file or a link planted there, or left by a process of the same pid
    /// -- is passed over for the next, so nothing is written through a link
    /// and renamed over the destination (review 2026-11-23, IO-2). On Unix
    /// it is created with the destination's permission bits (and the
    /// owner's write, to write it), so a private file's new bytes are never
    /// readable by others while they are written.
    pub(crate) fn create(dest: &Path) -> io::Result<(File, Self)> {
        let dest = resolve_symlink(dest);
        let mut tries = 0;
        loop {
            let tmp = temp_name(&dest, NEXT_TEMP.fetch_add(1, Ordering::Relaxed))?;
            match create_new(&tmp, &dest) {
                Ok(file) => {
                    return Ok((
                        file,
                        Self {
                            dest,
                            tmp,
                            pending: true,
                        },
                    ));
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists && tries < TEMP_TRIES => {
                    tries += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Flush the temporary to disk and rename it over the destination.
    ///
    /// The caller's handle is gone by now -- a `ParquetWriter` consumes the
    /// one it is given -- so the sync reopens the file rather than asking for
    /// it back. Reopening is microseconds; the sync is what costs, and
    /// without it the rename can reach the disk while the contents have not,
    /// so a power loss replaces a good state file with an empty one.
    ///
    /// It is not free. On Apple targets std's `sync_all` is
    /// `fcntl(F_FULLFSYNC)` -- and so is `sync_data`, so there is no cheaper
    /// honest option -- which flushes the drive's own cache: measured here,
    /// writing 396 KiB costs 0.10 ms, plain `fsync` 0.13 ms, `F_FULLFSYNC`
    /// 4.0 ms. A save is therefore ~4 ms rather than ~0.5 ms, and a caller
    /// saving after every small chunk should save less often instead. The
    /// alternative is a state file that is not there after the crash it
    /// exists for.
    pub(crate) fn commit(mut self) -> io::Result<()> {
        File::options().write(true).open(&self.tmp)?.sync_all()?;
        keep_mode(&self.dest, &self.tmp)?;
        rename_over(&self.tmp, &self.dest)?;
        self.pending = false;
        Ok(())
    }

    /// [`Self::commit`], and the rename made durable as well: on Unix the
    /// directory is synced after it, so a power loss just after the call
    /// returns cannot bring the old file back. A pipeline checkpoint needs
    /// this, since the caller commits its input's offsets once the call
    /// returns, and an old checkpoint under new offsets would skip the rows
    /// between. It costs a second sync: on macOS arm64 a directory's
    /// `sync_all` (`F_FULLFSYNC`) took 4.0 ms, as a small file's did (PLAN
    /// 542, the median of 21 after a small write); on Windows
    /// the standard library cannot sync a directory, so there the rename is
    /// durable once the filesystem writes its log, and a power loss before
    /// that brings back the old file, whole.
    pub(crate) fn commit_durable(self) -> io::Result<()> {
        let dir = self
            .dest
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        self.commit()?;
        sync_dir(&dir)
    }
}

/// Sync a directory, so a rename inside it is on the disk.
#[cfg(unix)]
fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> io::Result<()> {
    Ok(())
}

/// How many times a rename that Windows refused for a moment is tried, and
/// the pause before the first retry, doubled after each: about 0.8 s in
/// all.
#[cfg_attr(not(windows), allow(dead_code))]
const RENAME_TRIES: u32 = 7;
#[cfg_attr(not(windows), allow(dead_code))]
const RENAME_PAUSE: std::time::Duration = std::time::Duration::from_millis(12);

/// Rename `tmp` over `dest`, replacing it.
///
/// On Unix `rename` replaces the destination atomically, and its failures
/// are not passing, so it is tried once. On Windows std's `rename`
/// replaces it too (`MoveFileExW` with `MOVEFILE_REPLACE_EXISTING`, or a
/// POSIX-semantics rename where the filesystem has one), but fails with an
/// access-denied or sharing-violation error while another process holds
/// the destination open without sharing delete: a virus scanner or a
/// search indexer that opens each new file for a moment, as much as a
/// reader that keeps it. So there it is tried again, a few times over
/// about a second, before the error is returned. The temporary stays
/// whole meanwhile; the destination is never removed first, which would
/// leave no file at all if the process died between the two steps.
fn rename_over(tmp: &Path, dest: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        retry_while_held(|| fs::rename(tmp, dest), std::thread::sleep)
    }
    #[cfg(not(windows))]
    {
        fs::rename(tmp, dest)
    }
}

/// Whether a failed rename is Windows refusing it while another process
/// holds the destination: `ERROR_ACCESS_DENIED` (5) or
/// `ERROR_SHARING_VIOLATION` (32).
#[cfg_attr(not(windows), allow(dead_code))]
fn held_by_another(e: &io::Error) -> bool {
    matches!(e.raw_os_error(), Some(5 | 32))
}

/// `rename` tried until it succeeds, fails otherwise than
/// [`held_by_another`], or has been tried [`RENAME_TRIES`] times, with
/// `pause` between the tries. Generic in both, so the rule is tested on
/// every platform.
#[cfg_attr(not(windows), allow(dead_code))]
fn retry_while_held(
    mut rename: impl FnMut() -> io::Result<()>,
    mut pause: impl FnMut(std::time::Duration),
) -> io::Result<()> {
    let mut wait = RENAME_PAUSE;
    let mut tries = 1;
    loop {
        match rename() {
            Err(e) if held_by_another(&e) && tries < RENAME_TRIES => {
                pause(wait);
                wait *= 2;
                tries += 1;
            }
            other => return other,
        }
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        if self.pending {
            let _ = fs::remove_file(&self.tmp);
        }
    }
}

/// Give the temporary the mode of the file it replaces, on Unix, so a save
/// does not change who may read the state (review round 4, PA4): renamed
/// over a file made private (`chmod 600`), the temporary published it with
/// the process's default mode, 644 under a umask of 022. Called once the
/// temporary is written and synced, since a read-only mode would refuse the
/// sync's reopening for write. A destination that is not there yet, or
/// cannot be read, keeps the default, as `fs::write` would give a new file.
/// Elsewhere a no-op: there is no Unix mode to copy.
#[cfg(unix)]
fn keep_mode(dest: &Path, tmp: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    match fs::metadata(dest) {
        Ok(m) => fs::set_permissions(
            tmp,
            // The permission bits alone: setuid, setgid and sticky are not
            // a state file's to carry, and setting them can be refused
            // (review 2026-11-23, IO-2).
            fs::Permissions::from_mode(m.permissions().mode() & 0o777),
        ),
        Err(_) => Ok(()),
    }
}

#[cfg(not(unix))]
fn keep_mode(_dest: &Path, _tmp: &Path) -> io::Result<()> {
    Ok(())
}

/// Write `bytes` to `path`, atomically.
pub(crate) fn write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let (mut file, pending) = AtomicFile::create(path)?;
    file.write_all(bytes)?;
    drop(file); // Closed, so `commit` reopens exactly one handle to sync.
    pending.commit()
}

/// `fs::write` follows a symlink and writes its target; a rename would
/// replace the link itself. Resolve first, so switching to a rename changes
/// when the destination changes and nothing else. A dangling link has no
/// target to resolve to, and is replaced.
fn resolve_symlink(path: &Path) -> PathBuf {
    match fs::symlink_metadata(path) {
        Ok(m) if m.file_type().is_symlink() => {
            fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
        }
        _ => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The retry Windows takes: a held destination is tried again, with
    /// the pause doubled, until the rename succeeds or the tries run out;
    /// any other error is returned at once.
    #[test]
    fn a_rename_refused_while_held_is_tried_again() {
        let held = || io::Error::from_raw_os_error(32);
        let mut pauses = Vec::new();
        let mut left = 3;
        let r = retry_while_held(
            || {
                if left > 0 {
                    left -= 1;
                    Err(held())
                } else {
                    Ok(())
                }
            },
            |d| pauses.push(d.as_millis()),
        );
        assert!(r.is_ok());
        assert_eq!(pauses, [12, 24, 48]);

        let mut tries = 0;
        let mut pauses = 0;
        let r = retry_while_held(
            || {
                tries += 1;
                Err(io::Error::from_raw_os_error(5))
            },
            |_| pauses += 1,
        );
        assert_eq!(r.unwrap_err().raw_os_error(), Some(5));
        assert_eq!((tries, pauses), (RENAME_TRIES, RENAME_TRIES - 1));

        let mut tries = 0;
        let r = retry_while_held(
            || {
                tries += 1;
                Err(io::Error::from(io::ErrorKind::NotFound))
            },
            |_| panic!("no pause for another error"),
        );
        assert_eq!(r.unwrap_err().kind(), io::ErrorKind::NotFound);
        assert_eq!(tries, 1);
    }

    /// A durable commit replaces the file as a plain one does, in a
    /// directory named or in the working one.
    #[test]
    fn a_durable_commit_replaces_the_file() {
        let dir = std::env::temp_dir().join(format!("rheon-atomic-durable-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("checkpoint");
        fs::write(&dest, b"old").unwrap();
        let (mut f, pending) = AtomicFile::create(&dest).unwrap();
        f.write_all(b"new").unwrap();
        drop(f);
        pending.commit_durable().unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"new");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_failed_write_leaves_the_destination_alone() {
        let dir = std::env::temp_dir().join(format!("rheon-atomic-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("state.msgpack");
        fs::write(&dest, b"the good state").unwrap();

        // Make creating the temporary fail, by putting a file where it
        // goes. That is the same shape as a full disk: the failure happens
        // on the temporary, before the destination is touched. A taken name
        // is passed over for the next, up to `TEMP_TRIES` of them, and the
        // name carries a process-wide sequence number, so block that many
        // and a few hundred more: other tests in this binary take some in
        // between. A file, not a directory: Windows refuses to create a
        // file where a directory stands with ERROR_ACCESS_DENIED
        // (`PermissionDenied`), not as a name taken, so a directory there
        // ends the search at once, and on Windows the names passed over
        // went untested (CI's Windows legs, 2026-11-24). A file there is
        // `AlreadyExists` on every OS.
        let first = NEXT_TEMP.load(Ordering::Relaxed);
        let blockers: Vec<PathBuf> = (first..first + u64::from(TEMP_TRIES) + 512)
            .map(|n| temp_name(&dest, n).unwrap())
            .collect();
        for b in &blockers {
            fs::write(b, b"taken").unwrap();
        }

        let err = write(&dest, b"the new state").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists, "{err}");
        assert_eq!(fs::read(&dest).unwrap(), b"the good state", "{err}");

        for b in &blockers {
            fs::remove_file(b).unwrap();
        }
        write(&dest, b"the new state").unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"the new state");
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            1,
            "the temporary was left behind"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Two threads saving to one destination at the same moment: each
    /// write succeeds, the file is always exactly one of the two contents,
    /// and no temporary is left. With the pid-only name they shared a
    /// temporary, and the file could be a mixture.
    #[test]
    fn two_writers_of_one_destination_do_not_share_a_temporary() {
        let dir = std::env::temp_dir().join(format!("rheon-atomic-two-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("state.msgpack");
        let a = vec![b'a'; 200_000];
        let b = vec![b'b'; 200_000];

        let (a, b, dest) = (&a, &b, &dest);
        std::thread::scope(|s| {
            for bytes in [a, b] {
                s.spawn(move || {
                    for _ in 0..50 {
                        write(dest, bytes).unwrap();
                        let got = fs::read(dest).unwrap();
                        assert!(&got == a || &got == b, "a mixture of the two writes");
                    }
                });
            }
        });

        let got = fs::read(dest).unwrap();
        assert!(&got == a || &got == b);
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            1,
            "a temporary was left behind"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    /// A save keeps the mode of the file it replaces (review round 4, PA4).
    /// The temporary was created with the process's default mode and renamed
    /// over the destination, so a state file a user had made private came
    /// back world-readable: 600 became 644 under a umask of 022. A file that
    /// is not there yet takes the default, as `fs::write` gives it; and a
    /// read-only one stays read-only, which is why the mode goes on after
    /// the temporary is written and synced.
    #[cfg(unix)]
    #[test]
    fn a_save_keeps_the_destinations_mode() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("rheon-atomic-mode-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("state.msgpack");
        let mode = |p: &Path| {
            format!(
                "{:o}",
                fs::metadata(p).unwrap().permissions().mode() & 0o7777
            )
        };
        write(&dest, b"first").unwrap();
        let fresh = dir.join("plain.msgpack");
        fs::write(&fresh, b"first").unwrap();
        assert_eq!(mode(&dest), mode(&fresh), "a new file takes the default");
        for want in [0o600, 0o640, 0o604, 0o400] {
            fs::set_permissions(&dest, fs::Permissions::from_mode(want)).unwrap();
            write(&dest, format!("mode {want:o}").as_bytes()).unwrap();
            assert_eq!(mode(&dest), format!("{want:o}"), "the mode after the save");
            assert_eq!(
                fs::read(&dest).unwrap(),
                format!("mode {want:o}").as_bytes()
            );
        }
        fs::set_permissions(&dest, fs::Permissions::from_mode(0o600)).unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_symlink_is_written_through_not_replaced() {
        let dir = std::env::temp_dir().join(format!("rheon-atomic-link-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("real.msgpack");
        let link = dir.join("current.msgpack");
        fs::write(&target, b"old").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &link).unwrap();
        #[cfg(not(unix))]
        // Windows needs a privilege for symlinks; without it there is nothing
        // to test, and the resolve path is the same code either way.
        if std::os::windows::fs::symlink_file(&target, &link).is_err() {
            fs::remove_dir_all(&dir).unwrap();
            return;
        }

        write(&link, b"new").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"new", "wrote past the link");
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the link was replaced by a regular file"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Review 2026-11-23, IO-2 (1): the temporary is created new
    /// (`O_EXCL`), so a file or a link already at its name -- planted, or
    /// left by a process of the same pid -- is never written through: the
    /// name after it is taken. `File::create` followed a planted link,
    /// wrote into its target, and renamed the link over the destination.
    #[cfg(unix)]
    #[test]
    fn a_temporary_is_created_new_and_never_through_a_planted_link() {
        let dir = std::env::temp_dir().join(format!("rheon-atomic-excl-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("state.msgpack");
        let victim = dir.join("victim");
        fs::write(&victim, b"untouched").unwrap();
        let first = NEXT_TEMP.load(Ordering::Relaxed);
        for seq in first..first + 64 {
            std::os::unix::fs::symlink(&victim, temp_name(&dest, seq).unwrap()).unwrap();
        }
        write(&dest, b"new").unwrap();
        assert_eq!(fs::read(&victim).unwrap(), b"untouched");
        assert_eq!(fs::read(&dest).unwrap(), b"new");
        assert!(fs::symlink_metadata(&dest).unwrap().file_type().is_file());
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Review 2026-11-23, IO-2 (2) and (3): the temporary is created with
    /// the destination's mode, so a private file's new bytes are never
    /// readable by others while they are written (it was the process's
    /// default, 644 under a umask of 022, until the commit); and the mode
    /// copied is the permission bits alone, not setuid, setgid or sticky.
    #[cfg(unix)]
    #[test]
    fn the_temporary_is_as_private_as_the_destination_while_written() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("rheon-atomic-priv-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("state.msgpack");
        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o7777;
        fs::write(&dest, b"old").unwrap();
        fs::set_permissions(&dest, fs::Permissions::from_mode(0o600)).unwrap();
        let (file, pending) = AtomicFile::create(&dest).unwrap();
        assert_eq!(mode(&pending.tmp), 0o600, "while it is written");
        drop(file);
        pending.commit().unwrap();
        assert_eq!(mode(&dest), 0o600);
        fs::set_permissions(&dest, fs::Permissions::from_mode(0o4755)).unwrap();
        assert_eq!(mode(&dest), 0o4755, "the setuid bit took");
        write(&dest, b"new").unwrap();
        assert_eq!(mode(&dest), 0o755, "no special bit is copied");
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Review 2026-11-23, CLI-5: a path ending in a separator names a
    /// folder, and no file is made for it. Its temporary was made beside it,
    /// and the rename at the end failed (`ENOTDIR`), after the run.
    #[test]
    fn a_path_ending_in_a_separator_is_refused_at_once() {
        let dir = std::env::temp_dir().join(format!("rheon-atomic-slash-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let mut slashed = dir.join("out").into_os_string();
        slashed.push(std::path::MAIN_SEPARATOR_STR);
        let err = AtomicFile::create(Path::new(&slashed))
            .map(drop)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            0,
            "a temporary was made"
        );
        assert!(AtomicFile::create(&dir.join("out")).is_ok());
        fs::remove_dir_all(&dir).unwrap();
    }
}
