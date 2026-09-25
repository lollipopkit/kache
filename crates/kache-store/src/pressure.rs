//! The part of the store size that eviction cannot free.
//!
//! A blob still cloned or hardlinked into a build's target directory keeps
//! its blocks when the store drops its own name, so eviction refuses the
//! entries holding its last references (#725). Those bytes still count in
//! `SUM(blobs.size)`. Measured against that sum alone, a store whose clones
//! exceed `max_size` never gets back under it: every automatic sweep fires on
//! bytes it cannot free and backs off, and the size pass evicts every
//! freeable entry while chasing a target it cannot reach (#1206).
//!
//! A size sweep measures those bytes before it evicts and records the total
//! here, so the automatic triggers can leave them out of size pressure.
//!
//! Removals that unlink blobs drop the record without taking `gc.lock`, so a
//! sweep's publication and a drop serialize on a lock of their own: the
//! publication counts only measured blobs still in the index, and a drop
//! that waited for it removes what it wrote.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// How long a recorded measurement is left out of size pressure. After it
/// expires the triggers see the full physical size again, so a store whose
/// clones went away since is swept, and measured, again.
pub const UNRECLAIMABLE_RECORD_TTL: Duration = Duration::from_secs(6 * 3600);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct UnreclaimableRecord {
    bytes: u64,
    /// Unix seconds.
    measured_at: u64,
}

fn record_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("gc-unreclaimable.json")
}

fn record_lock(cache_dir: &Path) -> anyhow::Result<crate::StoreLock> {
    crate::StoreLock::acquire(&cache_dir.join("gc-unreclaimable.lock"))
}

pub(crate) fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// The recorded bytes while the record is at most
/// [`UNRECLAIMABLE_RECORD_TTL`] old. A record from the future, after the
/// clock went backwards, does not count.
fn bytes_if_current(record: UnreclaimableRecord, now: u64) -> u64 {
    let current =
        record.measured_at <= now && now - record.measured_at < UNRECLAIMABLE_RECORD_TTL.as_secs();
    if current { record.bytes } else { 0 }
}

/// Bytes the last size sweep found no eviction could free, or 0 when there
/// is no current record.
pub(crate) fn recorded_unreclaimable(cache_dir: &Path, now: u64) -> u64 {
    std::fs::read(record_path(cache_dir))
        .ok()
        .and_then(|json| serde_json::from_slice(&json).ok())
        .map_or(0, |record| bytes_if_current(record, now))
}

/// Drop the measurement once blobs it may count are gone: a removal that
/// unlinked blobs, a cleared store, a rewritten blob index. Kept, it would
/// go on subtracting their bytes from a store that no longer holds them and
/// hide real pressure until it expired. The next size sweep measures again.
///
/// Call it after the removal commits. Waiting for the record lock orders the
/// drop after any publication already counting the removed blobs.
pub(crate) fn forget_unreclaimable(cache_dir: &Path) {
    // Without the lock the record is still dropped: a stale credit costs
    // more than a lost measurement.
    let _lock = record_lock(cache_dir).ok();
    let _ = std::fs::remove_file(record_path(cache_dir));
}

/// Store a size sweep's measurement. The caller holds `gc.lock`.
///
/// `current_bytes` runs under the record lock and returns the measured bytes
/// still in the index. Returns the bytes written; an error leaves the
/// record as it was and only costs the next trigger its correction.
pub(crate) fn record_unreclaimable(
    cache_dir: &Path,
    now: u64,
    current_bytes: impl FnOnce() -> anyhow::Result<u64>,
) -> anyhow::Result<u64> {
    let _lock = record_lock(cache_dir)?;
    let bytes = current_bytes()?;
    let record = UnreclaimableRecord {
        bytes,
        measured_at: now,
    };
    crate::atomic::atomic_replace(&record_path(cache_dir), &serde_json::to_vec(&record)?)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_000_000;

    #[test]
    fn the_record_expires_after_six_hours_of_the_wall_clock() {
        assert_eq!(UNRECLAIMABLE_RECORD_TTL, Duration::from_secs(21_600));
        let wall = || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        };
        let before = wall();
        let now = unix_now_secs();
        assert!(before <= now && now <= wall());
    }

    #[test]
    fn a_record_counts_only_until_it_expires() {
        let ttl = UNRECLAIMABLE_RECORD_TTL.as_secs();
        let record = |measured_at| UnreclaimableRecord {
            bytes: 700,
            measured_at,
        };
        assert_eq!(bytes_if_current(record(NOW), NOW), 700);
        assert_eq!(bytes_if_current(record(NOW - ttl + 1), NOW), 700);
        assert_eq!(bytes_if_current(record(NOW - ttl), NOW), 0, "expired");
        assert_eq!(bytes_if_current(record(NOW + 1), NOW), 0, "from the future");
    }

    #[test]
    fn a_recorded_measurement_reads_back_until_it_expires() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(recorded_unreclaimable(dir.path(), NOW), 0, "no record");

        record_unreclaimable(dir.path(), NOW, || Ok(4096)).unwrap();
        assert_eq!(recorded_unreclaimable(dir.path(), NOW), 4096);
        let expiry = NOW + UNRECLAIMABLE_RECORD_TTL.as_secs();
        assert_eq!(recorded_unreclaimable(dir.path(), expiry - 1), 4096);
        assert_eq!(recorded_unreclaimable(dir.path(), expiry), 0);

        record_unreclaimable(dir.path(), NOW, || Ok(0)).unwrap();
        assert_eq!(recorded_unreclaimable(dir.path(), NOW), 0, "replaced");

        record_unreclaimable(dir.path(), NOW, || Ok(4096)).unwrap();
        forget_unreclaimable(dir.path());
        assert_eq!(recorded_unreclaimable(dir.path(), NOW), 0, "forgotten");

        std::fs::write(record_path(dir.path()), b"not json").unwrap();
        assert_eq!(recorded_unreclaimable(dir.path(), NOW), 0, "corrupt");

        record_unreclaimable(dir.path(), NOW, || Ok(4096)).unwrap();
        assert!(record_unreclaimable(dir.path(), NOW, || anyhow::bail!("index")).is_err());
        assert_eq!(
            recorded_unreclaimable(dir.path(), NOW),
            4096,
            "a failed count keeps the record"
        );
    }

    /// A drop that arrives while a publication is counting waits for it and
    /// removes what it wrote, instead of landing first and being overwritten.
    #[test]
    fn a_drop_during_a_publication_removes_what_it_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let (started, drop_started) = std::sync::mpsc::channel();
        let mut drop = None;
        record_unreclaimable(dir.path(), NOW, || {
            let path = dir.path().to_path_buf();
            drop = Some(std::thread::spawn(move || {
                started.send(()).unwrap();
                forget_unreclaimable(&path);
            }));
            drop_started.recv().unwrap();
            std::thread::sleep(Duration::from_millis(100));
            Ok(4096)
        })
        .unwrap();
        drop.unwrap().join().unwrap();
        assert_eq!(recorded_unreclaimable(dir.path(), NOW), 0);
    }
}
