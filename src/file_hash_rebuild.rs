//! Daemon-side rebuild of `file_hashes` as a `WITHOUT ROWID` table, a step
//! of [`crate::maintenance`].
//!
//! Versions before #1206 created `file_hashes` as a rowid table with a TEXT
//! primary key, which stores every path twice: in the table and in the key's
//! automatic index. The rebuild copies every row under the index write lock,
//! so like the blob index heal it runs only while the machine is quiet, and
//! only once the GC prune has left at most [`MAX_ROWS`] rows, none of them
//! expired. `kache doctor --repair` runs it without those limits.
//!
//! TODO: remove this step once no supported store can still hold the rowid
//! table.

use crate::cache_key::{FileHashRows, file_hashes_has_rowid};
use crate::config::Config;
use crate::maintenance::{Trigger, is_quiet};
use crate::store::Store;
use std::time::{Duration, Instant};

/// Most rows a rebuild copies. A local disk copied a million rows in about
/// 3.5 s, inside the wrappers' 5 s busy timeout, which a build that starts
/// meanwhile waits on.
pub(crate) const MAX_ROWS: u64 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SkipReason {
    /// Compiles hold permits, or a wrapper request arrived recently.
    Busy,
    /// The permit slots could not be read.
    UnknownLoad,
    /// A GC holds `gc.lock`; the two must not overlap.
    GcRunning,
    /// Rows past the prune window remain; the GC prune deletes them first.
    ExpiredRows,
    /// More than [`MAX_ROWS`] rows to copy.
    TooManyRows,
}

impl SkipReason {
    fn label(self) -> &'static str {
        match self {
            SkipReason::Busy => "builds are active",
            SkipReason::UnknownLoad => "build load is unknown",
            SkipReason::GcRunning => "a GC holds gc.lock",
            SkipReason::ExpiredRows => "the GC prune has not deleted the expired rows yet",
            SkipReason::TooManyRows => {
                "too many rows to copy unattended; run `kache doctor --repair`"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    NotNeeded,
    Skipped(SkipReason),
    /// Another connection held the index; nothing was changed.
    IndexBusy,
    Rebuilt {
        rows: usize,
        elapsed: Duration,
    },
}

impl Outcome {
    fn describe(&self) -> String {
        match self {
            Outcome::NotNeeded => "file hash table rebuild not needed".to_string(),
            Outcome::Skipped(reason) => {
                format!("file hash table rebuild deferred: {}", reason.label())
            }
            Outcome::IndexBusy => {
                "file hash table rebuild found the index busy; will retry".to_string()
            }
            Outcome::Rebuilt { rows, elapsed } => format!(
                "rebuilt the file hash table without rowids: {rows} rows in {} ms",
                elapsed.as_millis()
            ),
        }
    }
}

/// Only a quiet machine rebuilds. Shutdown waives the request age, as for
/// the other steps; the rebuild copies one table, not every entry's
/// metadata, so it needs no cap of its own.
fn decide(permits: Option<u32>, since_last_request: Duration) -> Option<SkipReason> {
    if is_quiet(permits, since_last_request) {
        return None;
    }
    Some(match permits {
        None => SkipReason::UnknownLoad,
        Some(_) => SkipReason::Busy,
    })
}

/// Copy only what the prune leaves, and not more than [`MAX_ROWS`].
fn rows_fit(rows: FileHashRows) -> Option<SkipReason> {
    if rows.expired > 0 {
        return Some(SkipReason::ExpiredRows);
    }
    if rows.rows > MAX_ROWS {
        return Some(SkipReason::TooManyRows);
    }
    None
}

fn is_index_busy(error: &rusqlite::Error) -> bool {
    use rusqlite::ErrorCode::{DatabaseBusy, DatabaseLocked};
    error
        .sqlite_error_code()
        .is_some_and(|code| matches!(code, DatabaseBusy | DatabaseLocked))
}

fn attempt(config: &Config, trigger: Trigger<'_>) -> anyhow::Result<Outcome> {
    let started = Instant::now();
    // Own connection, like GC: the copy must not sit on the daemon's Store
    // mutex.
    let store = Store::open(config)?;
    let index = store.file_hash_cache();
    if !file_hashes_has_rowid(index.db())? {
        return Ok(Outcome::NotNeeded);
    }
    let decide = || {
        decide(
            crate::scheduler::permits_in_use(&config.cache_dir),
            trigger.idle_for(),
        )
    };
    if let Some(reason) = decide() {
        return Ok(Outcome::Skipped(reason));
    }
    if let Some(reason) = rows_fit(index.file_hash_rows()?) {
        return Ok(Outcome::Skipped(reason));
    }
    let Some(_gc_lock) = store.try_gc_lock()? else {
        return Ok(Outcome::Skipped(SkipReason::GcRunning));
    };
    // A build can start between the first decision and the lock.
    if let Some(reason) = decide() {
        return Ok(Outcome::Skipped(reason));
    }
    // Yield to the first contender instead of queueing behind it.
    index.db().pragma_update(None, "busy_timeout", 0)?;
    match index.rebuild_file_hashes_without_rowid() {
        Ok(Some(rows)) => Ok(Outcome::Rebuilt {
            rows,
            elapsed: started.elapsed(),
        }),
        Ok(None) => Ok(Outcome::NotNeeded),
        Err(error) if is_index_busy(&error) => Ok(Outcome::IndexBusy),
        Err(error) => Err(error.into()),
    }
}

/// One rebuild attempt. Blocking: call from `spawn_blocking`. Errors are
/// logged and dropped; a later check retries.
pub(crate) fn run(config: &Config, trigger: Trigger<'_>) -> Option<Outcome> {
    match attempt(config, trigger) {
        Ok(outcome) => {
            if matches!(outcome, Outcome::Rebuilt { .. }) {
                tracing::info!("{}", outcome.describe());
            } else {
                tracing::debug!("{}", outcome.describe());
            }
            Some(outcome)
        }
        Err(error) => {
            tracing::warn!("file hash table rebuild failed: {error:#}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maintenance::{QUIET_AFTER, RequestClock};

    const SEC: Duration = Duration::from_secs(1);

    /// A store whose `file_hashes` is the rowid table older versions made.
    fn legacy_store(dir: &std::path::Path) -> Config {
        let config = crate::test_support::test_config(dir.to_path_buf());
        let store = Store::open(&config).unwrap();
        store
            .file_hash_cache()
            .db()
            .execute_batch(
                "DROP TABLE file_hashes;
                 CREATE TABLE file_hashes (
                     path       TEXT PRIMARY KEY,
                     size       INTEGER NOT NULL,
                     mtime_ns   INTEGER NOT NULL,
                     ctime_ns   INTEGER NOT NULL DEFAULT 0,
                     inode      INTEGER NOT NULL DEFAULT 0,
                     hash       TEXT NOT NULL,
                     updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                 );
                 INSERT INTO file_hashes (path, size, mtime_ns, hash)
                     VALUES ('/a', 1, 1, 'ha'), ('/b', 2, 2, 'hb');",
            )
            .unwrap();
        config
    }

    fn has_rowid(config: &Config) -> bool {
        let store = Store::open(config).unwrap();
        file_hashes_has_rowid(store.file_hash_cache().db()).unwrap()
    }

    #[test]
    fn only_a_quiet_machine_rebuilds() {
        assert_eq!(decide(Some(0), QUIET_AFTER), None);
        assert_eq!(decide(Some(0), QUIET_AFTER - SEC), Some(SkipReason::Busy));
        assert_eq!(decide(Some(1), Duration::MAX), Some(SkipReason::Busy));
        assert_eq!(decide(None, Duration::MAX), Some(SkipReason::UnknownLoad));
    }

    #[test]
    fn only_a_pruned_table_of_at_most_a_million_rows_is_copied() {
        assert_eq!(MAX_ROWS, 1_000_000);
        let rows = |rows, expired| FileHashRows { rows, expired };
        assert_eq!(rows_fit(rows(0, 0)), None);
        assert_eq!(rows_fit(rows(MAX_ROWS, 0)), None);
        assert_eq!(
            rows_fit(rows(MAX_ROWS + 1, 0)),
            Some(SkipReason::TooManyRows)
        );
        assert_eq!(rows_fit(rows(10, 1)), Some(SkipReason::ExpiredRows));
        assert_eq!(
            rows_fit(rows(MAX_ROWS + 1, 1)),
            Some(SkipReason::ExpiredRows)
        );
    }

    #[test]
    fn expired_rows_wait_for_the_prune() {
        let dir = tempfile::tempdir().unwrap();
        let config = legacy_store(dir.path());
        let store = Store::open(&config).unwrap();
        store
            .file_hash_cache()
            .db()
            .execute_batch(
                "INSERT INTO file_hashes (path, size, mtime_ns, hash, updated_at)
                 VALUES ('/old', 1, 1, 'h', '2000-01-01 00:00:00')",
            )
            .unwrap();
        assert_eq!(
            attempt(&config, Trigger::Shutdown).unwrap(),
            Outcome::Skipped(SkipReason::ExpiredRows)
        );
        assert!(has_rowid(&config));
        assert_eq!(store.file_hash_cache().prune_file_hashes().unwrap(), 1);
        assert!(matches!(
            attempt(&config, Trigger::Shutdown).unwrap(),
            Outcome::Rebuilt { rows: 2, .. }
        ));
    }

    #[test]
    fn outcomes_describe_what_happened() {
        assert_eq!(
            Outcome::NotNeeded.describe(),
            "file hash table rebuild not needed"
        );
        assert_eq!(
            Outcome::Skipped(SkipReason::GcRunning).describe(),
            "file hash table rebuild deferred: a GC holds gc.lock"
        );
        assert_eq!(
            Outcome::Skipped(SkipReason::UnknownLoad).describe(),
            "file hash table rebuild deferred: build load is unknown"
        );
        assert_eq!(
            Outcome::Skipped(SkipReason::ExpiredRows).describe(),
            "file hash table rebuild deferred: the GC prune has not deleted the expired rows yet"
        );
        assert_eq!(
            Outcome::Skipped(SkipReason::TooManyRows).describe(),
            "file hash table rebuild deferred: too many rows to copy unattended; \
             run `kache doctor --repair`"
        );
        assert_eq!(
            Outcome::IndexBusy.describe(),
            "file hash table rebuild found the index busy; will retry"
        );
        assert_eq!(
            Outcome::Rebuilt {
                rows: 12,
                elapsed: Duration::from_millis(340)
            }
            .describe(),
            "rebuilt the file hash table without rowids: 12 rows in 340 ms"
        );
    }

    #[test]
    fn a_current_table_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let config = crate::test_support::test_config(dir.path().to_path_buf());
        drop(Store::open(&config).unwrap());
        assert_eq!(run(&config, Trigger::Shutdown), Some(Outcome::NotNeeded));
    }

    #[test]
    fn a_quiet_machine_rebuilds_a_rowid_table() {
        let dir = tempfile::tempdir().unwrap();
        let config = legacy_store(dir.path());
        assert!(has_rowid(&config));

        let recent = RequestClock::new();
        assert_eq!(
            attempt(&config, Trigger::Periodic(&recent)).unwrap(),
            Outcome::Skipped(SkipReason::Busy)
        );
        assert!(has_rowid(&config));

        let idle = RequestClock::idle();
        assert!(matches!(
            attempt(&config, Trigger::Periodic(&idle)).unwrap(),
            Outcome::Rebuilt { rows: 2, .. }
        ));
        assert!(!has_rowid(&config));
        assert_eq!(
            attempt(&config, Trigger::Periodic(&idle)).unwrap(),
            Outcome::NotNeeded
        );
    }

    #[test]
    fn a_running_gc_defers_the_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        let config = legacy_store(dir.path());
        let gc_store = Store::open(&config).unwrap();
        let gc_lock = gc_store.try_gc_lock().unwrap().unwrap();
        assert_eq!(
            attempt(&config, Trigger::Shutdown).unwrap(),
            Outcome::Skipped(SkipReason::GcRunning)
        );
        drop(gc_lock);
        assert!(matches!(
            attempt(&config, Trigger::Shutdown).unwrap(),
            Outcome::Rebuilt { .. }
        ));
    }

    #[test]
    fn a_failed_rebuild_is_an_error_and_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let config = legacy_store(dir.path());
        let store = Store::open(&config).unwrap();
        store
            .file_hash_cache()
            .db()
            .execute_batch("CREATE TABLE file_hashes_rebuild (path TEXT)")
            .unwrap();
        assert!(attempt(&config, Trigger::Shutdown).is_err());
        assert_eq!(run(&config, Trigger::Shutdown), None);
        assert!(has_rowid(&config));
    }

    #[test]
    fn a_contended_index_yields() {
        let dir = tempfile::tempdir().unwrap();
        let config = legacy_store(dir.path());
        let writer = rusqlite::Connection::open(config.index_db_path()).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        let started = Instant::now();
        assert_eq!(
            attempt(&config, Trigger::Shutdown).unwrap(),
            Outcome::IndexBusy
        );
        assert!(started.elapsed() < Duration::from_secs(4));
        writer.execute_batch("ROLLBACK").unwrap();
        assert!(has_rowid(&config));
    }
}
