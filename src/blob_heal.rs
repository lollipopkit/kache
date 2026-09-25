//! Daemon-side repair of blob-index drift, a step of [`crate::maintenance`]
//! and of every daemon GC sweep.
//!
//! `blobs` and `entry_blobs` are derived from committed entries. When they
//! drift, leaked refcounts keep bytes counted against `max_size` that no
//! eviction can free. A SQL probe spots that; the repair rebuilds both tables
//! from every committed `meta.json` under the index write lock. Unbounded, it
//! runs only while the machine is quiet. A host that is never quiet still
//! gets the repair from its GC sweeps, which stop the metadata scan after
//! [`GC_BUDGET`] and change nothing then (#1206). The drift costs disk
//! budget, never a wrong hit.

use crate::config::Config;
use crate::maintenance::{Trigger, is_quiet, unix_now_secs};
use crate::store::{BlobIndexDrift, BlobRefcountDrift, ReconcileOutOfTime, Store};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Wait after a failed repair. The rebuild refuses to run past an entry it
/// cannot read, and retrying every check would take the write lock each time
/// for the same refusal.
pub(crate) const RETRY_AFTER_FAILURE: Duration = Duration::from_secs(6 * 3600);
/// Most entries a shutdown repair reads. The scan opens every committed
/// `meta.json`, and shutdown has to stay short.
pub(crate) const SHUTDOWN_MAX_ENTRIES: u64 = 25_000;
/// Longest a GC sweep's repair holds the index write lock for the metadata
/// scan. Well under the wrappers' 5 s busy timeout, so a build that queues
/// behind it waits instead of failing.
pub(crate) const GC_BUDGET: Duration = Duration::from_secs(2);
/// Longest a GC sweep's repair waits for the index write lock.
const GC_LOCK_WAIT: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SkipReason {
    /// Compiles hold permits, or a wrapper request arrived recently.
    Busy,
    /// The permit slots could not be read.
    UnknownLoad,
    /// The last repair failed less than [`RETRY_AFTER_FAILURE`] ago.
    BackingOff,
    /// More than [`SHUTDOWN_MAX_ENTRIES`] entries to read at shutdown.
    TooManyEntries,
    /// A GC holds `gc.lock`; the two must not overlap.
    GcRunning,
    /// The last heal in a GC ran out of time on a store this size.
    OutOfTimeRecently,
}

impl SkipReason {
    fn label(self) -> &'static str {
        match self {
            SkipReason::Busy => "builds are active",
            SkipReason::UnknownLoad => "build load is unknown",
            SkipReason::BackingOff => "the last attempt failed; waiting before the next",
            SkipReason::TooManyEntries => "too many entries to read during shutdown",
            SkipReason::GcRunning => "a GC holds gc.lock",
            SkipReason::OutOfTimeRecently => {
                "the last heal in GC ran out of time on a store this size"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    NotNeeded,
    Run,
    Skip(SkipReason),
}

/// Whether to repair now. Only a quiet machine does.
pub(crate) fn decide(
    drifted: bool,
    backing_off: bool,
    permits: Option<u32>,
    since_last_request: Duration,
) -> Decision {
    if !drifted {
        return Decision::NotNeeded;
    }
    if backing_off {
        return Decision::Skip(SkipReason::BackingOff);
    }
    if is_quiet(permits, since_last_request) {
        return Decision::Run;
    }
    Decision::Skip(match permits {
        None => SkipReason::UnknownLoad,
        Some(_) => SkipReason::Busy,
    })
}

/// [`decide`] for a trigger. Shutdown waives request age and adds the entry cap.
fn decide_for(
    trigger: Trigger<'_>,
    drifted: bool,
    backing_off: bool,
    permits: Option<u32>,
    entries: u64,
) -> Decision {
    let decision = decide(drifted, backing_off, permits, trigger.idle_for());
    let capped = matches!(trigger, Trigger::Shutdown) && entries > SHUTDOWN_MAX_ENTRIES;
    if decision == Decision::Run && capped {
        return Decision::Skip(SkipReason::TooManyEntries);
    }
    decision
}

/// The last failed repair. Kept on disk because a CI daemon restarts per job
/// and would otherwise retry on every start.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct FailedHeal {
    /// Unix seconds.
    failed_at: u64,
    reason: String,
}

fn state_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("blob-heal.json")
}

fn read_failure(cache_dir: &Path) -> Option<FailedHeal> {
    let json = std::fs::read(state_path(cache_dir)).ok()?;
    serde_json::from_slice(&json).ok()
}

fn record_failure(cache_dir: &Path, failed_at: u64, reason: &str) {
    let state = FailedHeal {
        failed_at,
        reason: reason.to_string(),
    };
    if let Ok(json) = serde_json::to_vec(&state)
        && let Err(error) = crate::atomic::atomic_replace(&state_path(cache_dir), &json)
    {
        tracing::debug!("blob index heal: could not record the failure: {error:#}");
    }
}

fn clear_state(cache_dir: &Path) {
    let _ = std::fs::remove_file(state_path(cache_dir));
}

/// The last heal in a GC that ran out of time. Kept on disk for the same
/// reason as [`FailedHeal`]: a CI daemon restarts per job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct GcOverrun {
    /// Unix seconds.
    at: u64,
    /// Entries read before the budget ran out.
    read: u64,
}

fn gc_overrun_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("blob-heal-gc.json")
}

fn read_gc_overrun(cache_dir: &Path) -> Option<GcOverrun> {
    let json = std::fs::read(gc_overrun_path(cache_dir)).ok()?;
    serde_json::from_slice(&json).ok()
}

fn record_gc_overrun(cache_dir: &Path, overrun: GcOverrun) {
    if let Ok(json) = serde_json::to_vec(&overrun)
        && let Err(error) = crate::atomic::atomic_replace(&gc_overrun_path(cache_dir), &json)
    {
        tracing::debug!("blob index heal: could not record the overrun: {error:#}");
    }
}

fn clear_gc_overrun(cache_dir: &Path) {
    let _ = std::fs::remove_file(gc_overrun_path(cache_dir));
}

/// Whether a GC skips the heal after `overrun`: for [`RETRY_AFTER_FAILURE`],
/// while the store holds more entries than the overrun read in time. Each
/// attempt would hold the write lock for the whole budget and change
/// nothing. An overrun from the future does not count.
fn gc_overrun_holds(overrun: Option<GcOverrun>, now: u64, entries: u64) -> bool {
    overrun.is_some_and(|overrun| {
        overrun.at <= now
            && now - overrun.at < RETRY_AFTER_FAILURE.as_secs()
            && entries > overrun.read
    })
}

/// A failure from the future, after the clock went backwards, does not count.
fn is_backing_off(failed_at: Option<u64>, now: u64) -> bool {
    failed_at.is_some_and(|at| at <= now && now - at < RETRY_AFTER_FAILURE.as_secs())
}

/// Index contention is not a failed repair: nothing was read, and the next
/// check may find the index free.
fn is_index_busy(error: &anyhow::Error) -> bool {
    use rusqlite::ErrorCode::{DatabaseBusy, DatabaseLocked};
    error
        .downcast_ref::<rusqlite::Error>()
        .and_then(rusqlite::Error::sqlite_error_code)
        .is_some_and(|code| matches!(code, DatabaseBusy | DatabaseLocked))
}

/// What happened to the blob files a repair left without a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sweep {
    /// Unlinked this many files holding this many bytes.
    Swept { files: usize, bytes: u64 },
    /// The sweep failed; the next GC retries it.
    Failed,
    /// Left to the orphan sweep of the GC the repair ran in.
    LeftToGc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Outcome {
    NotNeeded,
    Skipped(SkipReason),
    /// Another connection held the index; nothing was changed.
    IndexBusy,
    /// A GC sweep's repair reached [`GC_BUDGET`] after reading `read` of
    /// `total` entries. Nothing was changed.
    OutOfTime {
        read: usize,
        total: usize,
    },
    Healed {
        /// What the probe saw before the repair.
        probe: BlobRefcountDrift,
        /// Rows the rebuild found wrong against committed metadata.
        repaired: BlobIndexDrift,
        sweep: Sweep,
        elapsed: Duration,
    },
    /// The rebuild refused to run. Nothing was changed.
    Failed {
        reason: String,
    },
}

impl Outcome {
    fn describe(&self) -> String {
        match self {
            Outcome::NotNeeded => "blob index heal not needed".to_string(),
            Outcome::Skipped(reason) => format!("blob index heal deferred: {}", reason.label()),
            Outcome::IndexBusy => "blob index heal found the index busy; will retry".to_string(),
            Outcome::OutOfTime { read, total } => format!(
                "blob index heal in GC stopped after {} s, having read {read} of {total} \
                 entries, and changed nothing; GC skips it for {} h unless the store shrinks \
                 to {read} entries, and it runs unbounded once builds are idle",
                GC_BUDGET.as_secs(),
                RETRY_AFTER_FAILURE.as_secs() / 3600
            ),
            Outcome::Healed {
                probe,
                repaired,
                sweep,
                elapsed,
            } => {
                let swept = match sweep {
                    Sweep::Swept { files, bytes } => format!(
                        "swept {files} blob files ({})",
                        crate::report::format_bytes(*bytes)
                    ),
                    Sweep::Failed => "the blob sweep failed and is left to the next GC".to_string(),
                    Sweep::LeftToGc => "unowned blob files are left to this GC's sweep".to_string(),
                };
                format!(
                    "blob index healed in {} ms: {} blobs ({}) had no owner, {} refcounts were off; \
                     rewrote {} blob rows and {} mappings; {swept}",
                    elapsed.as_millis(),
                    probe.unowned,
                    crate::report::format_bytes(probe.unowned_bytes),
                    probe.mismatched() - probe.unowned,
                    repaired.blobs,
                    repaired.entry_mappings,
                )
            }
            Outcome::Failed { reason } => format!(
                "blob index heal refused, next attempt in {} h: {reason}. \
                 `kache doctor --repair` removes a corrupt entry and rebuilds the blob index",
                RETRY_AFTER_FAILURE.as_secs() / 3600
            ),
        }
    }

    /// A refusal is a warning. A repair, or a skip that waiting will not
    /// cure, is worth an info line. The rest repeats every check.
    fn is_refusal(&self) -> bool {
        matches!(self, Outcome::Failed { .. })
    }

    fn is_noteworthy(&self) -> bool {
        matches!(
            self,
            Outcome::Healed { .. }
                | Outcome::OutOfTime { .. }
                | Outcome::Skipped(SkipReason::TooManyEntries)
        )
    }
}

fn attempt(config: &Config, trigger: Trigger<'_>, now: u64) -> anyhow::Result<Outcome> {
    let started = Instant::now();
    let cache_dir = &config.cache_dir;
    // Own connection, like GC: the rebuild must not sit on the daemon's Store
    // mutex.
    let store = Store::open(config)?;
    let probe = store.blob_refcount_drift()?;
    if probe.is_clean() {
        clear_state(cache_dir);
        return Ok(Outcome::NotNeeded);
    }
    let backing_off = is_backing_off(read_failure(cache_dir).map(|f| f.failed_at), now);
    let entries = store.entry_count()? as u64;
    let decide = || {
        decide_for(
            trigger,
            true,
            backing_off,
            crate::scheduler::permits_in_use(cache_dir),
            entries,
        )
    };
    if let Decision::Skip(reason) = decide() {
        return Ok(Outcome::Skipped(reason));
    }
    let Some(_gc_lock) = store.try_gc_lock()? else {
        return Ok(Outcome::Skipped(SkipReason::GcRunning));
    };
    // A build can start between the first decision and the lock.
    if let Decision::Skip(reason) = decide() {
        return Ok(Outcome::Skipped(reason));
    }
    // Yield to the first contender instead of queueing behind it.
    store
        .file_hash_cache()
        .db()
        .pragma_update(None, "busy_timeout", 0)?;
    let repaired = match store.reconcile_blob_index() {
        Ok(repaired) => repaired,
        Err(error) if is_index_busy(&error) => return Ok(Outcome::IndexBusy),
        Err(error) => {
            let reason = format!("{error:#}");
            record_failure(cache_dir, now, &reason);
            return Ok(Outcome::Failed { reason });
        }
    };
    clear_state(cache_dir);
    let sweep = match store.sweep_orphan_blobs(crate::daemon::ORPHAN_BLOB_GRACE) {
        Ok(stats) => Sweep::Swept {
            files: stats.removed,
            bytes: stats.bytes_reclaimed,
        },
        Err(_) => Sweep::Failed,
    };
    Ok(Outcome::Healed {
        probe,
        repaired,
        sweep,
        elapsed: started.elapsed(),
    })
}

/// The repair inside a GC sweep. The caller holds `gc.lock`, and builds may
/// be running: the scan stops at `budget` and the repair then changes
/// nothing. The GC's own orphan sweep follows, so this one does not sweep.
fn attempt_in_gc(config: &Config, now: u64, budget: Duration) -> anyhow::Result<Outcome> {
    let started = Instant::now();
    let cache_dir = &config.cache_dir;
    let store = Store::open(config)?;
    let probe = store.blob_refcount_drift()?;
    if probe.is_clean() {
        clear_state(cache_dir);
        clear_gc_overrun(cache_dir);
        return Ok(Outcome::NotNeeded);
    }
    if is_backing_off(read_failure(cache_dir).map(|f| f.failed_at), now) {
        return Ok(Outcome::Skipped(SkipReason::BackingOff));
    }
    let entries = store.entry_count()? as u64;
    if gc_overrun_holds(read_gc_overrun(cache_dir), now, entries) {
        return Ok(Outcome::Skipped(SkipReason::OutOfTimeRecently));
    }
    // Wait for a build's write to finish, but not as long as a wrapper
    // would: the next sweep or quiet check retries.
    store.file_hash_cache().db().pragma_update(
        None,
        "busy_timeout",
        GC_LOCK_WAIT.as_millis() as i64,
    )?;
    let error = match store.reconcile_blob_index_by(Some(started + budget)) {
        Ok(repaired) => {
            clear_state(cache_dir);
            clear_gc_overrun(cache_dir);
            return Ok(Outcome::Healed {
                probe,
                repaired,
                sweep: Sweep::LeftToGc,
                elapsed: started.elapsed(),
            });
        }
        Err(error) => error,
    };
    if let Some(&ReconcileOutOfTime { read, total }) = error.downcast_ref() {
        record_gc_overrun(
            cache_dir,
            GcOverrun {
                at: now,
                read: read as u64,
            },
        );
        return Ok(Outcome::OutOfTime { read, total });
    }
    if is_index_busy(&error) {
        return Ok(Outcome::IndexBusy);
    }
    let reason = format!("{error:#}");
    record_failure(cache_dir, now, &reason);
    Ok(Outcome::Failed { reason })
}

/// Log an attempt's outcome, or its error, which is dropped: a later check
/// retries.
fn report(attempted: anyhow::Result<Outcome>) -> Option<Outcome> {
    match attempted {
        Ok(outcome) => {
            if outcome.is_refusal() {
                tracing::warn!("{}", outcome.describe());
            } else if outcome.is_noteworthy() {
                tracing::info!("{}", outcome.describe());
            } else {
                tracing::debug!("{}", outcome.describe());
            }
            Some(outcome)
        }
        Err(error) => {
            tracing::warn!("blob index heal failed: {error:#}");
            None
        }
    }
}

/// One repair inside a daemon GC sweep that holds `gc.lock`. Blocking.
pub(crate) fn run_in_gc(config: &Config) -> Option<Outcome> {
    report(attempt_in_gc(config, unix_now_secs(), GC_BUDGET))
}

/// One repair attempt. Blocking: call from `spawn_blocking`.
pub(crate) fn run(config: &Config, trigger: Trigger<'_>) -> Option<Outcome> {
    report(attempt(config, trigger, unix_now_secs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maintenance::{QUIET_AFTER, RequestClock};
    #[cfg(unix)]
    use crate::store::StoreLock;

    const SEC: Duration = Duration::from_secs(1);
    const NOW: u64 = 1_800_000_000;
    #[cfg(unix)]
    const PAYLOAD: &[u8] = b"blob shared by two entries";
    #[cfg(unix)]
    const STALE: &[u8] = b"stale unowned blob";
    #[cfg(unix)]
    const FRESH: &[u8] = b"fresh unowned blob!";

    #[test]
    fn limits_are_the_documented_ones() {
        assert_eq!(RETRY_AFTER_FAILURE, Duration::from_secs(21_600));
        assert_eq!(SHUTDOWN_MAX_ENTRIES, 25_000);
        assert_eq!(GC_BUDGET, Duration::from_secs(2));
        assert_eq!(GC_LOCK_WAIT, Duration::from_secs(1));
        assert_eq!(crate::daemon::ORPHAN_BLOB_GRACE, Duration::from_secs(3600));
    }

    #[test]
    fn a_clean_index_is_never_repaired() {
        for permits in [None, Some(0), Some(1)] {
            for backing_off in [false, true] {
                assert_eq!(
                    decide(false, backing_off, permits, Duration::MAX),
                    Decision::NotNeeded
                );
            }
        }
    }

    #[test]
    fn repair_needs_a_quiet_machine_and_has_no_forced_mode() {
        let drifted = |permits, idle| decide(true, false, permits, idle);
        assert_eq!(drifted(Some(0), QUIET_AFTER), Decision::Run);
        assert_eq!(drifted(Some(0), Duration::MAX), Decision::Run);
        assert_eq!(
            drifted(Some(0), QUIET_AFTER - SEC),
            Decision::Skip(SkipReason::Busy)
        );
        assert_eq!(
            drifted(Some(1), Duration::MAX),
            Decision::Skip(SkipReason::Busy)
        );
        assert_eq!(
            drifted(None, Duration::MAX),
            Decision::Skip(SkipReason::UnknownLoad)
        );
    }

    #[test]
    fn a_recent_failure_outranks_a_quiet_machine() {
        assert_eq!(
            decide(true, true, Some(0), Duration::MAX),
            Decision::Skip(SkipReason::BackingOff)
        );
        assert_eq!(
            decide(true, true, Some(1), Duration::ZERO),
            Decision::Skip(SkipReason::BackingOff)
        );
    }

    #[test]
    fn shutdown_waives_request_age_and_caps_the_entry_count() {
        let fresh = RequestClock::new();
        let periodic = Trigger::Periodic(&fresh);
        assert_eq!(
            decide_for(periodic, true, false, Some(0), 0),
            Decision::Skip(SkipReason::Busy)
        );
        assert_eq!(
            decide_for(
                Trigger::Shutdown,
                true,
                false,
                Some(0),
                SHUTDOWN_MAX_ENTRIES
            ),
            Decision::Run
        );
        assert_eq!(
            decide_for(
                Trigger::Shutdown,
                true,
                false,
                Some(0),
                SHUTDOWN_MAX_ENTRIES + 1
            ),
            Decision::Skip(SkipReason::TooManyEntries)
        );
        // The cap replaces only a run, and only at shutdown.
        assert_eq!(
            decide_for(Trigger::Shutdown, true, false, Some(1), u64::MAX),
            Decision::Skip(SkipReason::Busy)
        );
        assert_eq!(
            decide_for(Trigger::Shutdown, false, false, Some(0), u64::MAX),
            Decision::NotNeeded
        );
        let idle = RequestClock::idle();
        assert_eq!(
            decide_for(Trigger::Periodic(&idle), true, false, Some(0), u64::MAX),
            Decision::Run
        );
    }

    #[test]
    fn backoff_lasts_six_hours_from_the_failure() {
        assert!(!is_backing_off(None, NOW));
        assert!(is_backing_off(Some(NOW), NOW));
        assert!(is_backing_off(Some(NOW), NOW + 21_599));
        assert!(!is_backing_off(Some(NOW), NOW + 21_600));
        // The clock went backwards: retry instead of waiting out the gap.
        assert!(!is_backing_off(Some(NOW + 1), NOW));
    }

    #[test]
    fn failure_record_survives_a_restart_and_tolerates_garbage() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_failure(dir.path()), None);
        record_failure(dir.path(), 1_000, "entry abc: parsing");
        assert_eq!(
            std::fs::read_to_string(state_path(dir.path())).unwrap(),
            r#"{"failed_at":1000,"reason":"entry abc: parsing"}"#
        );
        assert_eq!(
            read_failure(dir.path()),
            Some(FailedHeal {
                failed_at: 1_000,
                reason: "entry abc: parsing".to_string()
            })
        );
        assert_eq!(state_path(dir.path()), dir.path().join("blob-heal.json"));

        std::fs::write(state_path(dir.path()), b"{not json").unwrap();
        assert_eq!(read_failure(dir.path()), None);
        clear_state(dir.path());
        assert!(!state_path(dir.path()).exists());
        clear_state(dir.path());
    }

    #[test]
    fn only_sqlite_contention_counts_as_a_busy_index() {
        let sqlite = |code| {
            anyhow::Error::from(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(code),
                None,
            ))
        };
        assert!(is_index_busy(&sqlite(rusqlite::ffi::SQLITE_BUSY)));
        assert!(is_index_busy(&sqlite(rusqlite::ffi::SQLITE_LOCKED)));
        assert!(is_index_busy(
            &sqlite(rusqlite::ffi::SQLITE_BUSY).context("opening a transaction")
        ));
        assert!(!is_index_busy(&sqlite(rusqlite::ffi::SQLITE_CORRUPT)));
        assert!(!is_index_busy(&anyhow::anyhow!("entry abc: parsing")));
    }

    #[test]
    fn reasons_have_distinct_labels() {
        let labels: std::collections::HashSet<_> = [
            SkipReason::Busy,
            SkipReason::UnknownLoad,
            SkipReason::BackingOff,
            SkipReason::TooManyEntries,
            SkipReason::GcRunning,
            SkipReason::OutOfTimeRecently,
        ]
        .map(SkipReason::label)
        .into_iter()
        .collect();
        assert_eq!(labels.len(), 6);
        assert!(labels.iter().all(|label| !label.is_empty()));
    }

    #[test]
    fn only_refusals_warn_and_only_repairs_inform() {
        let healed = Outcome::Healed {
            probe: BlobRefcountDrift::default(),
            repaired: BlobIndexDrift::default(),
            sweep: Sweep::Failed,
            elapsed: Duration::ZERO,
        };
        let failed = Outcome::Failed {
            reason: String::new(),
        };
        assert!(failed.is_refusal() && !failed.is_noteworthy());
        assert!(healed.is_noteworthy() && !healed.is_refusal());
        let too_many = Outcome::Skipped(SkipReason::TooManyEntries);
        assert!(too_many.is_noteworthy() && !too_many.is_refusal());
        let out_of_time = Outcome::OutOfTime { read: 1, total: 2 };
        assert!(out_of_time.is_noteworthy() && !out_of_time.is_refusal());
        let skipped = Outcome::Skipped(SkipReason::OutOfTimeRecently);
        assert!(!skipped.is_noteworthy() && !skipped.is_refusal());
        for quiet in [
            Outcome::NotNeeded,
            Outcome::IndexBusy,
            Outcome::Skipped(SkipReason::Busy),
            Outcome::Skipped(SkipReason::UnknownLoad),
            Outcome::Skipped(SkipReason::BackingOff),
            Outcome::Skipped(SkipReason::GcRunning),
        ] {
            assert!(!quiet.is_refusal() && !quiet.is_noteworthy(), "{quiet:?}");
        }
    }

    #[test]
    fn outcomes_describe_what_happened() {
        assert_eq!(Outcome::NotNeeded.describe(), "blob index heal not needed");
        assert_eq!(
            Outcome::Skipped(SkipReason::GcRunning).describe(),
            "blob index heal deferred: a GC holds gc.lock"
        );
        assert_eq!(
            Outcome::IndexBusy.describe(),
            "blob index heal found the index busy; will retry"
        );
        let healed = |sweep| Outcome::Healed {
            probe: BlobRefcountDrift {
                unowned: 1_430,
                unowned_bytes: 27 << 30,
                too_high: 571,
                too_low: 2,
                unindexed: 1,
                ..Default::default()
            },
            repaired: BlobIndexDrift {
                entry_mappings: 3,
                blobs: 2_004,
            },
            sweep,
            elapsed: Duration::from_millis(1500),
        };
        assert_eq!(
            healed(Sweep::Swept {
                files: 1_430,
                bytes: 1 << 20
            })
            .describe(),
            format!(
                "blob index healed in 1500 ms: 1430 blobs ({}) had no owner, 574 refcounts were off; \
                 rewrote 2004 blob rows and 3 mappings; swept 1430 blob files ({})",
                crate::report::format_bytes(27 << 30),
                crate::report::format_bytes(1 << 20)
            )
        );
        assert!(
            healed(Sweep::Failed)
                .describe()
                .ends_with("3 mappings; the blob sweep failed and is left to the next GC")
        );
        assert!(
            healed(Sweep::LeftToGc)
                .describe()
                .ends_with("3 mappings; unowned blob files are left to this GC's sweep")
        );
        assert_eq!(
            Outcome::OutOfTime {
                read: 65_000,
                total: 90_000
            }
            .describe(),
            "blob index heal in GC stopped after 2 s, having read 65000 of 90000 entries, \
             and changed nothing; GC skips it for 6 h unless the store shrinks to 65000 \
             entries, and it runs unbounded once builds are idle"
        );
        assert_eq!(
            Outcome::Skipped(SkipReason::OutOfTimeRecently).describe(),
            "blob index heal deferred: the last heal in GC ran out of time on a store this size"
        );
        assert_eq!(
            Outcome::Failed {
                reason: "entry abc: parsing authoritative meta.json".to_string()
            }
            .describe(),
            "blob index heal refused, next attempt in 6 h: entry abc: parsing authoritative meta.json. \
             `kache doctor --repair` removes a corrupt entry and rebuilds the blob index"
        );
    }

    /// Two entries sharing one blob, with its refcount inflated to 41 and two
    /// unowned blob rows: a stale file past the sweep grace and a fresh one.
    #[cfg(unix)]
    struct Drifted {
        config: Config,
        shared: String,
        stale: PathBuf,
        fresh: PathBuf,
    }

    #[cfg(unix)]
    fn unowned_blob(store: &Store, hash: &str, content: &[u8], age: Duration) -> PathBuf {
        let path = store.blob_path(hash);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let file = std::fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(std::time::SystemTime::now() - age)
            .unwrap();
        store
            .file_hash_cache()
            .db()
            .execute(
                "INSERT INTO blobs (hash, size, refcount) VALUES (?1, ?2, 3)",
                rusqlite::params![hash, content.len() as i64],
            )
            .unwrap();
        path
    }

    #[cfg(unix)]
    fn drifted_store(dir: &Path) -> Drifted {
        let config = crate::test_support::test_config(dir.join("cache"));
        let store = Store::open(&config).unwrap();
        for key in ["heal_a", "heal_b"] {
            let output = dir.join(format!("{key}.rlib"));
            std::fs::write(&output, PAYLOAD).unwrap();
            store
                .put(
                    key,
                    "heallib",
                    &["lib".to_string()],
                    &[],
                    "host",
                    "dev",
                    &[(output, format!("lib{key}.rlib"))],
                    "",
                    "",
                )
                .unwrap();
        }
        let shared = store.get("heal_a").unwrap().unwrap().files[0].hash.clone();
        store
            .file_hash_cache()
            .db()
            .execute(
                "UPDATE blobs SET refcount = 41 WHERE hash = ?1",
                rusqlite::params![shared],
            )
            .unwrap();
        let stale = unowned_blob(&store, &"f".repeat(64), STALE, 2 * 3600 * SEC);
        let fresh = unowned_blob(&store, &"e".repeat(64), FRESH, 1800 * SEC);
        Drifted {
            config,
            shared,
            stale,
            fresh,
        }
    }

    #[cfg(unix)]
    fn probe(config: &Config) -> BlobRefcountDrift {
        Store::open(config).unwrap().blob_refcount_drift().unwrap()
    }

    #[cfg(unix)]
    fn refcount(config: &Config, hash: &str) -> i64 {
        Store::open(config)
            .unwrap()
            .file_hash_cache()
            .db()
            .query_row(
                "SELECT refcount FROM blobs WHERE hash = ?1",
                rusqlite::params![hash],
                |row| row.get(0),
            )
            .unwrap()
    }

    #[cfg(unix)]
    const DRIFT: BlobRefcountDrift = BlobRefcountDrift {
        unowned: 2,
        unowned_bytes: (STALE.len() + FRESH.len()) as u64,
        too_high: 1,
        too_high_bytes: PAYLOAD.len() as u64,
        too_low: 0,
        too_low_bytes: 0,
        unindexed: 0,
    };

    #[cfg(unix)]
    fn assert_untouched(drifted: &Drifted) {
        assert_eq!(probe(&drifted.config), DRIFT);
        assert_eq!(refcount(&drifted.config, &drifted.shared), 41);
        assert!(drifted.stale.exists() && drifted.fresh.exists());
    }

    #[cfg(unix)]
    fn hold_permit(config: &Config) -> StoreLock {
        let permits = config.cache_dir.join("scheduler").join("permits");
        std::fs::create_dir_all(&permits).unwrap();
        StoreLock::try_acquire(&permits.join("0")).unwrap().unwrap()
    }

    #[test]
    fn clean_store_is_left_alone_and_a_stale_record_is_cleared() {
        let dir = tempfile::tempdir().unwrap();
        let config = crate::test_support::test_config(dir.path().to_path_buf());
        drop(Store::open(&config).unwrap());
        record_failure(dir.path(), NOW, "old");

        let clock = RequestClock::idle();
        assert_eq!(
            attempt(&config, Trigger::Periodic(&clock), NOW).unwrap(),
            Outcome::NotNeeded
        );
        assert!(!state_path(dir.path()).exists());
        assert_eq!(run(&config, Trigger::Shutdown), Some(Outcome::NotNeeded));
    }

    #[test]
    #[cfg(unix)]
    fn quiet_machine_heals_refcounts_and_sweeps_past_the_grace() {
        let dir = tempfile::tempdir().unwrap();
        let drifted = drifted_store(dir.path());
        assert_untouched(&drifted);
        record_failure(&drifted.config.cache_dir, NOW - 21_600, "an old failure");

        // A recent request defers.
        let recent = RequestClock::new();
        assert_eq!(
            attempt(&drifted.config, Trigger::Periodic(&recent), NOW).unwrap(),
            Outcome::Skipped(SkipReason::Busy)
        );
        assert_untouched(&drifted);

        let clock = RequestClock::idle();
        let Outcome::Healed {
            probe: seen,
            repaired,
            sweep,
            ..
        } = attempt(&drifted.config, Trigger::Periodic(&clock), NOW).unwrap()
        else {
            panic!("a quiet machine must heal");
        };
        assert_eq!(seen, DRIFT);
        assert_eq!(
            repaired,
            BlobIndexDrift {
                entry_mappings: 0,
                blobs: 3
            }
        );
        assert_eq!(
            sweep,
            Sweep::Swept {
                files: 1,
                bytes: STALE.len() as u64
            }
        );
        assert_eq!(refcount(&drifted.config, &drifted.shared), 2);
        assert_eq!(probe(&drifted.config), BlobRefcountDrift::default());
        assert!(!drifted.stale.exists());
        assert!(drifted.fresh.exists(), "inside the grace");
        assert!(!state_path(&drifted.config.cache_dir).exists());
        // Both entries still restore, and the next check has nothing to do.
        let store = Store::open(&drifted.config).unwrap();
        assert!(store.get("heal_a").unwrap().is_some());
        assert!(store.get("heal_b").unwrap().is_some());
        assert_eq!(
            run(&drifted.config, Trigger::Periodic(&clock)),
            Some(Outcome::NotNeeded)
        );
    }

    /// #1206: a host that is never quiet still gets the repair from its GC.
    #[test]
    #[cfg(unix)]
    fn gc_heals_while_builds_run_and_leaves_the_sweep_to_the_gc() {
        let dir = tempfile::tempdir().unwrap();
        let drifted = drifted_store(dir.path());
        let permit = hold_permit(&drifted.config);
        let Outcome::Healed {
            probe: seen,
            repaired,
            sweep,
            ..
        } = attempt_in_gc(&drifted.config, NOW, GC_BUDGET).unwrap()
        else {
            panic!("a GC sweep must heal despite running builds");
        };
        assert_eq!(seen, DRIFT);
        assert_eq!(
            repaired,
            BlobIndexDrift {
                entry_mappings: 0,
                blobs: 3
            }
        );
        assert_eq!(sweep, Sweep::LeftToGc);
        assert_eq!(refcount(&drifted.config, &drifted.shared), 2);
        assert_eq!(probe(&drifted.config), BlobRefcountDrift::default());
        assert!(drifted.stale.exists(), "the GC's own sweep unlinks it");
        assert_eq!(
            attempt_in_gc(&drifted.config, NOW, GC_BUDGET).unwrap(),
            Outcome::NotNeeded
        );
        drop(permit);
    }

    #[test]
    #[cfg(unix)]
    fn gc_heal_past_its_budget_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let drifted = drifted_store(dir.path());
        let cache_dir = &drifted.config.cache_dir;
        assert_eq!(
            attempt_in_gc(&drifted.config, NOW, Duration::ZERO).unwrap(),
            Outcome::OutOfTime { read: 0, total: 2 }
        );
        assert_untouched(&drifted);
        assert!(
            !state_path(cache_dir).exists(),
            "running out of time is not a failure"
        );
        assert_eq!(
            read_gc_overrun(cache_dir),
            Some(GcOverrun { at: NOW, read: 0 })
        );

        // The next sweeps skip it, and take no lock for it.
        for now in [NOW, NOW + 21_599] {
            assert_eq!(
                attempt_in_gc(&drifted.config, now, GC_BUDGET).unwrap(),
                Outcome::Skipped(SkipReason::OutOfTimeRecently)
            );
        }
        assert_untouched(&drifted);

        // After the window it tries again, and a heal clears the record.
        assert!(matches!(
            attempt_in_gc(&drifted.config, NOW + 21_600, GC_BUDGET).unwrap(),
            Outcome::Healed { .. }
        ));
        assert_eq!(read_gc_overrun(cache_dir), None);
    }

    #[test]
    #[cfg(unix)]
    fn gc_heal_retries_once_the_store_fits_what_it_read() {
        let dir = tempfile::tempdir().unwrap();
        let drifted = drifted_store(dir.path());
        let cache_dir = &drifted.config.cache_dir;
        // The overrun read one of the two entries in time.
        record_gc_overrun(cache_dir, GcOverrun { at: NOW, read: 1 });
        assert_eq!(
            attempt_in_gc(&drifted.config, NOW, GC_BUDGET).unwrap(),
            Outcome::Skipped(SkipReason::OutOfTimeRecently)
        );
        Store::open(&drifted.config)
            .unwrap()
            .remove_entry("heal_b")
            .unwrap();
        assert!(matches!(
            attempt_in_gc(&drifted.config, NOW, GC_BUDGET).unwrap(),
            Outcome::Healed { .. }
        ));
        assert_eq!(read_gc_overrun(cache_dir), None);
    }

    #[test]
    fn an_overrun_holds_for_six_hours_while_the_store_is_as_large() {
        let overrun = Some(GcOverrun { at: NOW, read: 100 });
        assert!(gc_overrun_holds(overrun, NOW, 101));
        assert!(gc_overrun_holds(overrun, NOW + 21_599, 101));
        assert!(!gc_overrun_holds(overrun, NOW + 21_600, 101), "expired");
        assert!(!gc_overrun_holds(overrun, NOW, 100), "fits what it read");
        assert!(!gc_overrun_holds(overrun, NOW - 1, 101), "from the future");
        assert!(!gc_overrun_holds(None, NOW, 101));
    }

    #[test]
    fn a_clean_store_clears_an_overrun() {
        let dir = tempfile::tempdir().unwrap();
        let config = crate::test_support::test_config(dir.path().to_path_buf());
        drop(Store::open(&config).unwrap());
        record_gc_overrun(dir.path(), GcOverrun { at: NOW, read: 0 });
        assert_eq!(run_in_gc(&config), Some(Outcome::NotNeeded));
        assert_eq!(read_gc_overrun(dir.path()), None);
    }

    #[test]
    #[cfg(unix)]
    fn gc_heal_waits_out_a_failure_and_records_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let drifted = drifted_store(dir.path());
        let cache_dir = &drifted.config.cache_dir;
        record_failure(cache_dir, NOW, "an earlier refusal");
        assert_eq!(
            attempt_in_gc(&drifted.config, NOW + 21_599, GC_BUDGET).unwrap(),
            Outcome::Skipped(SkipReason::BackingOff)
        );
        assert_untouched(&drifted);

        let meta = Store::open(&drifted.config)
            .unwrap()
            .entry_dir("heal_b")
            .join("meta.json");
        std::fs::write(&meta, b"{truncated").unwrap();
        assert!(matches!(
            attempt_in_gc(&drifted.config, NOW + 21_600, GC_BUDGET).unwrap(),
            Outcome::Failed { .. }
        ));
        assert_eq!(read_failure(cache_dir).unwrap().failed_at, NOW + 21_600);
        assert_untouched(&drifted);
    }

    #[test]
    #[cfg(unix)]
    fn gc_heal_yields_to_a_held_index_without_backing_off() {
        let dir = tempfile::tempdir().unwrap();
        let drifted = drifted_store(dir.path());
        let writer = rusqlite::Connection::open(drifted.config.index_db_path()).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        let started = Instant::now();
        assert_eq!(
            attempt_in_gc(&drifted.config, NOW, GC_BUDGET).unwrap(),
            Outcome::IndexBusy
        );
        assert!(started.elapsed() < Duration::from_secs(4));
        assert!(!state_path(&drifted.config.cache_dir).exists());
        writer.execute_batch("ROLLBACK").unwrap();
        assert_untouched(&drifted);
    }

    #[test]
    fn gc_heal_leaves_a_clean_store_alone() {
        let dir = tempfile::tempdir().unwrap();
        let config = crate::test_support::test_config(dir.path().to_path_buf());
        drop(Store::open(&config).unwrap());
        record_failure(dir.path(), NOW, "old");
        assert_eq!(run_in_gc(&config), Some(Outcome::NotNeeded));
        assert!(!state_path(dir.path()).exists());
    }

    #[test]
    #[cfg(unix)]
    fn held_permit_defers_the_heal_however_long_it_lasts() {
        let dir = tempfile::tempdir().unwrap();
        let drifted = drifted_store(dir.path());
        let permit = hold_permit(&drifted.config);
        let clock = RequestClock::idle();
        for now in [NOW, NOW + 21_600, NOW + 10 * 21_600] {
            assert_eq!(
                attempt(&drifted.config, Trigger::Periodic(&clock), now).unwrap(),
                Outcome::Skipped(SkipReason::Busy)
            );
        }
        assert_eq!(
            attempt(&drifted.config, Trigger::Shutdown, NOW).unwrap(),
            Outcome::Skipped(SkipReason::Busy)
        );
        assert_untouched(&drifted);
        assert!(!state_path(&drifted.config.cache_dir).exists());
        drop(permit);
    }

    #[test]
    #[cfg(unix)]
    fn shutdown_heals_despite_a_recent_request() {
        let dir = tempfile::tempdir().unwrap();
        let drifted = drifted_store(dir.path());
        let outcome = run(&drifted.config, Trigger::Shutdown);
        assert!(
            matches!(outcome, Some(Outcome::Healed { .. })),
            "{outcome:?}"
        );
        assert_eq!(refcount(&drifted.config, &drifted.shared), 2);
        assert_eq!(probe(&drifted.config), BlobRefcountDrift::default());
    }

    #[test]
    #[cfg(unix)]
    fn running_gc_defers_the_heal() {
        let dir = tempfile::tempdir().unwrap();
        let drifted = drifted_store(dir.path());
        let gc_store = Store::open(&drifted.config).unwrap();
        let gc_lock = gc_store.try_gc_lock().unwrap().unwrap();
        let clock = RequestClock::idle();

        assert_eq!(
            attempt(&drifted.config, Trigger::Periodic(&clock), NOW).unwrap(),
            Outcome::Skipped(SkipReason::GcRunning)
        );
        assert_untouched(&drifted);
        assert!(!state_path(&drifted.config.cache_dir).exists());

        drop(gc_lock);
        assert!(matches!(
            attempt(&drifted.config, Trigger::Periodic(&clock), NOW).unwrap(),
            Outcome::Healed { .. }
        ));
        // The lock was released: a GC can run.
        assert!(gc_store.try_gc_lock().unwrap().is_some());
    }

    #[test]
    #[cfg(unix)]
    fn contended_index_yields_without_backing_off() {
        let dir = tempfile::tempdir().unwrap();
        let drifted = drifted_store(dir.path());
        let writer = rusqlite::Connection::open(drifted.config.index_db_path()).unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();

        let clock = RequestClock::idle();
        let started = Instant::now();
        assert_eq!(
            attempt(&drifted.config, Trigger::Periodic(&clock), NOW).unwrap(),
            Outcome::IndexBusy
        );
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "the heal must not wait out the busy timeout"
        );
        assert!(!state_path(&drifted.config.cache_dir).exists());
        writer.execute_batch("ROLLBACK").unwrap();
        assert_untouched(&drifted);
    }

    #[test]
    #[cfg(unix)]
    fn corrupt_entry_fails_closed_and_backs_off() {
        let dir = tempfile::tempdir().unwrap();
        let drifted = drifted_store(dir.path());
        let cache_dir = &drifted.config.cache_dir;
        let meta = Store::open(&drifted.config)
            .unwrap()
            .entry_dir("heal_b")
            .join("meta.json");
        let good = std::fs::read(&meta).unwrap();
        std::fs::write(&meta, b"{truncated").unwrap();

        let clock = RequestClock::idle();
        let Some(Outcome::Failed { reason }) = run(&drifted.config, Trigger::Periodic(&clock))
        else {
            panic!("a corrupt entry must fail the heal");
        };
        assert!(reason.contains("entry heal_b"), "{reason}");
        assert_untouched(&drifted);
        let recorded = read_failure(cache_dir).unwrap();
        assert_eq!(recorded.reason, reason);
        assert!(recorded.failed_at >= unix_now_secs() - 60);

        // The write lock is not taken again inside the window.
        record_failure(cache_dir, NOW, &reason);
        for now in [NOW, NOW + 21_599] {
            assert_eq!(
                attempt(&drifted.config, Trigger::Periodic(&clock), now).unwrap(),
                Outcome::Skipped(SkipReason::BackingOff)
            );
        }
        assert_eq!(read_failure(cache_dir).unwrap().failed_at, NOW);

        // After it, one more attempt, recorded at its own time.
        assert!(matches!(
            attempt(&drifted.config, Trigger::Periodic(&clock), NOW + 21_600).unwrap(),
            Outcome::Failed { .. }
        ));
        assert_eq!(read_failure(cache_dir).unwrap().failed_at, NOW + 21_600);
        assert_untouched(&drifted);

        // A repaired entry heals on the next attempt and clears the record.
        std::fs::write(&meta, good).unwrap();
        assert!(matches!(
            attempt(&drifted.config, Trigger::Periodic(&clock), NOW + 2 * 21_600).unwrap(),
            Outcome::Healed { .. }
        ));
        assert!(!state_path(cache_dir).exists());
        assert_eq!(refcount(&drifted.config, &drifted.shared), 2);
    }
}
