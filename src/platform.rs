use std::env;

pub fn get_current_username() -> String {
    env::var("USER")
        .or_else(|_| env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Raise this process's soft open-file limit (`RLIMIT_NOFILE`) to its hard
/// limit, so jobs inherit the largest budget the kernel allows rather than the
/// launcher's soft default (a systemd user service starts at 1024, an
/// interactive login shell typically at the hard limit). Raising soft up to
/// hard needs no privilege. Returns `(soft_before, hard)` on success.
#[cfg(unix)]
pub fn raise_open_file_limit() -> std::io::Result<(libc::rlim_t, libc::rlim_t)> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `limit` is a valid, writable rlimit struct for the duration of the call.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let before = limit.rlim_cur;
    if limit.rlim_cur != limit.rlim_max {
        limit.rlim_cur = limit.rlim_max;
        // SAFETY: `limit` is a valid rlimit struct; soft <= hard by construction.
        if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok((before, limit.rlim_max))
}

#[cfg(all(test, unix))]
mod rlimit_tests {
    use super::raise_open_file_limit;

    fn nofile() -> (libc::rlim_t, libc::rlim_t) {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        assert_eq!(
            unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) },
            0
        );
        (limit.rlim_cur, limit.rlim_max)
    }

    #[test]
    fn raises_soft_open_file_limit_to_hard() {
        let (_, hard) = nofile();
        let (_, reported_hard) = raise_open_file_limit().expect("raise should succeed");
        assert_eq!(reported_hard, hard);
        assert_eq!(nofile(), (hard, hard));
        // Idempotent once soft == hard.
        assert_eq!(raise_open_file_limit().unwrap(), (hard, hard));
    }
}
