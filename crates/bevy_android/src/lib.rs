//! Provides Android functionality for Bevy Engine.

#![cfg(target_os = "android")]

pub use android_activity;

use std::sync::Mutex;

use android_activity::AndroidApp;

static ANDROID_APP: Mutex<Option<AndroidApp>> = Mutex::new(None);

/// Replaces the current Android application handle.
pub fn set(app: AndroidApp) {
    let previous = ANDROID_APP
        .lock()
        .expect("Android application storage lock was poisoned")
        .replace(app);
    drop(previous);
}

/// Returns the current Android application handle.
pub fn get() -> Option<AndroidApp> {
    ANDROID_APP
        .lock()
        .expect("Android application storage lock was poisoned")
        .clone()
}
