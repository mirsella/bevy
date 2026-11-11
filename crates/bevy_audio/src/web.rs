//! Browser playback of encoded audio on the web.
//!
//! Rodio decodes and mixes on the page's main thread, which the app also renders on. The browser
//! decodes off it: loops stream through an `<audio>` element, and other sounds play Web Audio
//! buffers decoded once per asset. Every voice ends in its own gain node. Components must be `Send`, so the JavaScript objects live in
//! a thread-local registry and sinks hold an id.

use crate::{AudioSink, Decodable, PlaybackMode, PlaybackSettings};
use alloc::rc::Rc;
use bevy_asset::{Asset, AssetEvent, Handle, UntypedAssetId};
use bevy_ecs::message::MessageReader;
use core::{
    cell::{Cell, RefCell},
    time::Duration,
};
use js_sys::{Array, Reflect, Uint8Array};
use rodio::source::SeekError;
use std::collections::HashMap;
use tracing::warn;
use wasm_bindgen::{closure::Closure, JsCast, JsValue};
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{
    AudioBuffer, AudioBufferSourceNode, AudioContext, AudioContextState, AudioScheduledSourceNode,
    Blob, Event, GainNode, HtmlAudioElement, MediaElementAudioSourceNode, Url,
};

thread_local! {
    static BROWSER: RefCell<Browser> = RefCell::default();
}

#[derive(Default)]
struct Browser {
    context: Option<AudioContext>,
    buffers: HashMap<UntypedAssetId, Decoded>,
    voices: HashMap<u64, Voice>,
    next_id: u64,
    next_decode: u64,
    _retry: Option<Closure<dyn FnMut(Event)>>,
}

enum Decoded {
    /// Voices waiting for the decode with this id.
    Pending(u64, Vec<u64>),
    Ready(AudioBuffer),
    Failed,
}

/// Creates the audio context up front, like rodio's output, so a page that resumes contexts on
/// user input can unlock it before the first sound.
pub(crate) fn init() {
    if BROWSER.with_borrow(|browser| browser.context.is_some()) {
        return;
    }
    let context = AudioContext::new()
        .inspect_err(|error| warn!(?error, "browser audio is unavailable"))
        .ok();
    // Autoplay rules may reject a track until the user interacts with the page.
    let retry = Closure::<dyn FnMut(Event)>::new(|_event: Event| {
        BROWSER.with_borrow(|browser| {
            for voice in browser.voices.values() {
                if let Kind::Track(track) = &voice.kind
                    && !voice.paused
                    && !track.stopped
                    && track.blocked.get()
                {
                    track.play();
                }
            }
        });
    });
    if let Some(window) = web_sys::window() {
        for event in ["pointerup", "keydown", "touchend"] {
            let _ = window.add_event_listener_with_callback(event, retry.as_ref().unchecked_ref());
        }
    }
    BROWSER.with_borrow_mut(|browser| {
        browser.context = context;
        browser._retry = Some(retry);
    });
}

/// Starts `source` in the browser, or returns `None` for audio it cannot play.
pub(crate) fn sink<S: Decodable + Asset>(
    source: &S,
    handle: &Handle<S>,
    settings: &PlaybackSettings,
) -> Option<AudioSink> {
    let looped = match settings.mode {
        // An element loops the whole clip only.
        PlaybackMode::Loop => {
            if settings.start_position.is_some() || settings.duration.is_some() {
                return None;
            }
            true
        }
        PlaybackMode::Once | PlaybackMode::Despawn | PlaybackMode::Remove => false,
    };
    let bytes = source.encoded()?;
    let asset = handle.id().untyped();
    let id = BROWSER.with_borrow_mut(|browser| {
        let id = browser.next_id;
        let voice = browser
            .voice(id, asset, bytes, settings, looped)
            .inspect_err(|error| warn!(?error, "browser could not start audio"))
            .ok()?;
        browser.next_id += 1;
        browser.voices.insert(id, voice);
        Some(id)
    })?;
    Some(AudioSink {
        sink: Player::Browser(BrowserPlayer(id)),
        managed_volume: None,
    })
}

impl Browser {
    fn voice(
        &mut self,
        id: u64,
        asset: UntypedAssetId,
        bytes: &[u8],
        settings: &PlaybackSettings,
        looped: bool,
    ) -> Result<Voice, JsValue> {
        let context = self
            .context
            .clone()
            .ok_or_else(|| JsValue::from_str("no audio context"))?;
        // Volume goes through a gain node because iOS ignores media element volume.
        let gain = context.create_gain()?;
        gain.connect_with_audio_node(&context.destination())?;
        let kind = if looped {
            Track::new(&context, &gain, bytes, settings.paused).map(Kind::Track)
        } else {
            self.buffer(&context, &gain, id, asset, bytes, settings)
                .map(Kind::Buffer)
        };
        let kind = kind.inspect_err(|_| {
            let _ = gain.disconnect();
        })?;
        Ok(Voice {
            gain,
            volume: 1.0,
            speed: 1.0,
            paused: settings.paused,
            kind,
        })
    }

    fn buffer(
        &mut self,
        context: &AudioContext,
        gain: &GainNode,
        id: u64,
        asset: UntypedAssetId,
        bytes: &[u8],
        settings: &PlaybackSettings,
    ) -> Result<Buffer, JsValue> {
        let ended = Closure::<dyn FnMut(Event)>::new(move |_event: Event| {
            BROWSER.with_borrow_mut(|browser| {
                if let Some(Voice {
                    kind: Kind::Buffer(buffer),
                    speed,
                    ..
                }) = browser.voices.get_mut(&id)
                {
                    buffer.rebase(*speed);
                    buffer.node = None;
                    buffer.finished = true;
                }
            });
        });
        let mut buffer = Buffer {
            context: context.clone(),
            node: None,
            buffer: None,
            ended,
            start: settings.start_position.unwrap_or_default().as_secs_f64(),
            offset: settings.start_position.unwrap_or_default().as_secs_f64(),
            played: 0.0,
            started_at: 0.0,
            end: settings
                .duration
                .map(|duration| settings.start_position.unwrap_or_default() + duration)
                .map(|end| end.as_secs_f64()),
            // A suspended context would queue sounds and play them all at once on resume.
            finished: context.state() != AudioContextState::Running,
        };
        if buffer.finished {
            return Ok(buffer);
        }
        match self.buffers.get_mut(&asset) {
            Some(Decoded::Ready(decoded)) => {
                buffer.buffer = Some(decoded.clone());
                if !settings.paused {
                    buffer.start(1.0, gain);
                }
            }
            Some(Decoded::Pending(_, waiting)) => waiting.push(id),
            Some(Decoded::Failed) => buffer.finished = true,
            None => {
                let decode_id = self.next_decode;
                decode(context, asset, decode_id, bytes)?;
                self.next_decode += 1;
                self.buffers
                    .insert(asset, Decoded::Pending(decode_id, vec![id]));
            }
        }
        Ok(buffer)
    }

    fn decoded(
        &mut self,
        asset: UntypedAssetId,
        decode_id: u64,
        result: Result<AudioBuffer, JsValue>,
    ) {
        // A changed or removed asset replaced or dropped its entry; a stale decode is discarded.
        if !matches!(self.buffers.get(&asset), Some(Decoded::Pending(pending, _)) if *pending == decode_id)
        {
            return;
        }
        let Some(Decoded::Pending(_, waiting)) = self.buffers.remove(&asset) else {
            unreachable!("checked above");
        };
        let decoded = result
            .inspect_err(|error| warn!(?error, ?asset, "browser could not decode audio"))
            .ok();
        for id in waiting {
            let Some(voice) = self.voices.get_mut(&id) else {
                continue;
            };
            let Kind::Buffer(buffer) = &mut voice.kind else {
                unreachable!("only buffer voices wait for decodes");
            };
            match &decoded {
                Some(decoded) => {
                    buffer.buffer = Some(decoded.clone());
                    if !voice.paused {
                        buffer.start(voice.speed, &voice.gain);
                    }
                }
                None => buffer.finished = true,
            }
        }
        self.buffers
            .insert(asset, decoded.map_or(Decoded::Failed, Decoded::Ready));
    }
}

fn decode(
    context: &AudioContext,
    asset: UntypedAssetId,
    decode_id: u64,
    bytes: &[u8],
) -> Result<(), JsValue> {
    let promise = context.decode_audio_data(&Uint8Array::from(bytes).buffer())?;
    spawn_local(async move {
        let result = JsFuture::from(promise).await.map(JsCast::unchecked_into);
        BROWSER.with_borrow_mut(|browser| browser.decoded(asset, decode_id, result));
    });
    Ok(())
}

/// Forgets decodes of changed or removed assets. Voices waiting on a forgotten decode finish
/// unplayed, so their despawn or removal still happens.
pub(crate) fn forget_changed_sources<T: Asset>(mut events: MessageReader<AssetEvent<T>>) {
    for event in events.read() {
        if let AssetEvent::Modified { id } | AssetEvent::Removed { id } = event {
            BROWSER.with_borrow_mut(|browser| {
                let Some(Decoded::Pending(_, waiting)) = browser.buffers.remove(&id.untyped())
                else {
                    return;
                };
                for id in waiting {
                    if let Some(Voice {
                        kind: Kind::Buffer(buffer),
                        ..
                    }) = browser.voices.get_mut(&id)
                    {
                        buffer.finished = true;
                    }
                }
            });
        }
    }
}

struct Voice {
    gain: GainNode,
    volume: f32,
    speed: f32,
    paused: bool,
    kind: Kind,
}

enum Kind {
    Buffer(Buffer),
    Track(Track),
}

struct Buffer {
    context: AudioContext,
    /// The node playing now; a paused or finished voice has none.
    node: Option<AudioBufferSourceNode>,
    /// `None` until decoded.
    buffer: Option<AudioBuffer>,
    ended: Closure<dyn FnMut(Event)>,
    /// Clip position in seconds where sink time zero is.
    start: f64,
    /// Clip position in seconds when the node started or playback paused.
    offset: f64,
    /// Sink position at `offset`. Like rodio, sink time is clip time divided by the speed it
    /// played at, and starts at zero even with a start position.
    played: f64,
    /// Context time in seconds when the node started.
    started_at: f64,
    /// Clip position in seconds where playback ends; the clip's end if `None`.
    end: Option<f64>,
    finished: bool,
}

impl Buffer {
    fn start(&mut self, speed: f32, gain: &GainNode) {
        let Some(buffer) = self.buffer.clone() else {
            return;
        };
        let end = self.end.unwrap_or_else(|| buffer.duration());
        if self.finished || self.node.is_some() || self.offset >= end {
            self.finished |= self.offset >= end;
            return;
        }
        let started = (|| {
            let node = self.context.create_buffer_source()?;
            node.set_buffer(Some(&buffer));
            node.playback_rate().set_value(speed);
            node.connect_with_audio_node(gain)?;
            AudioScheduledSourceNode::set_onended(&node, Some(self.ended.as_ref().unchecked_ref()));
            node.start_with_when_and_grain_offset_and_grain_duration(
                0.0,
                self.offset,
                end - self.offset,
            )?;
            Ok::<_, JsValue>(node)
        })();
        match started {
            Ok(node) => {
                self.started_at = self.context.current_time();
                self.node = Some(node);
            }
            Err(error) => {
                warn!(?error, "browser could not play audio");
                self.finished = true;
            }
        }
    }

    fn stop_node(&mut self) {
        if let Some(node) = self.node.take() {
            let node: &AudioScheduledSourceNode = &node;
            // Only natural ends finish the voice.
            node.set_onended(None);
            let _ = node.stop();
        }
    }

    /// Clip and sink seconds now.
    fn position(&self, speed: f32) -> (f64, f64) {
        if self.node.is_none() {
            return (self.offset, self.played);
        }
        let speed = f64::from(speed);
        let end = self
            .end
            .or_else(|| self.buffer.as_ref().map(AudioBuffer::duration));
        let mut elapsed = self.context.current_time() - self.started_at;
        if let Some(end) = end {
            elapsed = elapsed.min((end - self.offset) / speed);
        }
        (self.offset + elapsed * speed, self.played + elapsed)
    }

    /// Restarts position accounting from now, as before a speed change.
    fn rebase(&mut self, speed: f32) {
        (self.offset, self.played) = self.position(speed);
        self.started_at = self.context.current_time();
    }
}

struct Track {
    audio: HtmlAudioElement,
    source: MediaElementAudioSourceNode,
    url: String,
    blocked: Rc<Cell<bool>>,
    failed: Rc<Cell<bool>>,
    stopped: bool,
    _error: Closure<dyn FnMut(Event)>,
}

impl Track {
    fn new(
        context: &AudioContext,
        gain: &GainNode,
        bytes: &[u8],
        paused: bool,
    ) -> Result<Self, JsValue> {
        let blob = Blob::new_with_u8_array_sequence(&Array::of1(&Uint8Array::from(bytes)))?;
        let url = Url::create_object_url_with_blob(&blob)?;
        let routed = (|| {
            let audio = HtmlAudioElement::new_with_src(&url)?;
            let source = context.create_media_element_source(&audio)?;
            source.connect_with_audio_node(gain)?;
            Ok::<_, JsValue>((audio, source))
        })();
        let (audio, source) = routed.inspect_err(|_| {
            let _ = Url::revoke_object_url(&url);
        })?;
        let failed = Rc::<Cell<bool>>::default();
        let error_failed = failed.clone();
        let error = Closure::<dyn FnMut(Event)>::new(move |_event: Event| {
            warn!("browser could not play an audio track");
            error_failed.set(true);
        });
        audio.set_onerror(Some(error.as_ref().unchecked_ref()));
        audio.set_loop(true);
        // Rodio and buffer voices change pitch with speed.
        let _ = Reflect::set(&audio, &"preservesPitch".into(), &false.into());
        let track = Self {
            audio,
            source,
            url,
            blocked: Rc::default(),
            failed,
            stopped: false,
            _error: error,
        };
        if !paused {
            track.play();
        }
        Ok(track)
    }

    fn play(&self) {
        self.blocked.set(false);
        let Ok(promise) = self.audio.play() else {
            self.blocked.set(true);
            return;
        };
        let blocked = self.blocked.clone();
        spawn_local(async move {
            // Also rejects when a pause interrupts the start; retrying then is harmless.
            if JsFuture::from(promise).await.is_err() {
                blocked.set(true);
            }
        });
    }
}

impl Drop for Voice {
    fn drop(&mut self) {
        match &mut self.kind {
            Kind::Buffer(buffer) => buffer.stop_node(),
            Kind::Track(track) => {
                let _ = track.source.disconnect();
                track.audio.set_onerror(None);
                let _ = track.audio.pause();
                // An empty `src` would fire an error; removing it releases the media.
                let _ = track.audio.remove_attribute("src");
                track.audio.load();
                let _ = Url::revoke_object_url(&track.url);
            }
        }
        let _ = self.gain.disconnect();
    }
}

impl Voice {
    fn set_volume(&mut self, volume: f32) {
        // Browsers throw on non-finite values, which would unwind with the registry borrowed.
        if !volume.is_finite() {
            warn!(volume, "ignoring a non-finite audio volume");
            return;
        }
        self.volume = volume;
        self.gain.gain().set_value(volume);
    }

    fn set_speed(&mut self, speed: f32) {
        if !speed.is_finite() {
            warn!(speed, "ignoring a non-finite audio speed");
            return;
        }
        match &mut self.kind {
            Kind::Buffer(buffer) => {
                if let Some(node) = buffer.node.clone() {
                    // Later positions advance at the new speed.
                    buffer.rebase(self.speed);
                    node.playback_rate().set_value(speed);
                }
            }
            Kind::Track(track) => {
                // Elements reject unsupported rates, which vary by browser.
                if let Err(error) =
                    Reflect::set(&track.audio, &"playbackRate".into(), &speed.into())
                {
                    warn!(?error, speed, "browser rejected an audio track speed");
                    return;
                }
            }
        }
        self.speed = speed;
    }

    fn play(&mut self) {
        if !self.paused {
            return;
        }
        self.paused = false;
        match &mut self.kind {
            Kind::Buffer(buffer) => buffer.start(self.speed, &self.gain),
            // Like rodio, a stopped voice stays silent.
            Kind::Track(track) if !track.stopped => track.play(),
            Kind::Track(_) => {}
        }
    }

    fn pause(&mut self) {
        if self.paused {
            return;
        }
        self.paused = true;
        match &mut self.kind {
            Kind::Buffer(buffer) => {
                buffer.rebase(self.speed);
                buffer.stop_node();
            }
            Kind::Track(track) => {
                let _ = track.audio.pause();
            }
        }
    }

    fn position(&self) -> Duration {
        let seconds = match &self.kind {
            Kind::Buffer(buffer) => buffer.position(self.speed).1,
            // A looping element only knows its clip time.
            Kind::Track(track) => track.audio.current_time() / f64::from(self.speed),
        };
        Duration::from_secs_f64(seconds.max(0.0))
    }

    fn seek(&mut self, position: Duration) {
        match &mut self.kind {
            Kind::Buffer(buffer) => {
                let playing = buffer.node.is_some();
                buffer.stop_node();
                // Like rodio, the target is sink time, so the clip moves by speed times as much.
                buffer.played = position.as_secs_f64();
                buffer.offset = buffer.start + buffer.played * f64::from(self.speed);
                if playing {
                    buffer.start(self.speed, &self.gain);
                }
            }
            Kind::Track(track) => track
                .audio
                .set_current_time(position.as_secs_f64() * f64::from(self.speed)),
        }
    }

    fn stop(&mut self) {
        match &mut self.kind {
            Kind::Buffer(buffer) => {
                buffer.stop_node();
                buffer.finished = true;
            }
            Kind::Track(track) => {
                let _ = track.audio.pause();
                track.stopped = true;
            }
        }
    }

    fn empty(&self) -> bool {
        match &self.kind {
            Kind::Buffer(buffer) => buffer.finished,
            Kind::Track(track) => track.stopped || track.failed.get(),
        }
    }
}

/// A browser voice, removed from the registry when dropped.
pub(crate) struct BrowserPlayer(u64);

impl BrowserPlayer {
    fn with<R>(&self, f: impl FnOnce(&mut Voice) -> R) -> R {
        BROWSER.with_borrow_mut(|browser| {
            f(browser
                .voices
                .get_mut(&self.0)
                .expect("a browser player owns its voice"))
        })
    }
}

impl Drop for BrowserPlayer {
    fn drop(&mut self) {
        let voice = BROWSER.with_borrow_mut(|browser| browser.voices.remove(&self.0));
        drop(voice);
    }
}

/// The player behind an [`AudioSink`]: rodio, or the browser for encoded audio.
pub(crate) enum Player {
    Rodio(rodio::Player),
    Browser(BrowserPlayer),
}

impl From<rodio::Player> for Player {
    fn from(player: rodio::Player) -> Self {
        Self::Rodio(player)
    }
}

impl Player {
    pub(crate) fn volume(&self) -> f32 {
        match self {
            Self::Rodio(player) => player.volume(),
            Self::Browser(player) => player.with(|voice| voice.volume),
        }
    }

    pub(crate) fn set_volume(&self, volume: f32) {
        match self {
            Self::Rodio(player) => player.set_volume(volume),
            Self::Browser(player) => player.with(|voice| voice.set_volume(volume)),
        }
    }

    pub(crate) fn speed(&self) -> f32 {
        match self {
            Self::Rodio(player) => player.speed(),
            Self::Browser(player) => player.with(|voice| voice.speed),
        }
    }

    pub(crate) fn set_speed(&self, speed: f32) {
        match self {
            Self::Rodio(player) => player.set_speed(speed),
            Self::Browser(player) => player.with(|voice| voice.set_speed(speed)),
        }
    }

    pub(crate) fn play(&self) {
        match self {
            Self::Rodio(player) => player.play(),
            Self::Browser(player) => player.with(Voice::play),
        }
    }

    pub(crate) fn get_pos(&self) -> Duration {
        match self {
            Self::Rodio(player) => player.get_pos(),
            Self::Browser(player) => player.with(|voice| voice.position()),
        }
    }

    pub(crate) fn try_seek(&self, position: Duration) -> Result<(), SeekError> {
        match self {
            Self::Rodio(player) => player.try_seek(position),
            Self::Browser(player) => {
                player.with(|voice| voice.seek(position));
                Ok(())
            }
        }
    }

    pub(crate) fn pause(&self) {
        match self {
            Self::Rodio(player) => player.pause(),
            Self::Browser(player) => player.with(Voice::pause),
        }
    }

    pub(crate) fn is_paused(&self) -> bool {
        match self {
            Self::Rodio(player) => player.is_paused(),
            Self::Browser(player) => player.with(|voice| voice.paused),
        }
    }

    pub(crate) fn stop(&self) {
        match self {
            Self::Rodio(player) => player.stop(),
            Self::Browser(player) => player.with(Voice::stop),
        }
    }

    pub(crate) fn empty(&self) -> bool {
        match self {
            Self::Rodio(player) => player.empty(),
            Self::Browser(player) => player.with(|voice| voice.empty()),
        }
    }
}
