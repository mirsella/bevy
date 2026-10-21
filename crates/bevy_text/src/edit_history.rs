use alloc::collections::VecDeque;
use bevy_platform::time::Instant;
use core::time::Duration;
use parley::{PlainEditor, PlainEditorDriver};

use crate::{TextBrush, TextEdit};

const HISTORY_LIMIT: usize = 100;
const GROUP_INTERVAL: Duration = Duration::from_secs(1);

/// Per-editor undo and redo history, maintained by [`crate::EditableText`].
///
/// Stores up to 100 text changes with their directed selections. Consecutive typing,
/// backspaces, or forward deletes are grouped until the cursor moves, another kind
/// of edit occurs, or there is a pause of more than one second. Paste, cut, and IME
/// commits are separate changes. Navigation alone never creates an undo step.
///
/// History restores previously accepted text without reapplying input filters or
/// length limits. Call [`Self::clear`] if changing validation rules should prevent
/// restoring older values.
#[derive(Clone, Default)]
pub struct TextEditHistory {
    undo: VecDeque<Snapshot>,
    redo: VecDeque<Snapshot>,
    current: Option<Snapshot>,
    group: Option<(EditKind, Instant)>,
    composition: Option<Snapshot>,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Snapshot {
    text: String,
    anchor: usize,
    focus: usize,
}

impl Snapshot {
    pub(crate) fn capture(editor: &PlainEditor<TextBrush>) -> Self {
        let selection = editor.raw_selection();
        Self {
            text: editor.text().to_string(),
            anchor: selection.anchor().index(),
            focus: selection.focus().index(),
        }
    }

    pub(crate) fn restore(&self, driver: &mut PlainEditorDriver<TextBrush>) {
        driver.editor.set_text(&self.text);
        driver.select_byte_range(self.anchor, self.focus);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EditKind {
    Typing,
    Backspace,
    Delete,
}

impl EditKind {
    pub(crate) fn for_edit(edit: &TextEdit, editor: &PlainEditor<TextBrush>) -> Option<Self> {
        if !editor.raw_selection().is_collapsed() {
            return None;
        }
        match edit {
            TextEdit::Insert(text) if !text.contains(['\n', '\r']) => Some(Self::Typing),
            TextEdit::Backspace => Some(Self::Backspace),
            TextEdit::Delete => Some(Self::Delete),
            _ => None,
        }
    }
}

impl TextEditHistory {
    /// Discards undo, redo, and any active edit group.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Whether there is a text change to undo.
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    /// Whether there is an undone text change to redo.
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub(crate) fn synchronize(&mut self, editor: &PlainEditor<TextBrush>) {
        // Direct writes to the public editor are programmatic replacements, not
        // user edits. Never let undo resurrect a value from before such a reset.
        if !editor.is_composing()
            && (self.composition.is_some()
                || self
                    .current
                    .as_ref()
                    .is_some_and(|snapshot| editor.text() != snapshot.text.as_str()))
        {
            self.clear();
        }
    }

    pub(crate) fn begin_compose(&mut self, editor: &PlainEditor<TextBrush>) {
        self.group = None;
        self.composition
            .get_or_insert_with(|| Snapshot::capture(editor));
    }

    pub(crate) fn cancel_compose(&mut self, driver: &mut PlainEditorDriver<TextBrush>) {
        if let Some(before) = self.composition.take() {
            // Parley replaces the selection when preedit starts. Cancellation
            // must restore that text as well as removing the transient preedit.
            before.restore(driver);
            self.current = Some(before);
        }
        self.group = None;
    }

    pub(crate) fn before_commit(&mut self, driver: &mut PlainEditorDriver<TextBrush>) -> Snapshot {
        if let Some(before) = self.composition.take() {
            // Preedit replaced the original selection. Validate the commit against
            // that selection so rejected input leaves its text intact.
            before.restore(driver);
            before
        } else {
            Snapshot::capture(driver.editor)
        }
    }

    pub(crate) fn record(
        &mut self,
        mut before: Snapshot,
        editor: &PlainEditor<TextBrush>,
        kind: Option<EditKind>,
        now: Instant,
    ) {
        if editor.text() == before.text.as_str() {
            let selection = editor.raw_selection();
            before.anchor = selection.anchor().index();
            before.focus = selection.focus().index();
            self.current = Some(before);
            self.group = None;
            return;
        }

        let merge = kind.is_some_and(|kind| {
            self.group.is_some_and(|(previous, time)| {
                previous == kind && now.duration_since(time) <= GROUP_INTERVAL
            }) && self.current.as_ref() == Some(&before)
        });
        if !merge {
            self.undo.push_back(before);
            if self.undo.len() > HISTORY_LIMIT {
                self.undo.pop_front();
            }
        }
        self.current = Some(Snapshot::capture(editor));
        self.redo.clear();
        self.group = kind.map(|kind| (kind, now));
    }

    pub(crate) fn restore(&mut self, driver: &mut PlainEditorDriver<TextBrush>, redo: bool) {
        // The active IME owns undo/redo until composition ends.
        if driver.editor.is_composing() {
            return;
        }
        self.group = None;
        let (source, destination) = if redo {
            (&mut self.redo, &mut self.undo)
        } else {
            (&mut self.undo, &mut self.redo)
        };
        if let Some(snapshot) = source.pop_back() {
            // Save the live selection, including navigation since the last edit.
            destination.push_back(Snapshot::capture(driver.editor));
            snapshot.restore(driver);
            self.current = Some(snapshot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EditableText, Font, FontCx, LayoutCx};
    use bevy_clipboard::{Clipboard, ClipboardRead};
    use bevy_platform::sync::{Arc, Mutex};

    struct Editor {
        text: EditableText,
        fonts: FontCx,
        layout: LayoutCx,
        clipboard: Clipboard,
    }

    impl Editor {
        fn new(value: &str) -> Self {
            let mut fonts = FontCx::default();
            let font = Font::from_bytes(include_bytes!("FiraMono-subset.ttf").to_vec());
            fonts.collection.register_fonts(font.data, None);
            fonts.set_sans_serif_family("Fira Mono").unwrap();
            let mut editor = Self {
                text: EditableText::new(value),
                fonts,
                layout: LayoutCx::default(),
                clipboard: Clipboard::default(),
            };
            editor.apply([]);
            editor
        }

        fn apply(&mut self, edits: impl IntoIterator<Item = TextEdit>) {
            self.text.pending_edits.extend(edits);
            self.text.apply_pending_edits(
                &mut self.fonts,
                &mut self.layout,
                &mut self.clipboard,
                |_| true,
            );
        }

        fn value(&self) -> String {
            self.text.value().to_string()
        }

        fn select(&mut self, anchor: usize, focus: usize) {
            self.text
                .editor_mut()
                .driver(&mut self.fonts, &mut self.layout)
                .select_byte_range(anchor, focus);
        }
    }

    fn insert(text: &str) -> TextEdit {
        TextEdit::Insert(text.into())
    }

    fn preedit(text: &str) -> TextEdit {
        TextEdit::ImeSetCompose {
            value: text.into(),
            cursor: None,
        }
    }

    #[test]
    fn typing_groups_across_frames_and_preserves_queue_order() {
        let mut editor = Editor::new("");
        editor.apply([insert("a")]);
        editor.apply([insert("b"), insert("c"), TextEdit::Undo]);
        assert_eq!(editor.value(), "");
        editor.apply([TextEdit::Redo]);
        assert_eq!(editor.value(), "abc");
        assert_eq!(editor.text.editor.raw_selection().focus().index(), 3);
    }

    #[test]
    fn navigation_and_deletion_split_groups_without_undoing_navigation() {
        let mut editor = Editor::new("");
        editor.apply([insert("abc"), TextEdit::Left(false), insert("x")]);
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "abc");
        assert_eq!(editor.text.editor.raw_selection().focus().index(), 2);
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "");
        editor.apply([TextEdit::Redo, TextEdit::TextEnd(false)]);
        editor.apply([TextEdit::Backspace, TextEdit::Backspace, TextEdit::Undo]);
        assert_eq!(editor.value(), "abc");
        editor.apply([
            TextEdit::TextStart(false),
            TextEdit::Delete,
            TextEdit::Delete,
        ]);
        assert_eq!(editor.value(), "c");
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "abc");
    }

    #[test]
    fn restores_backward_unicode_selection_and_multiline_text() {
        let mut editor = Editor::new("aé💣\nz");
        editor.select("aé💣".len(), 1);
        editor.apply([insert("replacement\n")]);
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "aé💣\nz");
        let selection = editor.text.editor.raw_selection();
        assert_eq!(selection.anchor().index(), "aé💣".len());
        assert_eq!(selection.focus().index(), 1);
        editor.apply([TextEdit::Redo]);
        assert_eq!(editor.value(), "areplacement\n\nz");
    }

    #[test]
    fn alternating_undo_and_redo_preserves_live_selections() {
        let mut editor = Editor::new("");
        editor.apply([
            insert("a"),
            TextEdit::TextStart(false),
            insert("b"),
            TextEdit::TextEnd(false),
            insert("c"),
        ]);
        assert_eq!(editor.value(), "bac");
        editor.apply([TextEdit::Undo, TextEdit::TextStart(false), TextEdit::Redo]);
        assert_eq!(editor.value(), "bac");
        assert_eq!(editor.text.editor.raw_selection().focus().index(), 3);
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "ba");
        assert_eq!(editor.text.editor.raw_selection().focus().index(), 0);
        editor.apply([TextEdit::Undo, TextEdit::Redo, TextEdit::Redo]);
        assert_eq!(editor.value(), "bac");
        assert_eq!(editor.text.history.undo.len(), 3);
        assert!(editor.text.history.redo.is_empty());
    }

    #[test]
    fn redo_survives_navigation_and_rejected_input_but_not_new_edits() {
        let mut editor = Editor::new("");
        editor.apply([insert("abc"), TextEdit::Undo, TextEdit::TextEnd(false)]);
        editor.text.max_characters = Some(0);
        editor.apply([insert("rejected")]);
        assert!(editor.text.history.can_redo());
        editor.text.max_characters = None;
        editor.apply([insert("new"), TextEdit::Redo]);
        assert_eq!(editor.value(), "new");
        assert!(!editor.text.history.can_redo());
    }

    #[test]
    fn clear_and_direct_replacements_discard_history() {
        let mut editor = Editor::new("");
        editor.apply([insert("private")]);
        editor.text.clear();
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "");
        editor.apply([insert("old")]);
        editor.text.editor_mut().set_text("new account");
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "new account");
        assert!(!editor.text.history.can_undo());
        assert!(!editor.text.history.can_redo());
    }

    #[test]
    fn direct_replacement_during_composition_discards_preedit_baseline() {
        let mut editor = Editor::new("private");
        editor.apply([TextEdit::SelectAll, preedit("composing")]);
        editor.text.editor_mut().set_text("new account");
        editor.apply([preedit(""), TextEdit::Undo]);
        assert_eq!(editor.value(), "new account");
        assert!(!editor.text.history.can_undo());
    }

    #[test]
    fn rejected_ime_commit_preserves_original_selection() {
        let mut editor = Editor::new("123");
        editor.apply([TextEdit::SelectAll, preedit("not digits")]);
        editor.text.queue_edit(TextEdit::ImeCommit {
            value: "invalid".into(),
        });
        editor.text.apply_pending_edits(
            &mut editor.fonts,
            &mut editor.layout,
            &mut editor.clipboard,
            |character| character.is_ascii_digit(),
        );
        assert_eq!(editor.value(), "123");
        assert_eq!(editor.text.editor.raw_selection().text_range(), 0..3);
        assert!(!editor.text.history.can_undo());
    }

    #[test]
    fn paste_is_separate_from_typing() {
        let mut editor = Editor::new("abc");
        editor.apply([insert("x")]);
        // Inject the clipboard result rather than reading or writing the OS clipboard.
        editor.text.pending_paste = Some(ClipboardRead::Ready(Ok("paste".into())));
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "abcx");
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "abc");
    }

    #[test]
    fn pending_paste_is_recorded_before_queued_undo() {
        let mut editor = Editor::new("before");
        let result = Arc::new(Mutex::new(None));
        editor.text.pending_paste = Some(ClipboardRead::Pending(result.clone()));
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "before");
        assert_eq!(editor.text.pending_edits, [TextEdit::Undo]);
        *result.lock().unwrap() = Some(Ok("paste".into()));
        editor.apply([]);
        assert_eq!(editor.value(), "before");
        editor.apply([TextEdit::Redo]);
        assert_eq!(editor.value(), "beforepaste");
    }

    #[test]
    fn ime_updates_are_one_commit_and_cancellation_restores_selection() {
        let mut editor = Editor::new("old");
        editor.select(3, 0);
        editor.apply([preedit("n"), preedit("new"), TextEdit::Undo]);
        assert!(editor.text.is_composing());
        assert!(!editor.text.history.can_undo());
        editor.apply([TextEdit::ImeCommit {
            value: "new".into(),
        }]);
        assert_eq!(editor.value(), "new");
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "old");
        assert_eq!(editor.text.editor.raw_selection().anchor().index(), 3);
        editor.apply([preedit("cancel"), preedit("")]);
        assert_eq!(editor.value(), "old");
        assert!(editor.text.history.can_redo());
        // Winit may clear preedit immediately before delivering the commit.
        editor.apply([
            preedit("x"),
            preedit(""),
            TextEdit::ImeCommit { value: "x".into() },
        ]);
        assert_eq!(editor.value(), "x");
        editor.apply([TextEdit::Undo]);
        assert_eq!(editor.value(), "old");
    }

    #[test]
    fn ime_replacement_respects_limits_and_rejected_commit_keeps_original() {
        let mut editor = Editor::new("old");
        editor.text.max_characters = Some(3);
        editor.apply([
            TextEdit::SelectAll,
            preedit("new"),
            TextEdit::ImeCommit {
                value: "new".into(),
            },
        ]);
        assert_eq!(editor.value(), "new");
        editor.apply([
            TextEdit::Undo,
            preedit("longer"),
            TextEdit::ImeCommit {
                value: "longer".into(),
            },
        ]);
        assert_eq!(editor.value(), "old");
        assert!(editor.text.history.can_redo());
    }

    #[test]
    fn history_is_bounded_and_long_pauses_split_typing() {
        let mut editor = Editor::new("");
        editor.apply([insert("a")]);
        editor.text.history.group.as_mut().unwrap().1 = Instant::now() - Duration::from_secs(2);
        editor.apply([insert("b"), TextEdit::Undo]);
        assert_eq!(editor.value(), "a");
        editor.text.clear();
        for _ in 0..HISTORY_LIMIT + 5 {
            editor.apply([insert("\n")]);
        }
        assert_eq!(editor.text.history.undo.len(), HISTORY_LIMIT);
        for _ in 0..HISTORY_LIMIT + 5 {
            editor.apply([TextEdit::Undo]);
        }
        assert_eq!(editor.value(), "\n".repeat(5));
        assert_eq!(editor.text.history.redo.len(), HISTORY_LIMIT);
        for _ in 0..HISTORY_LIMIT {
            editor.apply([TextEdit::Redo]);
        }
        assert_eq!(editor.value(), "\n".repeat(HISTORY_LIMIT + 5));
        assert_eq!(editor.text.history.undo.len(), HISTORY_LIMIT);
    }
}
