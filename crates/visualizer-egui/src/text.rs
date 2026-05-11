use std::hash::Hash;

pub fn wrapped_label(ui: &mut egui::Ui, text: &str) {
    let width = ui.available_width().max(120.0);
    let display = hard_wrap_long_segments(text, 96);
    ui.scope(|ui| {
        ui.set_max_width(width);
        ui.add(egui::Label::new(display).wrap());
    });
}

pub fn hard_wrap_long_segments(text: &str, limit: usize) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = 0_usize;
    for ch in text.chars() {
        if ch == '\n' {
            run = 0;
            out.push(ch);
            continue;
        }

        if ch.is_whitespace() {
            run = 0;
            out.push(ch);
            continue;
        }

        if run >= limit {
            out.push('\n');
            run = 0;
        }
        out.push(ch);
        run += 1;
    }
    out
}

pub fn scroll_text_rows(
    ui: &mut egui::Ui,
    id_salt: impl Hash,
    rows: impl IntoIterator<Item = impl AsRef<str>>,
) {
    egui::ScrollArea::vertical()
        .id_salt(id_salt)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for row in rows {
                wrapped_label(ui, row.as_ref());
                ui.add_space(4.0);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_wrap_long_segments_breaks_unspaced_content() {
        let wrapped = hard_wrap_long_segments("abcdefghijkl", 4);
        assert_eq!(wrapped, "abcd\nefgh\nijkl");
    }
}
