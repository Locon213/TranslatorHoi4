from __future__ import annotations

from pathlib import Path


UI_INTERFACES = Path("translatorhoi4/ui/ui_interfaces.py")


def test_batch_controls_are_wired_without_starting_qt():
    source = UI_INTERFACES.read_text(encoding="utf-8")

    assert "self.spn_chunk_size.setRange(1, 1000)" in source
    assert "self.lbl_batch_chunk_warning = BodyLabel" in source
    assert "self.lbl_batch_chunk_warning.setVisible(large_chunk)" in source
    assert "self.chk_batch_mode.stateChanged.connect(self._update_batch_controls)" in source
    assert "self.spn_chunk_size.valueChanged.connect(self._update_batch_controls)" in source
    assert "widget.setChecked(False)" in source
    assert "widget.setEnabled(False)" in source
    assert "return False if self.chk_batch_mode.isChecked() else widget.isChecked()" in source
