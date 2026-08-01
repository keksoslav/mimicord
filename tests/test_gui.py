from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_window_lists_personas(qapp, personas_home, persona_dir):
    from mimicord.gui import MainWindow

    window = MainWindow()
    try:
        assert window.persona_list.count() == 1
        assert window.persona_list.item(0).text() == "testbot"
        assert window.current_persona() == "testbot"
        # config tab picked up the selected persona's toml
        assert 'name = "testbot"' in window.config_editor.toPlainText()
        # status line reflects missing artifacts
        assert "no corpus" in window.status_label.text()
    finally:
        window.close()


def test_create_persona_scaffolds_and_selects(qapp, personas_home):
    from mimicord.gui import MainWindow

    window = MainWindow()
    try:
        window.create_persona("fresh")
        assert (personas_home / "fresh" / "persona.toml").is_file()
        assert (personas_home / "fresh" / "persona.md").is_file()
        assert window.current_persona() == "fresh"
    finally:
        window.close()


def test_save_config_rejects_broken_toml(qapp, personas_home, persona_dir, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from mimicord.gui import MainWindow

    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args: warnings.append(args[2])
    )
    window = MainWindow()
    try:
        original = (persona_dir / "persona.toml").read_text(encoding="utf-8")
        window.config_editor.setPlainText("name = [broken")
        window.save_config_tab()
        assert warnings and "TOML" in warnings[0]
        # file untouched
        assert (persona_dir / "persona.toml").read_text(encoding="utf-8") == original
    finally:
        window.close()


def test_save_config_writes_valid_toml(qapp, personas_home, persona_dir):
    from mimicord.gui import MainWindow

    window = MainWindow()
    try:
        text = window.config_editor.toPlainText().replace(
            'provider = "ollama"', 'provider = "deepseek"'
        )
        window.config_editor.setPlainText(text)
        window.save_config_tab()
        saved = (persona_dir / "persona.toml").read_text(encoding="utf-8")
        assert 'provider = "deepseek"' in saved
    finally:
        window.close()
