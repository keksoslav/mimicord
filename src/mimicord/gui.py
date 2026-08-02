from __future__ import annotations

import asyncio
import logging
import sys
import tomllib
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mimicord.paths import PersonaPaths, personas_root

log = logging.getLogger(__name__)


class LogBridge(QObject):
    message = Signal(str)


class QtLogHandler(logging.Handler):
    """Routes mimicord log records into the gui thread via a signal."""

    def __init__(self, bridge: LogBridge) -> None:
        super().__init__(level=logging.INFO)
        self._bridge = bridge
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._bridge.message.emit(self.format(record))
        except Exception:
            pass


class PipelineWorker(QThread):
    """Runs one pipeline job off the gui thread, streaming progress lines."""

    line = Signal(str)
    done = Signal(bool, str)

    def __init__(self, job) -> None:
        super().__init__()
        self._job = job

    def run(self) -> None:
        try:
            result = self._job(self.line.emit)
            self.done.emit(True, result or "done")
        except Exception as error:
            self.done.emit(False, str(error))


class EngineLoader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def run(self) -> None:
        try:
            from mimicord.engine import PersonaEngine

            self.loaded.emit(PersonaEngine(self._name))
        except Exception as error:
            self.failed.emit(str(error))


class ChatWorker(QThread):
    replied = Signal(list)
    failed = Signal(str)

    def __init__(self, engine, context) -> None:
        super().__init__()
        self._engine = engine
        self._context = context

    def run(self) -> None:
        try:
            self.replied.emit(self._engine.reply(self._context))
        except Exception as error:
            self.failed.emit(str(error))


class BotWorker(QThread):
    """Runs the discord client on its own asyncio loop in this thread."""

    stopped = Signal(str)

    def __init__(self, name: str, dry_run: bool) -> None:
        super().__init__()
        self._name = name
        self._dry_run = dry_run
        self._client = None
        self._loop = None

    def run(self) -> None:
        try:
            from mimicord.bot import MimicClient
            from mimicord.engine import PersonaEngine

            logging.getLogger("mimicord").setLevel(logging.INFO)
            engine = PersonaEngine(self._name)
            token = engine.config.discord.token()
            self._client = MimicClient(engine, dry_run=self._dry_run)
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def runner():
                async with self._client:
                    await self._client.start(token)

            self._loop.run_until_complete(runner())
            self.stopped.emit("bot stopped")
        except Exception as error:
            self.stopped.emit(f"bot error: {error}")
        finally:
            if self._loop is not None:
                self._loop.close()

    def request_stop(self) -> None:
        if self._client is not None and self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("mimicord")
        self.resize(1020, 680)

        self._pipeline_worker: PipelineWorker | None = None
        self._chat_worker: ChatWorker | None = None
        self._engine_loader: EngineLoader | None = None
        self._bot_worker: BotWorker | None = None
        self._engine = None
        self._chat_context: list = []

        self._log_bridge = LogBridge()
        self._log_handler = QtLogHandler(self._log_bridge)
        mim_logger = logging.getLogger("mimicord")
        mim_logger.addHandler(self._log_handler)
        if mim_logger.getEffectiveLevel() > logging.INFO:
            mim_logger.setLevel(logging.INFO)
        self._log_bridge.message.connect(self._append_bot_log)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_tabs())
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.refresh_personas()

    # sidebar -------------------------------------------------------------

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("personas"))
        self.persona_list = QListWidget()
        self.persona_list.currentTextChanged.connect(self._persona_selected)
        layout.addWidget(self.persona_list)
        new_button = QPushButton("new persona")
        new_button.clicked.connect(self._new_persona_clicked)
        layout.addWidget(new_button)
        refresh = QPushButton("refresh")
        refresh.clicked.connect(self.refresh_personas)
        layout.addWidget(refresh)
        return panel

    def refresh_personas(self) -> None:
        current = self.current_persona()
        self.persona_list.clear()
        root = personas_root()
        if root.is_dir():
            for entry in sorted(root.iterdir()):
                if (entry / "persona.toml").is_file():
                    self.persona_list.addItem(entry.name)
        if current:
            matches = self.persona_list.findItems(current, Qt.MatchFlag.MatchExactly)
            if matches:
                self.persona_list.setCurrentItem(matches[0])
        elif self.persona_list.count():
            self.persona_list.setCurrentRow(0)

    def current_persona(self) -> str:
        item = self.persona_list.currentItem()
        return item.text() if item else ""

    def create_persona(self, name: str) -> None:
        from mimicord.scaffold import scaffold_persona

        try:
            scaffold_persona(name)
        except (FileExistsError, OSError) as error:
            QMessageBox.warning(self, "mimicord", str(error))
            return
        self.refresh_personas()
        matches = self.persona_list.findItems(name, Qt.MatchFlag.MatchExactly)
        if matches:
            self.persona_list.setCurrentItem(matches[0])

    def _new_persona_clicked(self) -> None:
        name, ok = QInputDialog.getText(self, "new persona", "name:")
        name = name.strip().lower()
        if ok and name:
            self.create_persona(name)

    def _persona_selected(self, name: str) -> None:
        self._engine = None
        self._chat_context.clear()
        self.chat_view.clear()
        self.load_config_tab()
        self.refresh_status()
        self.chat_status.setText("persona not loaded yet, hit load")

    # tabs ----------------------------------------------------------------

    def _build_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_config_tab(), "config")
        self.tabs.addTab(self._build_pipeline_tab(), "pipeline")
        self.tabs.addTab(self._build_chat_tab(), "chat")
        self.tabs.addTab(self._build_bot_tab(), "bot")
        return self.tabs

    # config tab ----------------------------------------------------------

    def _build_config_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.config_editor = QPlainTextEdit()
        self.config_editor.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(self.config_editor)
        row = QHBoxLayout()
        save = QPushButton("save")
        save.clicked.connect(self.save_config_tab)
        reload = QPushButton("reload")
        reload.clicked.connect(self.load_config_tab)
        row.addWidget(save)
        row.addWidget(reload)
        row.addStretch()
        layout.addLayout(row)
        return panel

    def load_config_tab(self) -> None:
        name = self.current_persona()
        if not name:
            self.config_editor.setPlainText("")
            return
        paths = PersonaPaths.for_persona(name)
        if paths.config.is_file():
            self.config_editor.setPlainText(
                paths.config.read_text(encoding="utf-8")
            )

    def save_config_tab(self) -> None:
        name = self.current_persona()
        if not name:
            return
        text = self.config_editor.toPlainText()
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError as error:
            QMessageBox.warning(self, "mimicord", f"not valid TOML: {error}")
            return
        paths = PersonaPaths.for_persona(name)
        paths.config.write_text(text, encoding="utf-8")
        try:
            from mimicord.config import load_config

            load_config(paths.config)
        except Exception as error:
            QMessageBox.warning(
                self, "mimicord", f"saved, but the config has a problem: {error}"
            )
            return
        self.statusBar().showMessage("config saved", 3000)
        self.refresh_status()

    # pipeline tab --------------------------------------------------------

    def _build_pipeline_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        grid = QGridLayout()
        self._pipeline_buttons: list[QPushButton] = []

        def button(row, col, label, slot):
            b = QPushButton(label)
            b.clicked.connect(slot)
            grid.addWidget(b, row, col)
            self._pipeline_buttons.append(b)
            return b

        button(0, 0, "ingest DCE exports...", self._ingest_dce_clicked)
        button(0, 1, "ingest data package...", self._ingest_package_clicked)
        button(1, 0, "compute stats", self._stats_clicked)
        button(1, 1, "analyze (LLM)", self._analyze_clicked)
        button(2, 0, "compile persona (LLM)", self._compile_clicked)
        button(2, 1, "build memory index", self._index_clicked)
        layout.addLayout(grid)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.pipeline_log = QPlainTextEdit()
        self.pipeline_log.setReadOnly(True)
        self.pipeline_log.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(self.pipeline_log)
        return panel

    def refresh_status(self) -> None:
        name = self.current_persona()
        if not name:
            self.status_label.setText("")
            return
        paths = PersonaPaths.for_persona(name)
        parts = []
        if paths.corpus.is_file():
            from mimicord.store import Store

            with Store(paths.corpus) as store:
                counts = store.counts()
            parts.append(
                f"corpus {counts['total']} msgs ({counts['target']} target)"
            )
        else:
            parts.append("no corpus")
        parts.append("stats ok" if paths.stats.is_file() else "no stats")
        parts.append("profile ok" if paths.profile.is_file() else "no profile")
        parts.append("persona.md ok" if paths.persona_md.is_file() else "no persona.md")
        parts.append("examples ok" if paths.examples.is_file() else "no examples")
        parts.append("memories ok" if paths.chroma_dir.is_dir() else "no memories")
        self.status_label.setText("  |  ".join(parts))

    def _run_pipeline(self, label: str, job) -> None:
        if self._pipeline_worker is not None and self._pipeline_worker.isRunning():
            QMessageBox.information(self, "mimicord", "a pipeline step is already running")
            return
        for b in self._pipeline_buttons:
            b.setEnabled(False)
        self.pipeline_log.appendPlainText(f"== {label} ==")
        worker = PipelineWorker(job)
        worker.line.connect(self.pipeline_log.appendPlainText)
        worker.done.connect(self._pipeline_done)
        self._pipeline_worker = worker
        worker.start()

    def _pipeline_done(self, ok: bool, message: str) -> None:
        self.pipeline_log.appendPlainText(message if ok else f"error: {message}")
        self.pipeline_log.appendPlainText("")
        for b in self._pipeline_buttons:
            b.setEnabled(True)
        self.refresh_status()

    def _ingest_dce_clicked(self) -> None:
        name = self.current_persona()
        if not name:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "DiscordChatExporter exports", "", "JSON files (*.json)"
        )
        if not files:
            return

        def job(emit):
            from mimicord.config import load_config
            from mimicord.ingest import ingest_dce
            from mimicord.store import Store

            paths = PersonaPaths.for_persona(name)
            cfg = load_config(paths.config)
            with Store(paths.corpus) as store:
                parsed = ingest_dce(store, [Path(f) for f in files], cfg.target)
                counts = store.counts()
            return (
                f"parsed {parsed} messages; corpus now {counts['total']} total, "
                f"{counts['target']} from target"
            )

        self._run_pipeline("ingest dce", job)

    def _ingest_package_clicked(self) -> None:
        name = self.current_persona()
        if not name:
            return
        directory = QFileDialog.getExistingDirectory(self, "data package folder")
        if not directory:
            return

        def job(emit):
            from mimicord.config import load_config
            from mimicord.ingest import ingest_package
            from mimicord.store import Store

            paths = PersonaPaths.for_persona(name)
            cfg = load_config(paths.config)
            with Store(paths.corpus) as store:
                parsed = ingest_package(store, Path(directory), cfg.target)
                counts = store.counts()
            return (
                f"parsed {parsed} messages; corpus now {counts['total']} total, "
                f"{counts['target']} from target"
            )

        self._run_pipeline("ingest package", job)

    def _stats_clicked(self) -> None:
        name = self.current_persona()
        if not name:
            return

        def job(emit):
            import json

            from mimicord.analyze import stats as stats_mod
            from mimicord.store import Store

            paths = PersonaPaths.for_persona(name)
            with Store(paths.corpus) as store:
                result = stats_mod.compute(store)
            paths.stats.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            for line in stats_mod.summary_lines(result):
                emit(line)
            return f"wrote {paths.stats.name}"

        self._run_pipeline("stats", job)

    def _analyze_clicked(self) -> None:
        name = self.current_persona()
        if not name:
            return

        def job(emit):
            import json

            from mimicord.analyze.chunker import build_chunks, sample_chunks
            from mimicord.analyze.mapper import analyze_chunks
            from mimicord.analyze.reducer import reduce_profiles
            from mimicord.config import load_config
            from mimicord.llm.factory import get_provider
            from mimicord.store import Store

            paths = PersonaPaths.for_persona(name)
            cfg = load_config(paths.config)
            with Store(paths.corpus) as store:
                chunks = sample_chunks(build_chunks(store))
                if not chunks:
                    return "no target messages to analyze"
                emit(f"{len(chunks)} chunks to analyze")
                provider = get_provider(cfg.llm, role="map")
                position = 0

                def progress(chunk, cached):
                    nonlocal position
                    position += 1
                    emit(
                        f"chunk {position}/{len(chunks)} "
                        f"({chunk.channel_name or chunk.channel_id}) "
                        + ("cached" if cached else "done")
                    )

                results = analyze_chunks(
                    chunks, provider, cfg.name, paths.chunks_dir, progress=progress
                )
            emit("merging into one profile...")
            profile = reduce_profiles(
                results, get_provider(cfg.llm, role="reduce"), cfg.name
            )
            paths.profile.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return f"wrote {paths.profile.name}"

        self._run_pipeline("analyze", job)

    def _compile_clicked(self) -> None:
        name = self.current_persona()
        if not name:
            return

        def job(emit):
            import json

            from mimicord.compile.examples import build_examples
            from mimicord.compile.persona import compile_persona
            from mimicord.config import load_config
            from mimicord.llm.factory import get_provider
            from mimicord.store import Store

            paths = PersonaPaths.for_persona(name)
            cfg = load_config(paths.config)
            profile = json.loads(paths.profile.read_text(encoding="utf-8"))
            stats_data = json.loads(paths.stats.read_text(encoding="utf-8"))
            provider = get_provider(cfg.llm, role="reduce")
            emit("writing persona.md...")
            paths.persona_md.write_text(
                compile_persona(profile, stats_data, provider, cfg.name),
                encoding="utf-8",
            )
            emit("curating few-shot examples...")
            with Store(paths.corpus) as store:
                data = build_examples(store, stats_data, provider, cfg.name)
            paths.examples.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return f"wrote persona.md and {len(data['examples'])} examples"

        self._run_pipeline("compile", job)

    def _index_clicked(self) -> None:
        name = self.current_persona()
        if not name:
            return

        def job(emit):
            from mimicord.config import load_config
            from mimicord.rag import build_index
            from mimicord.store import Store

            paths = PersonaPaths.for_persona(name)
            cfg = load_config(paths.config)
            emit("indexing (first run downloads a small embedding model)...")
            with Store(paths.corpus) as store:
                total = build_index(
                    paths,
                    cfg.rag,
                    store,
                    progress=lambda done, all_: emit(f"{done}/{all_} windows"),
                )
            return f"indexed {total} conversation windows"

        self._run_pipeline("index", job)

    # chat tab ------------------------------------------------------------

    def _build_chat_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        row = QHBoxLayout()
        self.chat_load = QPushButton("load persona")
        self.chat_load.clicked.connect(self._load_engine_clicked)
        self.chat_status = QLabel("persona not loaded yet, hit load")
        row.addWidget(self.chat_load)
        row.addWidget(self.chat_status)
        row.addStretch()
        clear = QPushButton("clear context")
        clear.clicked.connect(self._clear_chat)
        row.addWidget(clear)
        layout.addLayout(row)

        self.chat_view = QPlainTextEdit()
        self.chat_view.setReadOnly(True)
        layout.addWidget(self.chat_view)

        send_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self._send_clicked)
        self.chat_send = QPushButton("send")
        self.chat_send.clicked.connect(self._send_clicked)
        send_row.addWidget(self.chat_input)
        send_row.addWidget(self.chat_send)
        layout.addLayout(send_row)
        return panel

    def _clear_chat(self) -> None:
        self._chat_context.clear()
        self.chat_view.clear()

    def _load_engine_clicked(self) -> None:
        name = self.current_persona()
        if not name:
            return
        self.chat_load.setEnabled(False)
        self.chat_status.setText("loading...")
        loader = EngineLoader(name)
        loader.loaded.connect(self._engine_loaded)
        loader.failed.connect(self._engine_failed)
        self._engine_loader = loader
        loader.start()

    def _engine_loaded(self, engine) -> None:
        self._engine = engine
        llm = engine.config.llm
        memory = "memories on" if engine.rag is not None else "memories off"
        self.chat_status.setText(f"{llm.provider}/{llm.model}, {memory}")
        self.chat_load.setEnabled(True)

    def _engine_failed(self, message: str) -> None:
        self.chat_status.setText("load failed")
        self.chat_load.setEnabled(True)
        QMessageBox.warning(self, "mimicord", message)

    def _send_clicked(self) -> None:
        if self._engine is None:
            QMessageBox.information(self, "mimicord", "load the persona first")
            return
        if self._chat_worker is not None and self._chat_worker.isRunning():
            return
        text = self.chat_input.text().strip()
        if not text:
            return
        from mimicord.engine import ContextMessage

        self.chat_input.clear()
        self._chat_context.append(ContextMessage("you", text))
        self.chat_view.appendPlainText(f"you: {text}")
        window = self._engine.config.discord.context_messages
        worker = ChatWorker(self._engine, self._chat_context[-window:])
        worker.replied.connect(self._chat_replied)
        worker.failed.connect(self._chat_failed)
        self._chat_worker = worker
        self.chat_send.setEnabled(False)
        worker.start()

    def _chat_replied(self, bursts: list) -> None:
        from mimicord.engine import ContextMessage

        persona = self._engine.config.name
        for burst in bursts:
            self.chat_view.appendPlainText(f"{persona}: {burst}")
            self._chat_context.append(ContextMessage(persona, burst))
        self.chat_send.setEnabled(True)

    def _chat_failed(self, message: str) -> None:
        self.chat_view.appendPlainText(f"(error: {message})")
        self.chat_send.setEnabled(True)

    # bot tab -------------------------------------------------------------

    def _build_bot_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        row = QHBoxLayout()
        self.bot_dry_run = QCheckBox("dry run (log replies, send nothing)")
        self.bot_dry_run.setChecked(True)
        self.bot_start = QPushButton("start bot")
        self.bot_start.clicked.connect(self._start_bot_clicked)
        self.bot_stop = QPushButton("stop bot")
        self.bot_stop.clicked.connect(self._stop_bot_clicked)
        self.bot_stop.setEnabled(False)
        self.bot_status = QLabel("not running")
        row.addWidget(self.bot_dry_run)
        row.addWidget(self.bot_start)
        row.addWidget(self.bot_stop)
        row.addWidget(self.bot_status)
        row.addStretch()
        layout.addLayout(row)
        self.bot_log = QPlainTextEdit()
        self.bot_log.setReadOnly(True)
        self.bot_log.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(self.bot_log)
        return panel

    def _append_bot_log(self, line: str) -> None:
        self.bot_log.appendPlainText(line)

    def _start_bot_clicked(self) -> None:
        name = self.current_persona()
        if not name:
            return
        if self._bot_worker is not None and self._bot_worker.isRunning():
            return
        worker = BotWorker(name, self.bot_dry_run.isChecked())
        worker.stopped.connect(self._bot_stopped)
        self._bot_worker = worker
        self.bot_start.setEnabled(False)
        self.bot_dry_run.setEnabled(False)
        self.bot_stop.setEnabled(True)
        mode = "dry run" if self.bot_dry_run.isChecked() else "LIVE"
        self.bot_status.setText(f"running ({mode})")
        self.bot_log.appendPlainText(f"== starting {name} ({mode}) ==")
        worker.start()

    def _stop_bot_clicked(self) -> None:
        if self._bot_worker is not None:
            self.bot_status.setText("stopping...")
            self._bot_worker.request_stop()

    def _bot_stopped(self, message: str) -> None:
        self.bot_log.appendPlainText(message)
        self.bot_status.setText("not running")
        self.bot_start.setEnabled(True)
        self.bot_dry_run.setEnabled(True)
        self.bot_stop.setEnabled(False)

    def closeEvent(self, event) -> None:
        if self._bot_worker is not None and self._bot_worker.isRunning():
            self._bot_worker.request_stop()
            self._bot_worker.wait(5000)
        logging.getLogger("mimicord").removeHandler(self._log_handler)
        super().closeEvent(event)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.WARNING)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
