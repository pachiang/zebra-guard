"""Smoke: 建構 MainWindow 看會不會炸。"""

from __future__ import annotations

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, "src")

from PySide6.QtWidgets import QApplication  # noqa: E402

from zebraguard.ui import theme  # noqa: E402
from zebraguard.ui.main_window import MainWindow  # noqa: E402

app = QApplication([])
theme.apply(app)
w = MainWindow()
w.show()
print("MainWindow built OK")
print("Views:", [type(w.stack.widget(i)).__name__ for i in range(w.stack.count())])
# 乾淨關閉(避免 DecoderThread 還活著時 Qt app destruct)
w.close()
app.processEvents()
sys.exit(0)
