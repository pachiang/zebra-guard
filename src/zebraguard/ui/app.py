"""QApplication 入口:主題、免責聲明、主視窗。"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from zebraguard.ui import theme
from zebraguard.ui.disclaimer_dialog import DisclaimerDialog, user_has_accepted_latest
from zebraguard.ui.main_window import MainWindow


def run() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("ZebraGuard")
    app.setOrganizationName("ZebraGuard")

    theme.apply(app)

    if not user_has_accepted_latest():
        dlg = DisclaimerDialog()
        if dlg.exec() != DisclaimerDialog.DialogCode.Accepted:
            return 0

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
