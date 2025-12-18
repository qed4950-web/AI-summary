import sys
import os

# Ensure project root is in sys.path so 'desktop_app' and 'core' imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import signal
from PySide6.QtWidgets import QApplication
from desktop_app.ui import LauncherWindow

def exception_hook(exctype, value, traceback):
    print(f"[CRITICAL_ERROR] {exctype.__name__}: {value}")
    # Show GUI error if possible
    try:
        from PySide6.QtWidgets import QMessageBox, QApplication
        if QApplication.instance():
            QMessageBox.critical(None, "Application Error", f"{exctype.__name__}:\n{value}")
    except:
        pass
    sys.__excepthook__(exctype, value, traceback)
    sys.exit(1)

sys.excepthook = exception_hook

def main():
    # Allow Ctrl+C to kill the app from terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Cycle 5: Keep app running in background
    
    try:
        import qdarktheme  # type: ignore

        if hasattr(qdarktheme, "setup_theme"):
            qdarktheme.setup_theme("dark")
        elif hasattr(qdarktheme, "load_stylesheet"):
            app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    except Exception:
        pass

    # Create and show the launcher
    window = LauncherWindow()
    window.show()

    # Cycle 5: System Tray Support
    from PySide6.QtWidgets import QSystemTrayIcon, QMenu
    from PySide6.QtGui import QIcon
    
    # Simple unicode icon placeholder if no file, or create a pixmap
    # For now, let's try to use a rigorous approach or standard icon
    tray_icon = QSystemTrayIcon(app)
    
    icon_path = os.path.join(current_dir, "assets", "logo.png")
    if os.path.exists(icon_path):
        tray_icon.setIcon(QIcon(icon_path))
    else:
        # Fallback to standard system icon to avoid "No Icon set" warning
        from PySide6.QtWidgets import QStyle
        tray_icon.setIcon(app.style().standardIcon(QStyle.SP_ComputerIcon))
    # If no icon is set, it might not show. Let's try to create a simple pixmap if needed or rely on window icon
    
    # Create Menu
    tray_menu = QMenu()
    action_show = tray_menu.addAction("열기")
    action_quit = tray_menu.addAction("종료")
    
    action_show.triggered.connect(window.show_and_activate)
    action_quit.triggered.connect(app.quit)
    
    tray_icon.setContextMenu(tray_menu)
    tray_icon.show()
    
    # Handle Tray Activation (Click)
    def tray_activated(reason):
        if reason == QSystemTrayIcon.Trigger:
            window.show_and_activate()
            
    tray_icon.activated.connect(tray_activated)

    # Hook app exit to close bridge process
    app.aboutToQuit.connect(window.cleanup)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
