import sys
from PyQt5.QtWidgets import QApplication
from main.main_widget import Main

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())