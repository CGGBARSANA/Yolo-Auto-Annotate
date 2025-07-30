import sys
from PyQt5.QtWidgets import QApplication
from annotator import Annotator

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Annotator()
    window.show()
    sys.exit(app.exec_())