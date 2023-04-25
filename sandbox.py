import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication, QComboBox, QVBoxLayout, QWidget

# Generate an array of 1000 signals with 1000 samples each
signals = np.random.rand(1000, 1000)

def plot_signal(signal):
    plt.clf()
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

class SignalPlotter(QWidget):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.signal_dropdown = QComboBox()
        self.signal_dropdown.addItems(['Signal {}'.format(i) for i in range(len(self.signals))])
        self.signal_dropdown.currentIndexChanged.connect(self.plot_selected_signal)
        layout.addWidget(self.signal_dropdown)

        self.plot = plt.figure()

        self.plot_widget = plt.show(block=False)
        layout.addWidget(self.plot_widget)

    def plot_selected_signal(self):
        selected_signal = self.signal_dropdown.currentIndex()
        plot_signal(self.signals[selected_signal])

app = QApplication([])
window = SignalPlotter(signals)
window.show()
app.exec_()
