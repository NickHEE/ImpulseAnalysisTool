"""
Impulse Analysis Tool - Nicholas Huttemann - 2019-08-30

Created during a summer co-op with BCIT Applied Research and SoundQA


"""
import os, sys, queue, tempfile, shutil, multiprocessing, time, itertools, json, concurrent.futures, sqlite3, hashlib
import datetime

from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QSplitter, QFrame, QSlider,
                             QWidget, QTabWidget, QApplication, QFileSystemModel, QTreeView, QPushButton, QStackedWidget,
                             QSizePolicy, QFileDialog, QTableWidget, QTableWidgetItem, QAbstractItemView, QScrollArea,
                             QMessageBox, QLabel, QButtonGroup, QLineEdit, QComboBox, QProgressDialog, QShortcut,
                             QGridLayout)
from PyQt5.QtCore import (QDir, pyqtSignal, QObject, pyqtSlot, Qt, QThread, QSize)
from PyQt5.QtGui import QColor, QKeySequence, QIcon

import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

import numpy as np, pandas as pd
from scipy.optimize import curve_fit
import sounddevice, soundfile

import Utils


class ImpulseAnalysisTool(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Impulse Analysis Tool')
        self.width = 1280
        self.height = 720
        self.setGeometry(0, 0, self.width, self.height)

        self.mainWidget = MainView(self)
        self.setCentralWidget(self.mainWidget)
        self.show()

class MainView(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.recordTab = RecordTab()
        self.analyzeTab = AnalyzeTab()
        self.tabs.addTab(self.recordTab, "Record")
        self.tabs.addTab(self.analyzeTab, "Analyze")
        self.tabs.currentChanged.connect(self.toggle_audio_stream)
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def toggle_audio_stream(self, tab):
        """
        Stop the recording stream when the recording tab is not in view
        """
        if tab == 0:
            if not self.recordTab.stream.inputStream.active:
                self.recordTab.stream.inputStream.start()
        if tab == 1:
            self.recordTab.stream.inputStream.stop()


class RecordTab(QWidget):

    def __init__(self):
        super().__init__()

        self.processed_fields = {}
        self.state = 'Waiting'

        # Setup Layouts
        self.mainLayout = QVBoxLayout()
        self.hBox = QHBoxLayout()
        self.outputGrid = QGridLayout()
        self.lGrpBox = QGroupBox('Output Options')
        self.lGrpBox.setLayout(self.outputGrid)
        self.rGrpBox = QGroupBox('Input Device')
        self.rVBox = QVBoxLayout()
        self.rGrpBox.setLayout(self.rVBox)

        # Center
        self.recordBtn = QPushButton()

        # Microphone Stream and live microphone canvas
        self.tempFile = None
        self.stream = MicrophoneStream()
        self.stream.recordingDone.connect(self.on_recording_done)

        self.canvasStack = QStackedWidget()
        self.microphoneCanvas = MicrophoneCanvas(self.stream)
        self.recordBtn.clicked.connect(self.stream.toggle_recording)
        self.recordBtn.clicked.connect(self.on_recording_toggle)
        self.plotCanvas = PlotCanvas(self)
        self.canvasStack.addWidget(self.microphoneCanvas)
        self.canvasStack.addWidget(self.plotCanvas)
        self.canvasStack.setCurrentWidget(self.microphoneCanvas)
        self.mainLayout.addWidget(self.canvasStack, 5)

        # Output Options (LEFT)
        self.dbLabel = QLabel('Database:')
        self.dbPath = QLineEdit(os.path.dirname(__file__))
        self.dbBrowse = QPushButton('Browse')
        self.dbBrowse.clicked.connect(self.on_browse_db)
        self.dbChk = QCheckBox()
        self.dbChk.clicked.connect(self.handle_chk_state)
        Utils.addWidgets_to_grid(self.outputGrid,
            [(self.dbLabel,1,1,1,1), (self.dbPath, 1,2,1,1), (self.dbBrowse, 1,3,1,1), (self.dbChk, 1,4,1,1)])

        self.savePathLabel = QLabel('Directory:')
        self.savePath = QLineEdit(os.path.dirname(__file__))
        self.savePathBrowse = QPushButton('Browse')
        self.savePathBrowse.clicked.connect(self.on_browse_local_dir)
        self.savePathChk = QCheckBox()
        self.savePathChk.setChecked(True)
        self.savePathChk.clicked.connect(self.handle_chk_state)
        Utils.addWidgets_to_grid(self.outputGrid,
            [(self.savePathLabel, 2,1,1,1), (self.savePath, 2,2,1,1), (self.savePathBrowse, 2,3,1,1),(self.savePathChk,2,4,1,1)])

        self.saveNameLabel = QLabel('Name:')
        self.saveName = QLineEdit()
        Utils.addWidgets_to_grid(self.outputGrid, [(self.saveNameLabel, 3,1,1,1), (self.saveName, 3,2,1,2)])

        self.deleteBtn = QPushButton('Cancel')
        self.deleteBtn.setEnabled(False)
        self.deleteBtn.clicked.connect(self.on_press_delete)
        self.saveBtn = QPushButton('Save')
        self.saveBtn.setEnabled(False)
        self.saveBtn.clicked.connect(self.on_press_save)
        Utils.addWidgets_to_grid(self.outputGrid, [(self.saveBtn, 4,1,1,2), (self.deleteBtn, 4,3,1,2)])
        self.outputGrid.setColumnMinimumWidth(4, 20)

        # Sound Device Controls (RIGHT)
        self.inputDropDown = QComboBox()
        self.inputDevices = [device['name'] for device in sounddevice.query_devices() if device['max_input_channels'] > 0]
        self.inputDropDown.addItems(self.inputDevices)
        activeDevice = sounddevice.query_devices(device=self.stream.inputStream.device)
        self.inputDropDown.setCurrentIndex(self.inputDevices.index(activeDevice['name']))
        self.inputDropDown.currentIndexChanged.connect(self.on_input_changed)

        # Add widgets to layouts
        self.mainLayout.addLayout(self.hBox, 1)
        self.hBox.addWidget(self.lGrpBox, 5)
        self.hBox.addStretch(2)
        self.hBox.addWidget(self.recordBtn, 3)
        self.hBox.addStretch(1)
        self.hBox.addWidget(self.rGrpBox, 1)
        Utils.setAlignments(self.hBox, [(self.recordBtn, Qt.AlignCenter), (self.rGrpBox, Qt.AlignRight)])
        self.rVBox.addWidget(self.inputDropDown)
        self.setLayout(self.mainLayout)

        # Style
        self.recordBtn.setIcon(QIcon(r'.\assets\record.png'))
        self.recordBtn.setIconSize(QSize(100,100))
        self.recordBtn.setStyleSheet('QPushButton{border: 1px solid;}')

        # Shortcuts
        self.recordShortCut = QShortcut(QKeySequence('Space'), self)
        self.recordShortCut.activated.connect(self.on_recording_toggle)
        self.recordShortCut.activated.connect(self.stream.toggle_recording)

        self.handle_chk_state()

    def handle_chk_state(self):
        self.dbBrowse.setEnabled(self.dbChk.isChecked())
        self.dbPath.setEnabled(self.dbChk.isChecked())
        self.savePathBrowse.setEnabled(self.savePathChk.isChecked())
        self.savePath.setEnabled(self.savePathChk.isChecked())
        self.saveName.setEnabled(self.savePathChk.isChecked())

        if self.state == 'saving':
            if not self.dbChk.isChecked() and not self.savePathChk.isChecked():
                self.saveBtn.setEnabled(False)
            else:
                self.saveBtn.setEnabled(True)

    def on_browse_local_dir(self):
        options = QFileDialog.Options()
        path = QFileDialog.getExistingDirectory(self, caption="Choose save directory", options=options)
        self.savePath.setText(path)

    def on_browse_db(self):
        options = QFileDialog.Options()
        path, _ = QFileDialog.getOpenFileName(self, "Open .db File", "",
                                                    "Sqlite3 DB File (*.db)", options=options)
        self.dbPath.setText(path)

    @pyqtSlot(int)
    def on_input_changed(self, index):
        deviceName = self.inputDevices[index]
        for i, dev in enumerate(sounddevice.query_devices()):
            if dev['name'] == deviceName:
                device = i
                break
        else:
            device = sounddevice.default.device[0]
            print('Failed to set input device')

        self.stream.inputStream.stop()
        self.stream.inputStream = sounddevice.InputStream(samplerate=self.stream.inputStream.samplerate,
                                                          device=device,
                                                          channels=1,
                                                          callback=self.stream.audio_callback)
        self.stream.inputStream.start()

    @pyqtSlot()
    def on_recording_toggle(self):
        if self.state == 'Waiting':
            self.on_recording_start()

    @pyqtSlot()
    def on_recording_start(self):
        self.inputDropDown.setEnabled(False)
        self.saveBtn.setEnabled(False)
        self.deleteBtn.setEnabled(False)
        self.recordBtn.setIcon(QIcon(r'.\assets\stopWithSquare.png'))
        self.state = 'recording'


    @pyqtSlot(tuple)
    def on_recording_done(self, tempFile):
        """ Triggered when the microphone stream has finished recording. Enables / Disables
        appropriate buttons and receives tempFile from the microphone stream. Also processes an FFT for display

        :param tempFile: (file handle: int, file path: str) temporary audio file generated by
        the microphone stream to store the audio while the user decides whether or not to
        save it"""

        self.state = 'saving'
        self.handle_chk_state()
        self.deleteBtn.setEnabled(True)
        self.recordBtn.setEnabled(False)
        self.recordShortCut.setEnabled(False)
        self.recordBtn.setIcon(QIcon(r'.\assets\record.png'))
        self.inputDropDown.setEnabled(True)
        self.tempFile = tempFile

        # Process FFT
        tChopped, vChopped, fVals, \
        powerFFT, peakFreqs, peakAmps = Utils.AnalyzeFFT(tempFile[1], tChop=None)

        # Get these fields ready for a possible insertion into DB
        self.processed_fields['PCM'] = str(list(vChopped))
        self.processed_fields['Date'] = str(datetime.datetime.now())
        self.processed_fields['FFT_Processed'] = str(list(powerFFT))
        self.processed_fields['Sample_Rate'] = str(self.stream.sampleRate)
        self.processed_fields['Hash'] = hashlib.sha256(str(list(powerFFT)).encode('utf-8')).hexdigest()
        self.processed_fields['Peaks_Processed'] = []
        for freq, amp in zip(peakFreqs, peakAmps):
            self.processed_fields['Peaks_Processed'].append({"Frequency": freq, "Amplitude": amp})
        self.processed_fields['Peaks_Processed'].sort(reverse=True, key=lambda peak: peak['Amplitude'])
        self.processed_fields['Peaks_Processed'] = str(self.processed_fields['Peaks_Processed'])

        os.close(self.tempFile[0])
        os.remove(self.tempFile[1])

        # Make a new .wav file from the processed data
        self.tempFile = tempfile.mkstemp(prefix='temp_processed_',
                                         suffix='.wav',
                                         dir='')
        fileStream = soundfile.SoundFile(self.tempFile[1],
                                         mode='w',
                                         samplerate=self.stream.sampleRate,
                                         channels=max(self.stream.channels),
                                         subtype=self.stream.subtype)
        fileStream.write(vChopped)
        fileStream.close()

        self.plotCanvas.plot(tChopped, vChopped, fVals, powerFFT, peakFreqs, peakAmps, 'Impulse Recording')
        self.canvasStack.setCurrentWidget(self.plotCanvas)


    @pyqtSlot()
    def on_press_save(self):
        """ Triggered when the save button is pressed. Opens a save file dialog to permanently
        save the temporary audio file. Removes temporary file after."""

        if self.dbChk.isChecked():
            self.processed_fields['db'] = self.dbPath.text()
            self.dbForm = DBFormWindow(self.processed_fields, self)
            self.dbForm.show()

        if self.savePathChk.isChecked():
            if self.savePath.text():
                shutil.copy(self.tempFile[1], os.path.join(self.savePath.text(), self.saveName.text()+'.wav'))
                os.close(self.tempFile[0])
                os.remove(self.tempFile[1])

                QMessageBox.information(self, 'Saved', f'Saved to: {os.path.join(self.savePath.text(), self.saveName.text()+".wav")}')

        self.saveBtn.setEnabled(False)
        self.deleteBtn.setEnabled(False)
        self.recordBtn.setEnabled(True)
        self.recordBtn.setIcon(QIcon(r'.\assets\record.png'))
        self.recordShortCut.setEnabled(True)
        self.inputDropDown.setEnabled(True)
        self.canvasStack.setCurrentWidget(self.microphoneCanvas)
        self.state = 'Waiting'


    @pyqtSlot()
    def on_press_delete(self):
        """ Triggered when the delete button is pressed. Removes temporary audio file"""

        os.close(self.tempFile[0])
        os.remove(self.tempFile[1])
        self.recordBtn.setEnabled(True)
        self.recordBtn.setIcon(QIcon(r'.\assets\record.png'))
        self.deleteBtn.setEnabled(False)
        self.saveBtn.setEnabled(False)
        self.inputDropDown.setEnabled(True)
        self.canvasStack.setCurrentWidget(self.microphoneCanvas)
        self.state = 'Waiting'


class DBFormWindow(QMainWindow):

    def __init__(self, fields, parent=None):
        super(DBFormWindow, self).__init__(parent)

        self.width = 680
        self.height = 720
        self.setGeometry(0, 0, self.width, self.height)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.DBForm = DBForm(fields, self)
        self.scrollArea.setWidget(self.DBForm)
        self.setCentralWidget(self.scrollArea)


class DBForm(QWidget):

    def __init__(self, fields, parent=None):
        super(DBForm, self).__init__(parent)

        self.parent = parent
        self.shapeLineEdits = {}
        self.recordLineEdits = {}

        self.fields = fields
        self.layout = QVBoxLayout()

        self.header = QVBoxLayout()
        self.header.setAlignment(Qt.AlignVCenter)
        self.connectedTo = QLabel(f'Connected to: {fields["db"]}')
        self.shapeDropDown = QComboBox()
        self.shapeDropDown.addItems(self.get_shapes())
        self.shapeDropDown.currentTextChanged.connect(lambda _: self.create_fields(self.shapeGrid, self.get_shape_fields(),
                                                                                   fType='shape'))
        self.header.addWidget(self.connectedTo)
        self.header.addWidget(self.shapeDropDown)

        self.shapeGrid = QGridLayout()
        self.shapeGroup = QGroupBox('Shape Fields')
        self.shapeGroup.setLayout(self.shapeGrid)

        self.recordGrid = QGridLayout()
        self.recordGroup = QGroupBox('Record Fields')
        self.recordGroup.setLayout(self.recordGrid)

        self.footer = QHBoxLayout()
        self.submitBtn = QPushButton('Submit')
        self.submitBtn.clicked.connect(self.submit)
        self.cancelBtn = QPushButton('Cancel')
        self.cancelBtn.clicked.connect(self.parent.close)
        self.footer.addWidget(self.submitBtn)
        self.footer.addWidget(self.cancelBtn)

        self.layout.addLayout(self.header, 1)
        self.layout.addWidget(self.shapeGroup, 5)
        self.layout.addWidget(self.recordGroup, 5)
        self.layout.addLayout(self.footer, 1)
        self.setLayout(self.layout)

        self.create_fields(self.shapeGrid, self.get_shape_fields(), fType='shape')
        self.create_fields(self.recordGrid, self.get_record_fields(), fType='record')

    def submit(self):
        with sqlite3.connect(self.fields['db']) as conn:
            cur = conn.cursor()
            shape = self.shapeDropDown.currentText()

            record_fields = {k: v.text() for k, v in self.recordLineEdits.items()}
            shape_fields = {k: v.text() for k, v in self.shapeLineEdits.items()}
            format_fields = lambda l: ', '.join(l)
            format_d_fields = lambda l: ':' + ',:'.join(l)

            try:
                cur.execute(f'SELECT {shape}_ID FROM {shape} WHERE Sample_Name = "{shape_fields["Sample_Name"]}"')

                block_ID = cur.fetchone()

                # If the block already exists
                if block_ID:
                    print(f'SELECT Shape_ID FROM Shapes WHERE {shape}_ID = {block_ID[0]}')
                    cur.execute(f'SELECT Shape_ID FROM Shapes WHERE {shape}_ID = {block_ID[0]}')
                    shape_id = cur.fetchone()[0]
                    record_fields['Shape_ID'] = shape_id
                    print(f'Block exists: {shape_fields["Sample_Name"]}')

                # Create new block
                else:
                    cur.execute(f'INSERT INTO {shape} ({format_fields(shape_fields)}) '.replace('Set', '"Set"') +
                                f'VALUES ({format_d_fields(shape_fields)})', shape_fields)

                    cur.execute(f'SELECT Shape_ID FROM Shapes WHERE '
                                f'{shape}_ID = (SELECT {shape}_ID FROM {shape} '
                                f'WHERE Sample_Name = "{shape_fields["Sample_Name"]}")')
                    shape_id = cur.fetchone()[0]
                    record_fields['Shape_ID'] = shape_id
                print(f'INSERT INTO Records ({format_fields(record_fields.keys())})'
                      f'VALUES ({format_d_fields(record_fields.keys())})')
                cur.execute(f'INSERT INTO Records ({format_fields(record_fields.keys())})'
                            f'VALUES ({format_d_fields(record_fields.keys())})', record_fields)


                QMessageBox.information(self, 'Success!','Record successfully inserted!')
                self.parent.close()

            except Exception as e:
                QMessageBox.information(self, 'Error', f'Failed to insert record: {e}')

    def get_shapes(self):
        with sqlite3.connect(self.fields['db']) as conn:
            cur = conn.cursor()
            cur.execute('SELECT Name FROM Shape_Types')
            return [result[0] for result in cur.fetchall()]

    def get_shape_fields(self):
        with sqlite3.connect(self.fields['db']) as conn:
            cur = conn.cursor()
            cur.execute(f'SELECT * FROM {self.shapeDropDown.currentText()} LIMIT 1')
            return [field[0] for field in cur.description[1:]]

    def get_record_fields(self):
        with sqlite3.connect(self.fields['db']) as conn:
            cur = conn.cursor()
            cur.execute(f'SELECT * FROM Records LIMIT 1')
            return [field[0] for field in cur.description[1:]]

    def create_fields(self, grid, fields, fType, cols=2):
        # Remove existing fields
        for i in reversed(range(grid.count())):
            try:
                layout = grid.itemAt(i)
                for j in reversed(range(layout.count())):
                    layout.itemAt(j).widget().setParent(None)
                grid.removeItem(layout)
            except:
                print(grid.itemAt(i))
        self.shapeLineEdits = {}

        row = 1
        col = 1
        for field in fields:
            hbox = QHBoxLayout()
            text = QLabel(field)
            text.setFixedWidth(200)
            lineEdit = QLineEdit()
            lineEdit.setMaxLength(1_500_000)
            lineEdit.setMaximumWidth(400)
            hbox.addWidget(text,1)
            hbox.setAlignment(text, Qt.AlignLeft)
            hbox.addWidget(lineEdit, 4)
            hbox.setAlignment(lineEdit, Qt.AlignCenter)
            if fType == 'shape':
                self.shapeLineEdits[field] = lineEdit
            else:
                self.recordLineEdits[field] = lineEdit

            grid.addLayout(hbox, row, col)
            col += 1
            if col > cols:
                col = 1
                row += 1

        for field, lineEdit in {**self.shapeLineEdits, **self.recordLineEdits}.items():
            if field in self.fields.keys():
                lineEdit.setText(self.fields[field])
            if field in ['Shape_ID', 'PCM', 'Hash', 'FFT_Processed', 'Peaks_Processed', 'Strength_StrikeIt',
                                'Strength_Processed', 'Date', 'Sample_Rate']:
                lineEdit.setEnabled(False)


class MicrophoneCanvas(FigureCanvasQTAgg, FuncAnimation):

    #TODO: update Params
    def __init__(self, stream, parent=None, width=5, height=4, dpi=100,
                 channels=1, device=1, window=750, interval=25, blocksize=1024,
                 samplerate=48000, downsample=10, subtype='FLOAT'):

        self.stream = stream
        self.length = int(window * samplerate / (1000 * downsample))

        self.plotData = np.zeros((self.length, channels))
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvasQTAgg.__init__(self, fig)
        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(self, QSizePolicy.Expanding,
                                              QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        self.lines = self.axes.plot(self.plotData)
        self.axes.axis((0, len(self.plotData), -1, 1))
        self.axes.set_yticks([0])
        self.axes.yaxis.grid(True)
        self.axes.tick_params(bottom='off', top='off', labelbottom='off',
                       right='off', left='off', labelleft='off')
        fig.tight_layout(pad=0)

        FuncAnimation.__init__(self, fig, self.update_plot, interval=interval, blit=True)

    def update_plot(self, frame):
        while True:
            try:
                data = self.stream.plotQ.get_nowait()
            except queue.Empty:
                break

            shift = len(data)
            self.plotData = np.roll(self.plotData, -shift, axis=0)
            self.plotData[-shift:, :] = data
        for column, line in enumerate(self.lines):
            line.set_ydata(self.plotData[:, column])
        return self.lines

    # def toggle_state(self, recording):
    #     if recording:
    #         self.isRecording = True
    #         self.plotData = np.array([], dtype='float')
    #         self.recordingLine = matplotlib.lines.Line2D(np.array([]),
    #                                                      np.array([]))
    #     else:
    #         self.isRecording = False


class MicrophoneStream(QObject):

    recordingDone = pyqtSignal('PyQt_PyObject')

    def __init__(self, channels=[1], device=sounddevice.default.device[0], samplerate=44100, downsample=10, subtype='FLOAT'):
        super().__init__()

        self.subtype = subtype
        self.channels = channels
        self.isRecording = False
        self.sampleRate = samplerate
        self.downSample = downsample
        self.mapping = [c - 1 for c in channels]
        self.inputStream = sounddevice.InputStream(samplerate=samplerate,
                                                   device=device,
                                                   channels=max(channels),
                                                   callback=self.audio_callback)
        self.fileStream = None
        self.tempFile = None

        self.plotQ = queue.Queue()
        self.fileQ = queue.Queue()
        self.inputStream.start()

    def audio_callback(self, indata, frames, time, status):
        if self.inputStream.active:
            self.plotQ.put(indata[::self.downSample, self.mapping])
            if self.isRecording:
                self.fileQ.put(indata.copy())

    @pyqtSlot()
    def toggle_recording(self):
        self.isRecording = not self.isRecording

        if self.isRecording:
            # mkstemp returns (file handle: Int, file path: Str)
            self.tempFile = tempfile.mkstemp(prefix='temp_recording_',
                                             suffix='.wav',
                                             dir='')
            self.fileStream = soundfile.SoundFile(self.tempFile[1],
                                                  mode='w',
                                                  samplerate=self.sampleRate,
                                                  channels=max(self.channels),
                                                  subtype=self.subtype)

        if not self.isRecording:
            while True:
                try:
                    self.fileStream.write(self.fileQ.get_nowait())
                except queue.Empty:
                    break

            self.fileStream.close()
            self.recordingDone.emit(self.tempFile)


class AnalyzeTab(QWidget):

    analyzeDone = pyqtSignal('PyQt_PyObject', 'PyQt_PyObject', 'PyQt_PyObject', 'PyQt_PyObject',
                             'PyQt_PyObject', 'PyQt_PyObject', 'PyQt_PyObject')

    def __init__(self):
        super().__init__()

        # Layouts
        self.mainLayout = QHBoxLayout()
        self.lVbox = QVBoxLayout()
        self.lHbox = QHBoxLayout()
        self.lHbox_top = QHBoxLayout()
        self.rVbox = QVBoxLayout()
        self.rHbox = QHBoxLayout()
        self.rVbox2 = QVBoxLayout()
        self.stack = QStackedWidget()
        self.stack_Vbox = QVBoxLayout()
        self.stack_Hbox1 = QHBoxLayout()
        self.stack_Hbox2 = QHBoxLayout()
        self.hSplit = QSplitter(Qt.Horizontal)
        self.hSplit.setFrameShape(QFrame.StyledPanel)
        self.vSplit = QSplitter(Qt.Vertical)
        self.vSplit.setFrameShape(QFrame.StyledPanel)
        self.mainLayout.addLayout(self.lVbox, 1)
        self.mainLayout.addLayout(self.rVbox, 3)

        # Setup file browser
        self.fileModel = QFileSystemModel()
        self.fileModel.setNameFilters(['*.wav'])
        self.fileModel.setRootPath(QDir.currentPath())
        self.fileTree = QTreeView()
        self.fileTree.setModel(self.fileModel)
        self.fileTree.setRootIndex(self.fileModel.index(r'./'))
        self.fileTree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.fileTree.setColumnHidden(2, True)
        self.fileTree.setColumnHidden(1, True)
        self.rootDirEdit = QLineEdit(os.path.dirname(__file__))
        self.rootDirEdit.returnPressed.connect(self.on_edit_root)
        self.browseBtn = QPushButton('Browse')
        self.browseBtn.clicked.connect(self.on_browse)
        self.lHbox_top.addWidget(self.rootDirEdit, 3)
        self.lHbox_top.addWidget(self.browseBtn, 1)

        # Setup Canvas
        self.canvas = PlotCanvas(self)

        self.analyzeDone.connect(self.canvas.plot)
        self._analyze = lambda _: self.analyze(self.fileTree.selectedIndexes())
        self.analyzeBtn = QPushButton('Analyze')
        self.analyzeBtn.clicked.connect(self._analyze)

        ## BATCH ANALYSIS CONTROLS ##

        self.batchAnalyzeChk = QCheckBox('Batch Analysis')
        self.dataTable = QTableWidget()
        self.batchCtrlBox = QGroupBox("Batch Analysis")
        self.batchCtrlBox.setLayout(self.stack_Vbox)

        # Analysis Mode
        self.modeGroup = QButtonGroup()
        self.modeBox = QGroupBox('Analysis Mode')
        self.modeBox.setLayout(self.stack_Hbox1)
        self.stack_Vbox.addWidget(self.modeBox)
        self.wavAnalysisChk = QCheckBox('Wav analysis')
        self.wavAnalysisChk.setChecked(True)
        self.calibrationLocationBox = QComboBox()
        self.calibrationLocationBox.addItems([str(n) for n in range (1,11)])
        self.calibrationCurveChk = QCheckBox('Calibration Curve')
        self.calibrationCurveChk.toggled.connect(lambda state: self.calibrationLocationBox.setEnabled(state))
        self.calibrationCurveChk.setChecked(False)
        self.stack_Hbox1.addWidget(self.wavAnalysisChk, 3)
        self.stack_Hbox1.addWidget(self.calibrationCurveChk, 3)
        self.stack_Hbox1.addWidget(QLabel('Location: '), 1)
        self.stack_Hbox1.addWidget(self.calibrationLocationBox, 1)
        self.stack_Vbox.addLayout(self.stack_Hbox1)
        self.modeGroup.addButton(self.wavAnalysisChk)
        self.modeGroup.addButton(self.calibrationCurveChk)
        self.modeGroup.setExclusive(True)

        # Outputs
        self.outputCtrlBox = QGroupBox('Outputs')
        self.outputCtrlBox.setLayout(self.stack_Hbox2)
        self.stack_Vbox.addWidget(self.outputCtrlBox)
        self.toCSVchk = QCheckBox('.csv')
        self.toJSONchk = QCheckBox('.json')
        self.toCSVchk.stateChanged.connect(lambda _: self.update_settings('output',
                                                                          'toCSV', self.toCSVchk.isChecked()))
        self.toJSONchk.stateChanged.connect(lambda _: self.update_settings('output',
                                                                          'toJSON', self.toJSONchk.isChecked()))

        self.stack_Hbox2.addWidget(self.toCSVchk)
        self.stack_Hbox2.addWidget(self.toJSONchk)
        self.stack_Vbox.addLayout(self.stack_Hbox2)

        self.stack.addWidget(self.dataTable)
        self.stack.addWidget(self.batchCtrlBox)
        self.stack.setCurrentWidget(self.dataTable)
        self.stack.show()
        self.batchAnalyzeChk.stateChanged.connect(self.toggle_stack)
        self.batchAnalyzeChk.setChecked(False)
        self.stack_Vbox.addStretch()

        ## PROCESSING CONTROLS ##
        self.processControls = QGroupBox('Signal Processing')
        self.tOffsetSlider = QSlider(Qt.Horizontal,)
        self.tOffsetSlider.setMinimum(1)
        self.tOffsetSlider.setMaximum(100)
        self.tOffsetSlider.setValue(100)
        self.tOffsetSlider.setTickPosition(QSlider.TicksBelow)
        self.tOffsetSlider.setTickInterval(10)
        self.tOffsetSlider.valueChanged.connect(lambda val: self.update_settings('processing','tChop', val))
        self.tOffsetLayout = QHBoxLayout()
        self.tOffsetSlider_Box = QGroupBox(f'Chop Signal - {self.tOffsetSlider.value()}%')
        self.tOffsetSlider.valueChanged.connect(lambda val: self.tOffsetSlider_Box.setTitle(f'Chop Signal - {val}%'))
        self.tOffsetSlider_Box.setLayout(self.tOffsetLayout)
        self.tOffsetLayout.addWidget(self.tOffsetSlider)

        self.nFFTSlider = QSlider(Qt.Horizontal,)
        self.nFFTSlider.setMinimum(1)
        self.nFFTSlider.setMaximum(16)
        self.nFFTSlider.setValue(1)
        self.nFFTSlider.setTickPosition(QSlider.TicksBelow)
        self.nFFTSlider.setTickInterval(2)
        self.nFFTSlider.valueChanged.connect(lambda val: self.update_settings('processing','detail', val))
        self.nFFTLayout = QHBoxLayout()
        self.nFFTSlider.valueChanged.connect(lambda val: self.nFFTSlider_Box.setTitle(f'FFT Size - {val*65536}'))
        self.nFFTSlider_Box = QGroupBox(f'FFT Size - {self.nFFTSlider.value()*65536}')
        self.nFFTSlider_Box.setLayout(self.nFFTLayout)
        self.nFFTLayout.addWidget(self.nFFTSlider)

        self.rVbox2.addWidget(self.tOffsetSlider_Box)
        self.rVbox2.addWidget(self.nFFTSlider_Box)
        self.processControls.setLayout(self.rVbox2)

        self.lVbox.addLayout(self.lHbox_top, 1)
        self.lVbox.addWidget(self.fileTree, 7)
        self.lVbox.addLayout(self.lHbox, 1)
        self.lHbox.addWidget(self.analyzeBtn, 2)
        self.lHbox.addWidget(self.batchAnalyzeChk, 1)
        self.vSplit.addWidget(self.canvas)
        self.vSplit.addWidget(self.hSplit)
        self.rVbox.addWidget(self.vSplit)
        self.hSplit.addWidget(self.stack)
        self.hSplit.addWidget(self.processControls)

        self.settings = {
            'processing': {'tChop': self.tOffsetSlider.value(), 'detail': self.nFFTSlider.value()},

            'output': {'toCSV': self.toCSVchk.isChecked(), 'toJSON': self.toJSONchk.isChecked()}
        }
        self.setLayout(self.mainLayout)

    def on_browse(self):
        # Browse to file tree root directory
        options = QFileDialog.Options()
        path = QFileDialog.getExistingDirectory(self, caption="Choose root directory", options=options)
        self.rootDirEdit.setText(path)
        self.fileTree.setRootIndex(self.fileModel.index(path))

    def on_edit_root(self):
        # Update the file tree root directory
        self.fileTree.setRootIndex(self.fileModel.index(self.rootDirEdit.text()))

    def update_settings(self, category, setting, value):
        # Update settings and reprocess FFT if in single analysis mode
        self.settings[category][setting] = value

        if category == 'processing' and self.fileTree.selectedIndexes():
            self.analyze(self.fileTree.selectedIndexes())

    def toggle_stack(self, state):
        if state == 2:
            self.stack.setCurrentWidget(self.batchCtrlBox)
            self.fileTree.setSelectionMode(QAbstractItemView.MultiSelection)
        else:
            self.stack.setCurrentWidget(self.dataTable)
            self.fileTree.setSelectionMode(QAbstractItemView.SingleSelection)

    def analyze(self, filePaths):
        if self.batchAnalyzeChk.isChecked():
            if self.wavAnalysisChk.isChecked():
                self.batch_analyze_wav([self.fileModel.filePath(path) for path in filePaths[::4]])
            if self.calibrationCurveChk.isChecked():
                self.generate_calibration_curve([self.fileModel.filePath(path) for path in filePaths[::4]])

        else:
            if os.path.isdir(self.fileModel.filePath(filePaths[0])) or len(filePaths) > 4:
                QMessageBox.information(self, 'Error', 'Please select only 1 file for single analysis.')
                return
            self.single_analyze_wav(self.fileModel.filePath(filePaths[0]))

    def single_analyze_wav(self, filePath):
        """
        Do an FFT and find peaks on a single wav file

        :param filePath: file path to .wav file
        """

        tChopped, vChopped, fVals,\
        powerFFT, peakFreqs, peakAmps = Utils.AnalyzeFFT(filePath, tChop=self.settings['processing']['tChop'],
                                                                   detail=self.settings['processing']['detail'])

        self.analyzeDone.emit(tChopped, vChopped, fVals, powerFFT, peakFreqs, peakAmps, filePath)
        self.update_table(peakFreqs, peakAmps)

    def batch_analyze_wav(self, filePaths):
        """
        Perform a batch analysis of many .wav files. Outputs FFTs and peaks in .csv or .json format

        :param filePaths: A list of folders containing the .wav files to be analyzed
        """

        toCSV = self.settings['output']['toCSV']
        toJSON = self.settings['output']['toJSON']

        start = time.time()

        fileTotal = 0
        for path in filePaths:
            if os.path.isdir(path):
                blockName = os.path.basename(path)
                print(f'Block: {blockName}')

                files = [os.path.join(path, file) for file in os.listdir(path) if '.wav' in file]
                fileTotal += len(files)

                if toCSV:
                    if not os.path.exists(os.path.join(path, 'fft_results_csv')):
                        os.makedirs(os.path.join(path, 'fft_results_csv'))
                    resultFilePath = os.path.join(path, 'fft_results_csv')

                    print('Processing FFTs...')
                    with multiprocessing.Pool(processes=4) as pool:
                        results = pool.starmap(Utils.AnalyzeFFT, zip(files, itertools.repeat(True),
                                                                            itertools.repeat(True)))
                    results = [result for result in results if result is not None]

                    peaks = [result[0] for result in results]
                    ffts = [result[1] for result in results]

                    print('Writing to .csv...')
                    resultFileName = os.path.join(resultFilePath, f'{blockName}_Peaks.csv')
                    peakFrames = pd.concat(peaks)
                    peakFrames.to_csv(resultFileName, index=False, header=True)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                        executor.map(self.multi_csv_write, ffts)

                if toJSON:
                    if not os.path.exists(os.path.join(path, 'fft_results_json')):
                        os.makedirs(os.path.join(path, 'fft_results_json'))
                    print(os.path.join(path, 'fft_results_json'))

                    print('Processing FFTs...')
                    with multiprocessing.Pool(processes=4) as pool:
                        results = pool.starmap(Utils.AnalyzeFFT, zip(files, itertools.repeat(True),
                                                                            itertools.repeat(False),
                                                                            itertools.repeat(True)))
                        results = [result for result in results if result is not None]

                    print('Writing to .json...')
                    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                        executor.map(self.multi_json_write, results)

        end = time.time()
        print(f'**Done!** {len(filePaths)} blocks with {fileTotal} files took {round(end-start, 1)}s')

    def generate_calibration_curve(self, filePaths):
        """
        Attempt to fit an exponential function to a set of data points (x: Peak Frequency, y: Compressive strength)
        provided in JSON format.

        ex:{
              "shape": "2-Hole",
              "testData": {
                "location": "1",
                "strength": 3.092453552,
                "peaks": [
                  {
                    "frequency": 1134.5561082797967,
                    "magnitude": 0.349102384777402
                  }]
              },
              "waveData": [...],
              "freqData": [...]
        }

        Plot the curve, data points and give the function if successful.

        ** NOTE ** This function is still experimental and a bit buggy. Sometimes the scipy.optimize curve_fit won't
        converge with the initial guess given for the coeffecients. You're probably better off writing your own code.

        :param filePaths: A list of folders containing .jsons
        """
        # Strike Location
        location = self.calibrationLocationBox.currentText()

        # Function to fit to the data
        exp_f = lambda x, a, b, c: a * np.exp(b * x) + c

        # Threaded method for opening all the .jsons and fitting
        calibCurve = ThreadedCalibrationCurve(filePaths, location, exp_f)
        progressDialog = QProgressDialog(f'Gettings samples for location: {location}', None, 0, len(filePaths), self)
        progressDialog.setModal(True)
        calibCurve.blocksSearched.connect(progressDialog.setValue)
        try:
            peakFreqs, strengths, popt, pcov, fitX = calibCurve.run()
        except Exception as e:
            QMessageBox.information(self, 'Error', e)
            return

        # Calculate R Squared
        residuals = strengths - exp_f(peakFreqs, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((strengths - np.mean(strengths)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Plot Results
        fig = Figure()
        plt.scatter(peakFreqs, strengths)
        plt.plot(fitX, exp_f(fitX, *popt), '-k')
        ax = plt.gca()
        plt.text(0.05, 0.9, f'y = {round(popt[0],3)}*exp({round(popt[1], 5)}x) + {round(popt[2], 3)}\n',
                 ha='left', va='center', transform=ax.transAxes)
        plt.text(0.05, 0.85, f'R^2 = {round(r_squared,3)}',
                 ha='left', va='center', transform=ax.transAxes)

        plt.title(f'Calibration Curve, Location: {location}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Compressive Strength (MPa)')

        plt.show()

    def multi_csv_write(self, frameTuple):
        frame = frameTuple[1]
        wavPath = frameTuple[0]

        resultFileDir = os.path.join(os.path.dirname(wavPath), 'fft_results_csv')
        resultFileName = os.path.basename(wavPath)+'_fft.csv'
        resultFilePath = os.path.join(resultFileDir, resultFileName)

        frame.to_csv(resultFilePath, index=False, header=True)

    def multi_json_write(self, results):
        data = results[0]
        wavPath = results[1]

        jsonFileDir = os.path.join(os.path.dirname(wavPath), 'fft_results_json')
        resultFileName = os.path.basename(wavPath) + '_fft.json'
        resultFilePath = os.path.join(jsonFileDir, resultFileName)

        # blockName = os.path.basename(os.path.dirname(wavPath))
        # blockDir = os.path.join(jsonFileDir, blockName)
        # if not os.path.exists(blockDir):
        #     os.makedirs(blockDir)
        # print(resultFilePath)
        with open(resultFilePath, 'w') as f:
            json.dump(data, f, indent=2)

    def update_table(self, peakFreqs, peakAmps):
        """

        :param peakFreqs:
        :param peakAmps:
        :return:
        """
        self.dataTable.setRowCount(2)
        self.dataTable.setColumnCount(len(peakFreqs)+1)

        self.dataTable.setItem(0, 0, QTableWidgetItem("Frequencies: "))
        self.dataTable.setItem(1, 0, QTableWidgetItem("Powers: "))

        for col, freq in enumerate(peakFreqs, start=1):
            self.dataTable.setItem(0, col, QTableWidgetItem(str(round(freq))))
        for col, power in enumerate(peakAmps, start=1):
            item = QTableWidgetItem(str(round(power, 3)))
            if power > 0.7:
                item.setBackground(QColor(239, 81, 28))
            elif power >= 0.4:
                item.setBackground(QColor(232, 225, 34))
            elif power < 0.4:
                item.setBackground(QColor(113, 232, 34))
            self.dataTable.setItem(1, col, item)


class ThreadedCalibrationCurve(QThread):
    """
    QThread implementation for opening jsons and fitting the freq/strength data
    """

    # Progress Signal
    blocksSearched = pyqtSignal(int)

    def __init__(self, filePaths, location, f):
        QThread.__init__(self, parent=None)

        self.filePaths = filePaths
        self.location = [location, location + 'a']
        self.f = f

    def run(self):
        """
        Method ran inside the QThread.

        Get all the peaks and strengths from the JSONs in 'filePaths'. Emit progress to the progress dialog.

        :return Fit coefficients and frequency/strength data
        """

        peakFreqs = np.array([], dtype='float')
        strengths = np.array([], dtype='float')

        _blocksSearched = 0
        self.blocksSearched.emit(_blocksSearched)

        for path in self.filePaths:
            if os.path.isdir(path):
                blockName = os.path.basename(path)
                print(f'Block: {blockName}')

                files = [os.path.join(path, file) for file in os.listdir(path) if '.json' in file]
                locationsFound = 0
                for file in files:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        if data['testData']['location'] in self.location and Utils.blockStrengths[blockName]:
                            locationsFound += 1
                            maxPeak = max(data['testData']['peaks'], key=lambda peak: peak['magnitude'])
                            peakFreqs = np.append(peakFreqs, maxPeak['frequency'])
                            strengths = np.append(strengths, Utils.blockStrengths[blockName])

                print(f'Found {locationsFound} files for strike location {self.location}')
            _blocksSearched += 1
            self.blocksSearched.emit(_blocksSearched)

        if peakFreqs.size == 0 or strengths.size == 0:
            #QMessageBox.information(self, 'Error', f"No .json's found for location {location}")
            return -2

        peakFreqs, strengths = Utils.removeOutliers(peakFreqs, strengths, outlierConstant=0.8)
        fitX = np.linspace(min(peakFreqs) - 100, max(peakFreqs) + 100, 100)

        try:
            popt, pcov = curve_fit(self.f, peakFreqs, strengths, p0=(0.5, 0.001, 1))
        except:
            QMessageBox.information(self, 'Error', 'Unable to fit points. Did you provide enough samples?'
                                                   ' Coefficient guess may need to be adjusted. (curve_fit())')
            return

        return peakFreqs, strengths, popt, pcov, fitX


class PlotCanvas(FigureCanvasQTAgg):
    """
    Matplotlib canvas used to display Time and frequency data in subplots.
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvasQTAgg.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

        self.timePlot = self.fig.add_subplot(211)
        self.timePlot.set_title('Time Plot')
        self.freqPlot = self.fig.add_subplot(212)
        self.freqPlot.set_title('Frequency Plot')

    def plot(self, tChopped, vChopped, fVals, powerFFT, peakFreqs, peakAmps, filePath):
        self.timePlot.clear()
        self.timePlot.plot(tChopped, vChopped)
        self.timePlot.set_title(filePath)
        self.timePlot.set(xlabel='Time (s)')

        self.freqPlot.clear()
        self.freqPlot.plot(fVals, powerFFT)
        self.freqPlot.plot(peakFreqs, peakAmps, 'ro')
        self.freqPlot.set_xlim(0, 11025)
        self.freqPlot.set(xlabel='Frequency (Hz)')
        self.fig.subplots_adjust(hspace=0.42)
        self.draw()


if __name__ == '__main__':
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    ex = ImpulseAnalysisTool()
    sys.exit(app.exec_())