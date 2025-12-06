# inference_gui.py
import sys, os
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas

MODEL_PATH = "runs/detect/rtg_anomaly/weights/best.pt"  # adjust after training

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTG YOLO Anomaly Viewer")
        self.model = None
        self.image = None
        self.orig_np = None
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        btn_load = QtWidgets.QPushButton("Load Image (.bmp)")
        btn_load.clicked.connect(self.load_image)
        btn_load_model = QtWidgets.QPushButton("Load Model")
        btn_load_model.clicked.connect(self.load_model)
        btn_run = QtWidgets.QPushButton("Run Analysis")
        btn_run.clicked.connect(self.run_analysis)
        self.canvas = QtWidgets.QLabel()
        self.canvas.setFixedSize(1000,600)
        self.canvas.setStyleSheet("background-color: #222;")
        layout.addWidget(btn_load)
        layout.addWidget(btn_load_model)
        layout.addWidget(btn_run)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open BMP", "", "Bitmap Files (*.bmp)")
        if not path:
            return
        im = Image.open(path).convert('L')
        self.orig_np = np.array(im)
        qimg = QtGui.QImage(self.orig_np.data, self.orig_np.shape[1], self.orig_np.shape[0], self.orig_np.strides[0], QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.canvas.size(), QtCore.Qt.KeepAspectRatio)
        self.canvas.setPixmap(pix)
        self.image_path = path

    def load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load YOLO model", "", "Weights (*.pt);;All Files (*)")
        if not path:
            return
        self.model = YOLO(path)
        QtWidgets.QMessageBox.information(self, "Model", "Model załadowany: " + path)

    def run_analysis(self):
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Załaduj najpierw model.")
            return
        if not hasattr(self, 'image_path'):
            QtWidgets.QMessageBox.warning(self, "Error", "Załaduj obraz.")
            return
        results = self.model.predict(self.image_path, imgsz=1280, conf=0.25, max_det=50)
        # results[0].boxes -> xyxy, conf, cls
        im = cv2.cvtColor(self.orig_np, cv2.COLOR_GRAY2BGR)
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cv2.rectangle(im, (xyxy[0],xyxy[1]), (xyxy[2],xyxy[3]), (0,0,255), 2)
            cv2.putText(im, f"{conf:.2f}", (xyxy[0], max(0, xyxy[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # prepare QPixmap
        h,w = im.shape[:2]
        bytes_per_line = 3*w
        qimg = QtGui.QImage(im.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.canvas.size(), QtCore.Qt.KeepAspectRatio)
        self.canvas.setPixmap(pix)
        # opcjonalnie: eksport do PDF
        self.export_report(im, results[0])

    def export_report(self, im_np, result):
        # Simple PDF: zapis obrazu z boxami
        out_pdf = "report.pdf"
        # zapisz tymczasowo obraz
        tmp_img = "tmp_detected.png"
        cv2.imwrite(tmp_img, im_np)
        c = canvas.Canvas(out_pdf, pagesize=(im_np.shape[1], im_np.shape[0]))
        c.drawImage(tmp_img, 0, 0, width=im_np.shape[1], height=im_np.shape[0])
        c.showPage()
        c.save()
        os.remove(tmp_img)
        QtWidgets.QMessageBox.information(self, "Report", f"Raport zapisany: {out_pdf}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
