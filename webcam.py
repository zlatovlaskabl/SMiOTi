import sys, os, logging
from datetime import datetime

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi

import cv2 as cv
import tensorflow as tf
import numpy as np

model_path = 'domino.tflite'

boxes_idx, classes_idx, scores_idx = 1, 3, 0

min_conf_threshold = 0.8 # minimum confidence threshold
imW = 640 # image width
imH = 480 # image height

labels = ['domino'] # class

# interface class
class Interface(QDialog):
    
    def __init__(self):
        super(Interface, self).__init__()

        # target number of detected objects (input)
        self.validate = 0
        self.debug = 0
        self.target = 0

        # load ui
        loadUi('inter.ui', self)
        self.numberText.setText("0")
        self.spinBox.setText("0")

        # thread
        self.Processing = Processing()

        # buttons
        self.startWebcam.clicked.connect(self.Start)
        self.stopWebcam.clicked.connect(self.CancelFeed)
        self.valButton.clicked.connect(self.Validate)
        self.debButton.clicked.connect(self.Debug)

        # connect signal to slot
        self.Processing.ImageUpdate.connect(self.ImageUpdateSlot)

    # start
    def Start(self):
        self.Processing.start()
        self.logEdit.append(f"Captura video deschisa cu succes")

    # validate
    def Validate(self):
        self.target = int(self.spinBox.toPlainText())    # read target (input) number of detected objects
        self.validate = 1

    # Debug
    def Debug(self):
        self.target = int(self.spinBox.toPlainText())    # read target (input) number of detected objects
        self.debug = 1

    # slot (function)
    def ImageUpdateSlot(self, Coords, Frame, Image, DetectionsNumber):
        self.videoLabel.setPixmap(QPixmap.fromImage(Image)) # display image on videoLabel
        self.numberText.setText(f"{DetectionsNumber}") # write number of detected objects

        # save screen if number of detections = target
        if self.validate and self.target and self.target == DetectionsNumber:
            if "poze" not in os.listdir('.') and not os.path.isfile("poze"):
                os.mkdir("poze")

            name = r'{}'.format("poze/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".jpg")
            self.logEdit.append(f"Poza salvata cu succes: {name}")
            cv.imwrite(name, cv.cvtColor(Frame, cv.COLOR_BGR2RGB))

        # save log if number of detections != target
        if self.debug and self.target and self.target != DetectionsNumber:
            if "logs" not in os.listdir('.') and not os.path.isfile("logs"):
                os.mkdir("logs")

            name = r'{}'.format("logs/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".log")
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(message)s',
                filename=name,
                filemode='w',
            )

            logging.debug(f'Obiecte detectate: {DetectionsNumber} \nObiecte de detectat: {self.target} \nCoordonatele obiectelor detectate: {Coords}')
            self.logEdit.append(f"Fisier log salvat cu succes: {name}")

        self.validate = 0
        self.debug = 0
        self.target = 0

    # cancel
    def CancelFeed(self):
        self.Processing.stop()
        self.logEdit.append(f"Captura video oprita cu succes")

# processing class
# retrieves img & converts it & processes it & sends signal back
class Processing(QThread):

    # signal
    ImageUpdate = pyqtSignal(list, np.ndarray, QImage, int)
    
    def run(self):

        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # load the input shape required by the model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.ThreadActive = True
        capture = cv.VideoCapture(0)

        while self.ThreadActive:
            isTrue, frame = capture.read()
                
            if isTrue:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # BGR >> RGB
                frame = cv.resize(frame_rgb, (imW, imH))
                frame_resized = cv.resize(frame, (320, 320)) # resize to model shape

                input_data = np.expand_dims(frame_resized, axis=0)  # convert to batch shape

                # perform actual detection by running the model with the image as input
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # retrieve detection results
                boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
                classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
                scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

                n = len([x for x in scores if x > min_conf_threshold])  # number of detected objects
                coords = []

                for i in range(len(scores)):
                    if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))

                        coords.append([xmin, ymin, xmax, ymax])

                        # Draw rectangle from bounding box coordinates
                        cv.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                        # Draw label
                        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'domino: 72%'
                        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        cv.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv.FILLED) # Draw white box to put label text in
                        cv.putText(frame, label, (xmin, label_ymin-7), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                # create QImage object
                qimage = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                resized = qimage.scaled(imW, imH, Qt.KeepAspectRatio)

                # send/emit signal back to main GUI thread
                self.ImageUpdate.emit(coords, frame, resized, n)

        capture.release()
        
    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    interface = Interface()
    interface.show()
    interface.setWindowTitle("Aplicatie Procesare Imagini")

    sys.exit(app.exec())
