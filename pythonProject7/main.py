
import cv2
import numpy as np



class FaceDetector:
         def __init__(self,prototxt_path="deploy.prototxt.txt",model_path="res10_300x300_ssd_iter_140000.caffemodel",confidence=0.5):
             self.net= cv2.dnn.readNetFromCaffe(prototxt_path,model_path)
             self.confidence=confidence

         def detect(self,frame):
             detected_faces=[]
             (h,w)=frame.shape[:2]
             blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
             self.net.setInput(blob)
             detections= self.net.forward()
             for i in range(0,detections.shape[2]):
                 confidence=detections[0,0,i,2]
                 if confidence > self.confidence:
                     box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                     (x1,y1,x2,y2)= box.astype("int")
                     detected_faces.append({'start':(x1,y1),'end':(x2,y2),'confidence':confidence})

             return detected_faces
         def draw(self,frame):
             detected_faces=self.detect(frame)
             for face in detected_faces:
                 cv2.rectangle(frame,face['start'],face['end'],(0,255,0),2)
                 y=face['start'][1]-10 if face['start'][1]-10 > 10 else face['start'][1]+10
                 cv2.putText(frame,"{:.4f}".format(face['confidence']),(face['start'][0],y),cv2.FONT_HERSHEY_DUPLEX,0.45,(0,255,0),2)
             return frame


def Image(path):
    face_detector = FaceDetector()
    frame = cv2.imread(path)
    annotated_frame = face_detector.draw(frame)
    cv2.imshow('faces', annotated_frame)
    cv2.waitKey(0)



def Video():
    face_detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    while not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cv2.waitKey(1000)
    label = ''
    cntr = 1
    while True:
        flag, frame = cap.read()
        if flag:
            frame = cv2.flip(frame, 1)
            annotated_frame = face_detector.draw(frame)
            cv2.imshow('camera_feed', annotated_frame)
        # cv2.imshow('input',frame[100:300,400:600])

        else:
            cv2.waitKey(1000)
        k = cv2.waitKey(5)
        if k == 27:
            break


if __name__ == "__main__":
    Video()
    cv2.destroyAllWindows()
