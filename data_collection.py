import cv2
import numpy as np
import warnings


warnings.ignore()

def saveFaceData(prn):
    path = "data/"
    warnings.filterwarnings("ignore")
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(r"C:\Users\archi\Desktop\Faceial-Recogniton-Based-Security-System\haarcascade_frontalface_alt.xml")

    skip = 0
    face_data = []
    print("Capturing Face Data. Please Wait...")

    while True:

        if skip / 10 == 40:
            break
        
        ret, frame = cap.read()
        if ret == False:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(frame, 1.3, 4)
        faces = sorted(faces, key = lambda f:f[2]*f[3], reverse = True)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        try:
            l = faces[0]
            x, y, w, h = l
            offset = 10
            face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            face_section = cv2.resize(face_section, (100, 100))
            
            skip += 1
            if skip%10 ==0:
                face_data.append(face_section)
                print(skip/10)
            cv2.imshow("Face Section", face_section)
        except:
            a=0
        finally:
            cv2.imshow("Frame", frame)
            
            # cv2.imshow("bw", gray)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                break

    face_data=np.asarray(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))
    np.save(path+filename+'.npy', face_data)
    print("Data Saved.")

    cap.release()
    cv2.destroyAllWindows()