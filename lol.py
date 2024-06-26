from flask import Flask
import cv2
import face_recognition

myApp = Flask(__name__)

@myApp.route('/')
def index():
    cap = cv2.VideoCapture(0)
    frame_resizing = 0.25

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), fx=frame_resizing, fy=frame_resizing)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            print('matches',matches)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                
                faceLoc = np.array(faceLoc)
                faceLoc = faceLoc / 0.25
                faceLoc=faceLoc.astype(int)
                #y1,x2,y2,x1 = faceLoc # in the other code we risize the face
                #y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                y1, x2, y2, x1 = faceLoc[0], faceLoc[1], faceLoc[2], faceLoc[3]
            
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                print("Open the Doooooooooooooooooooooooooooooooooooooooooooooooooooooor")
    
        cv2.imshow('Webcam',img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return 'Hello from Flask Server'

@myApp.route('/Eslam')
def name():
    return 'Hello, Eslam'

if __name__ == '__main__':
    myApp.run(debug=True, port=9000)