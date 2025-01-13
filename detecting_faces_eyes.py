
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 6)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_face_gray = gray[y:y+w, x:x+w]
        roi_face = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_face_gray, 1.2, 6)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            roi_eyes=roi_face[ey:ey+eh,ex:ex+ew]

            resized_img = cv2.resize(roi_eyes, (224, 224))
            cv2.imwrite("eye.png", resized_img)
            print(resized_img.shape)
            cv2.imshow('Live Cam ',roi_face)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()