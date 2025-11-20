import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.detectMarkers(gray, dictionary)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        detected = []
        for i, _id in enumerate(ids):
            detected.append(int(_id[0]))
            x = int(corners[i][:, 0].mean())
            y = int(corners[i][:, 1].mean())
            print(f"center: ({x}, {y})")
        print(detected)

    cv2.imshow("Aruco Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
