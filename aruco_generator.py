import cv2
import cv2.aruco as aruco
import random

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

marker_amount = 2
marker_ids = []
marker_size = 300
for i in range(marker_amount):
    marker_id = random.randint(0, 49)
    if marker_id in marker_ids:
        continue
    marker_ids.append(marker_id)
    marker_img = aruco.generateImageMarker(dictionary, marker_id, marker_size)
    cv2.imwrite(f"aruco_{i}_{marker_id}.png", marker_img)
    print(f"aruco_{i}_{marker_id}.png saved")
