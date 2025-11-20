from ultralytics import YOLO
import numpy as np
import cv2.aruco as aruco
import cv2
import torch
import time

# MiDas
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

transform = midas_transforms.small_transform

depth_value = 0
real_depth_value = 0
mouse_x, mouse_y = 0, 0
def show_depth(event, x, y, _, param):
    global depth_value, real_depth_value, mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
        depth_value = param[y, x]
        real_depth_value = a * depth_value + b

cv2.namedWindow("depth")

# ArUco markers
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
markers = {} # id: depth (cm)
a, b = 0, 0 # scale factors
for _ in range(2):
    marker = input("marker_id marker_depth (cm): ")
    if marker == "":
        break
    markers[int(marker.split()[0])] = int(marker.split()[1])

def get_registered(_ids, _markers, _corners):
    registered_markers = []
    registered_corners = []
    if _ids is None:
        return registered_markers, registered_corners
    for i, _id in enumerate(_ids):
        if _id[0] in _markers:
            registered_markers.append([_id[0]])
            registered_corners.append(_corners[i])
    print((registered_markers, registered_corners))
    return registered_markers, registered_corners # ([[0], [1], ...], [corner0, corner1, ...])


# YOLO
model = YOLO('models/yolo11s-seg.pt')
cap = cv2.VideoCapture(0)

prev_crops = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FPS
    start = time.time()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    # YOLO
    results = model(frame, stream=True)

    # MiDaS
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map_norm = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)
    cv2.setMouseCallback("depth", show_depth, depth_map_norm)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    depth_map_vis = (depth_map_norm * 255).astype(np.uint8)
    depth_map_vis = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_MAGMA)

    # ArUco
    corners, ids, rejected = aruco.detectMarkers(gray, dictionary)
    keys, corners = get_registered(ids, markers, corners)
    aruco.drawDetectedMarkers(frame, corners, np.array(keys))
    centers = [(int(corner[:, 0].mean()), int(corner[:, 1].mean())) for corner in corners]
    ...

    # FPS
    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    # Depth info
    cv2.putText(
        depth_map_vis,
        f"depth_norm: {depth_value:.3f} depth: {real_depth_value:.3f} fps: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.imshow("depth", depth_map_vis)

    current_crops = set()

    for result in results:
        cv2.imshow('result', result.plot())
        masks = result.masks
        boxes = result.boxes

        if result.masks is not None:
            for i, mask in enumerate(result.masks.xy):
                pts = np.array(mask, dtype=np.int32)

                mask_img = np.zeros_like(frame)
                cv2.fillPoly(mask_img, [pts], (255, 255, 255))

                mask_gray = gray if len(mask_img.shape) == 3 else mask_img
                masked_depth = depth_map_norm[mask_gray > 0]
                depth = masked_depth.mean() if len(masked_depth) > 0 else 0

                obj_only = cv2.bitwise_and(frame, mask_img)

                cv2.putText(
                    obj_only,
                    f"avg_depth: {depth:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                cropped_name = f"{i}: {model.names[int(boxes[i].cls[0])]}"
                current_crops.add(cropped_name)

                cv2.imshow(cropped_name, obj_only)

    for old_crop in prev_crops - current_crops:
        cv2.destroyWindow(old_crop)

    prev_crops = current_crops

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
