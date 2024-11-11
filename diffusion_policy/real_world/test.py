import cv2

# 尝试找到可用的摄像头
available_cameras = []

for i in range(12):  # 假设设备编号在 0 到 11 之间
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
        cap.release()

if not available_cameras:
    print("未检测到可用的摄像头。")
else:
    print(f"检测到的摄像头设备编号：{available_cameras}")
    caps = [cv2.VideoCapture(i) for i in available_cameras]

    while True:
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f'Camera {available_cameras[idx]}', frame)
        
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
