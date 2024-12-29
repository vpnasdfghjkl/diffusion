import cv2

# 打开视频文件
video_path = 'KuavoToy_2024-12-29-22-57-59_speedx.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # 获取视频的帧宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建视频写入对象
    output_path = 'output_rgb_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # 或使用 'XVID' 或 'H264' 等
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        # 读取视频的一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 将 BGR 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将 RGB 帧写入输出视频
        out.write(rgb_frame)

    cap.release()
    out.release()
    print(f"Output video saved to {output_path}")
