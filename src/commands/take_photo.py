from uvc_camera import UVCCamera
import cv2

cam = UVCCamera(0, capture_size=(2560,1440), fps=30)
cam.capture()

for i in range(50):
    cam.update_frame()

frame = cam.color_frame()
# frame = cv2.flip(frame, -1)
cv2.imwrite("p.jpg", frame)