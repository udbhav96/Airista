import cv2
import fitz
import numpy as np

doc = fitz.open("Presentation1.pdf")
slide_images = []

for page in doc:
    pix = page.get_pixmap(alpha=True)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)
    slide_images.append(img)

current_slide = 0
total_slides = len(slide_images)

video = cv2.VideoCapture(0)

cv2.namedWindow("Webcam Presenter", cv2.WINDOW_NORMAL)

while True:
    ret, frame = video.read()
    if not ret:
        print("Webcam failed.")
        break

    frame = cv2.flip(frame, 1)
    slide_img = slide_images[current_slide]

    slide_img_resized = cv2.resize(slide_img, (frame.shape[1], frame.shape[0]))

    slide_rgb = slide_img_resized[:, :, :3]
    alpha = 0.4
    overlayed = cv2.addWeighted(slide_rgb, alpha, frame, 1 - alpha, 0)

    cv2.imshow("Webcam Presenter", overlayed)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n') and current_slide < total_slides - 1:
        current_slide += 1
    elif key == ord('b') and current_slide > 0:
        current_slide -= 1

    if cv2.getWindowProperty("Webcam Presenter", cv2.WND_PROP_VISIBLE) < 1:
        break

video.release()
cv2.destroyAllWindows()