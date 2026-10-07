import os
import cv2
from ultralytics import YOLO

# load yolo model
model = YOLO("yolov8n.pt")


video_path = os.path.join(".", "test2.mp4")
cap = cv2.VideoCapture(video_path)

ret = True

# read frames
while ret:
    ret, frame = cap.read()

    # detect objs
    # persist let the model remeber all the objs seen in past frame
    results = model.track(frame, persist=True)

    # plot results
    # get the current frame, in the video is treated like an array but should be an iteretor (?)
    result = next(iter(results))
    frame_ = result.plot()

    # visualize
    cv2.imshow("current frame", frame_)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
