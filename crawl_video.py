import json
import os

import cv2
from vidgear.gears import CamGear

# Read json file
with open("config/config.json") as file:
    data = json.load(file)


# Define the Youtube video URL
youtube_url = data["youtube_video_url"]

# Create a Pafy object to fetch the video stream
stream = CamGear(
    source=youtube_url, stream_mode=True, logging=True
).start()  # YouTube Video URL as input

# Define the path to save video
video_path = data["video_path"]

# if video_path doesn't exist, make folder to save
if not os.path.exists(video_path):
    dirpath = os.path.dirname(video_path)
    os.mkdir(dirpath)

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
frame_rate = data["fps"]
[frame_width, frame_height] = data["frame_size"]

out = cv2.VideoWriter(
    filename=video_path,
    fourcc=fourcc,
    fps=frame_rate,
    frameSize=(frame_width, frame_height),
)

# Define button "record"
btn_record = False


while True:
    # read frame
    frame = stream.read()

    if frame is None:
        # if True, break the infinite loop
        break

    frame = cv2.resize(frame, (1600, 800))
    cv2.imshow("Lane video", frame)

    if cv2.waitKey(1) == ord("r"):
        btn_record = True

    if btn_record:
        out.write(frame)

    if cv2.waitKey(1) == ord("q"):
        break

# Close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()
