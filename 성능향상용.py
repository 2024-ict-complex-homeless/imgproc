import cv2
import numpy as np
import argparse
import imutils
from collections import deque
from imutils.video import VideoStream
import time
import json

# 인수 파서를 구성하고 인수를 구문 분석합니다.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="(선택 사항) 비디오 파일 경로")
ap.add_argument("-b", "--buffer", type=int, default=64, help="최대 버퍼 크기")
args = vars(ap.parse_args())

# tablet_Lower = (0, 0, 200)
# tablet_Upper = (180, 20, 255)  # 흰색

tablet_Lower = (20, 40, 100)
tablet_Upper = (50, 250, 250) #노란색

tablet_pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 얼굴 검출을 위한 카스케이드 분류기 추가
if mouth_cascade.empty() or face_cascade.empty():
    raise IOError('입 또는 얼굴 검출을 위한 Haar cascade 분류기 XML 파일을 로드할 수 없습니다.')

is_mouth_detected = False
prev_mouth_rect = None

tablet_counter = 0
tablet_state = "not_tablet_detected"
previous_tablet_state = "not_tablet_detected"
no_detection_frames = 0

backSub = cv2.createBackgroundSubtractorMOG2()

# 감마 보정 함수
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break

    # # 감마 값을 조정하여 프레임의 밝기를 조절합니다.
    # gamma = 0.9  # 감마 값을 낮추면 프레임이 어두워집니다.
    # frame = adjust_gamma(frame, gamma=gamma)

    frame = imutils.resize(frame, width=1200)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    fgMask = backSub.apply(frame)
    fg = cv2.bitwise_and(frame, frame, mask=fgMask)

    # tablet_mask를 전경 이미지를 사용하여 생성
    fg_hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV)
    tablet_mask = cv2.inRange(fg_hsv, tablet_Lower, tablet_Upper)
    tablet_mask = cv2.erode(tablet_mask, None, iterations=2)
    tablet_mask = cv2.dilate(tablet_mask, None, iterations=2)

    tablet_cnts = cv2.findContours(tablet_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tablet_cnts = imutils.grab_contours(tablet_cnts)
    tablet_center = None

    if len(tablet_cnts) > 0:
        c = max(tablet_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        tablet_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # 반지름과 면적 조건 수정
        if radius > 2 and cv2.contourArea(c) < 600:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(frame, tablet_center, 5, (0, 0, 255), -1)
        no_detection_frames = 0
    else:
        no_detection_frames += 1

    tablet_pts.appendleft(tablet_center)

    for i in range(1, len(tablet_pts)):
        if tablet_pts[i - 1] is None or tablet_pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, tablet_pts[i - 1], tablet_pts[i], (0, 0, 255), thickness)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # face 영역을 이용해서 roi 설정

    for (fx, fy, fw, fh) in faces:
        # 얼굴 하단 1/3 부분만 관심 영역으로 설정
        roi_gray = gray[fy + 2*fh//3:fy + fh, fx:fx + fw]
        mouth_rects = mouth_cascade.detectMultiScale(roi_gray, 1.2, 15)

        if len(mouth_rects) > 0:
            for (mx, my, mw, mh) in mouth_rects:
                my += fy + 2*fh//3  # 전체 이미지 좌표로 변환
                cv2.rectangle(frame, (fx + mx, my), (fx + mx + mw, my + mh), (0, 255, 0), 3)
                prev_mouth_rect = (fx + mx, my, mw, mh)
                is_mouth_detected = True
                break
        else:
            if is_mouth_detected:
                x, y, w, h = prev_mouth_rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    if no_detection_frames > 30:
        tablet_state = "not_tablet_detected"
    else:
        if tablet_center is not None:
            if prev_mouth_rect is not None:
                x, y, w, h = prev_mouth_rect
                y = int(y - 0.15 * h)
                if x <= tablet_center[0] <= x + w and y <= tablet_center[1] <= y + h:
                    tablet_state = "tablet_detected_inside_mouth"
                else:
                    tablet_state = "tablet_detected_outside_mouth"
    if previous_tablet_state == "tablet_detected_inside_mouth" and tablet_state == "not_tablet_detected":
        tablet_counter += 1
    if no_detection_frames > 30:
        tablet_state = "not_tablet_detected"
    previous_tablet_state = tablet_state

    status_text = f"State: {tablet_state}"
    counter_text = f"Count: {tablet_counter}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, counter_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
if not args.get("video", False):
    vs.stop()
else:
    output = tablet_counter
    print(json.dumps(output))
    vs.release()
