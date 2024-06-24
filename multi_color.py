import cv2
import numpy as np
import argparse
import imutils
from collections import deque
from imutils.video import VideoStream
import time
import json

# 여기서 흰색은 어느정도 감지 완료

# 인수 파서를 구성하고 인수를 구문 분석합니다.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="(선택 사항) 비디오 파일 경로")
ap.add_argument("-b", "--buffer", type=int, default=64, help="최대 버퍼 크기")
args = vars(ap.parse_args())
#args["video"] = "C:\\all\ict_tablet\data\\eg_3.mp4"

white_tablet_Lower = (0, 0, 200)
white_tablet_Upper = (180, 100, 255)  # 흰색

# yellow_tablet_Lower = (20, 40, 100)
# yellow_tablet_Upper = (50, 250, 250) #노란색

white_tablet_pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
if mouth_cascade.empty():
    raise IOError('입 검출을 위한 Haar cascade 분류기 XML 파일을 로드할 수 없습니다.')

is_mouth_detected = False
prev_mouth_rect = None

white_tablet_counter = 0
white_tablet_state = "not_white_tablet_detected"
previous_white_tablet_state = "not_white_tablet_detected"
no_detection_frames = 0

backSub = cv2.createBackgroundSubtractorMOG2()

frame_index = 0  # 프레임 인덱스


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

    # 홀수 프레임일 때만 처리
    if frame_index % 2 == 1:
        # # 감마 값을 조정하여 프레임의 밝기를 조절합니다.
        # gamma = 0.9  # 감마 값을 낮추면 프레임이 어두워집니다.
        # frame = adjust_gamma(frame, gamma=gamma)

        frame = imutils.resize(frame, width=1200)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        fgMask = backSub.apply(frame)
        fg = cv2.bitwise_and(frame, frame, mask=fgMask)

        # 커널 크기 조정
        kernel_size = (5, 5)  # 커널 크기를 늘려 침식 강도 조절
        kernel = np.ones(kernel_size, np.uint8)
        # 전경 마스크를 축소하여 테두리 부분 제거
        # 여기서 커널크기랑 이터레이션 횟수로 테두리 깎아내기 패러미터 조절
        fgMask_eroded = cv2.erode(fgMask, kernel, iterations=2)
        fg_eroded = cv2.bitwise_and(frame, frame, mask=fgMask_eroded)

        # tablet_mask를 축소된 전경 이미지를 사용하여 생성
        fg_hsv = cv2.cvtColor(fg_eroded, cv2.COLOR_BGR2HSV)
        white_tablet_mask = cv2.inRange(fg_hsv, white_tablet_Lower, white_tablet_Upper)
        white_tablet_mask = cv2.erode(white_tablet_mask, None, iterations=2)
        white_tablet_mask = cv2.dilate(white_tablet_mask, None, iterations=2)

        white_tablet_cnts = cv2.findContours(white_tablet_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_tablet_cnts = imutils.grab_contours(white_tablet_cnts)
        white_tablet_center = None

        if len(white_tablet_cnts) > 0:
            c = max(white_tablet_cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            white_tablet_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # 반지름과 면적 조건 수정
            if radius > 2 and cv2.contourArea(c) < 600:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                cv2.circle(frame, white_tablet_center, 5, (0, 0, 255), -1)
            no_detection_frames = 0
        else:
            no_detection_frames += 1

        white_tablet_pts.appendleft(white_tablet_center)

        for i in range(1, len(white_tablet_pts)):
            if white_tablet_pts[i - 1] is None or white_tablet_pts[i] is None:
                continue
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, white_tablet_pts[i - 1], white_tablet_pts[i], (0, 0, 255), thickness)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)

        if len(mouth_rects) > 0:
            is_mouth_detected = True
            prev_mouth_rect = mouth_rects[0]
        else:
            if is_mouth_detected:
                x, y, w, h = prev_mouth_rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            else:
                pass

        for (x, y, w, h) in mouth_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            break

        if no_detection_frames > 10:
            white_tablet_state = "not_white_tablet_detected"
        else:
            if white_tablet_center is not None:
                if prev_mouth_rect is not None:
                    x, y, w, h = prev_mouth_rect
                    y = int(y - 0.15 * h)
                    if x <= white_tablet_center[0] <= x + w and y <= white_tablet_center[1] <= y + h:
                        white_tablet_state = "white_tablet_detected_inside_mouth"
                    else:
                        white_tablet_state = "white_tablet_detected_outside_mouth"
        if previous_white_tablet_state == "white_tablet_detected_inside_mouth" and white_tablet_state == "not_white_tablet_detected":
            white_tablet_counter += 1
        if no_detection_frames > 30:
            white_tablet_state = "not_white_tablet_detected"
        previous_white_tablet_state = white_tablet_state

        status_text = f"State: {white_tablet_state}"
        counter_text = f"Count: {white_tablet_counter}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, counter_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    frame_index += 1

cv2.destroyAllWindows()


if not args.get("video", False):
    vs.stop()
else:
    output = white_tablet_counter
    print(json.dumps(output))
    vs.release()
