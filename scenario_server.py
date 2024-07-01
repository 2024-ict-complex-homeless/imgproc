import cv2
import numpy as np
import argparse
import imutils
import json
import time

# 시작 시간 기록
start_time = time.time()

# 인수 파서를 구성하고 인수를 구문 분석합니다.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="(선택 사항) 비디오 파일 경로")
ap.add_argument("-wl", "--white_lower", type=str, default="0,0,200", help="흰색 색상 범위 Lower")
ap.add_argument("-wu", "--white_upper", type=str, default="180,100,255", help="흰색 색상 범위 Upper")
ap.add_argument("-yl", "--yellow_lower", type=str, default="30,40,100", help="노란색 색상 범위 Lower")
ap.add_argument("-yu", "--yellow_upper", type=str, default="50,255,255", help="노란색 색상 범위 Upper")
args = vars(ap.parse_args())

# 비디오 경로 설정
args["video"] = "C:\\all\ict_tablet\data\\z1.mp4"

# 색상 범위를 튜플로 변환
white_Lower = tuple(map(int, args["white_lower"].split(',')))
white_Upper = tuple(map(int, args["white_upper"].split(',')))

yellow_Lower = tuple(map(int, args["yellow_lower"].split(',')))
yellow_Upper = tuple(map(int, args["yellow_upper"].split(',')))

# 비디오 스트림 설정
vs = cv2.VideoCapture(args["video"])
#vs = cv2.VideoCapture(0)


# 입 검출을 위한 분류기 로드
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
if mouth_cascade.empty():
    raise IOError('입 검출을 위한 Haar cascade 분류기 XML 파일을 로드할 수 없습니다.')

backSub = cv2.createBackgroundSubtractorMOG2()
is_mouth_detected = False
prev_mouth_rect = None

# 태블릿 상태 정의
NOT_TABLET_DETECTED = 0
TABLET_DETECTED_OUTSIDE_MOUTH = 1
TABLET_DETECTED_INSIDE_MOUTH = 2

# 초기 변수 설정
white_counter = 0
white_state = NOT_TABLET_DETECTED
previous_white_state = NOT_TABLET_DETECTED
no_white_frames = 0

yellow_counter = 0
yellow_state = NOT_TABLET_DETECTED
previous_yellow_state = NOT_TABLET_DETECTED
no_yellow_frames = 0

frame_index = 0  # 프레임 인덱스 초기화

def find_tablet_center(frame, lower, upper):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    fgMask = backSub.apply(frame)
    fg = cv2.bitwise_and(frame, frame, mask=fgMask)

    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, np.uint8)
    fgMask_eroded = cv2.erode(fgMask, kernel, iterations=2)
    fg_eroded = cv2.bitwise_and(frame, frame, mask=fgMask_eroded)

    fg_hsv = cv2.cvtColor(fg_eroded, cv2.COLOR_BGR2HSV)
    tablet_mask = cv2.inRange(fg_hsv, lower, upper)
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

        # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
        # cv2.circle(frame, tablet_center, 5, (0, 0, 255), -1)

        return tablet_center, 0
    else:
        return None, 1

def detect_and_draw_mouth(frame, mouth_cascade, is_mouth_detected, prev_mouth_rect):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)

    if len(mouth_rects) > 0:
        is_mouth_detected = True
        prev_mouth_rect = mouth_rects[0]
    else:
        if is_mouth_detected:
            x, y, w, h = prev_mouth_rect
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # for (x, y, w, h) in mouth_rects:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    #     break

    return frame, is_mouth_detected, prev_mouth_rect

def update_tablet_state_and_counter(tablet_center, prev_mouth_rect, tablet_state, previous_tablet_state, tablet_counter, no_detection_frames):

    if no_detection_frames > 10:
        tablet_state = NOT_TABLET_DETECTED
    else:
        if tablet_center is not None:
            if prev_mouth_rect is not None:
                x, y, w, h = prev_mouth_rect
                y = int(y - 0.15 * h)
                if x <= tablet_center[0] <= x + w and y <= tablet_center[1] <= y + h:
                    tablet_state = TABLET_DETECTED_INSIDE_MOUTH
                else:
                    tablet_state = TABLET_DETECTED_OUTSIDE_MOUTH
    if previous_tablet_state == TABLET_DETECTED_INSIDE_MOUTH and tablet_state == NOT_TABLET_DETECTED:
        tablet_counter += 1
    if no_detection_frames > 30:
        tablet_state = NOT_TABLET_DETECTED

    return tablet_state, tablet_counter

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


while True:
    ret, frame = vs.read()
    if not ret:
        break

    if frame_index % 2 == 1 or frame_index % 2 == 0:
        frame = imutils.resize(frame, width=300)
        
        # # 감마 값을 조정하여 프레임의 밝기를 조절합니다.
        gamma = 0.4  # 감마 값을 낮추면 프레임이 어두워집니다.
        frame = adjust_gamma(frame, gamma=gamma)

        # 입영역 검출 먼저
        frame, is_mouth_detected, prev_mouth_rect = detect_and_draw_mouth(frame, mouth_cascade, is_mouth_detected, prev_mouth_rect)

        # # 흰색 함수돌리기
        # white_center, white_flag = find_tablet_center(frame, white_Lower, white_Upper)
        # if white_flag == 0:
        #     no_white_frames = 0
        # else:
        #     no_white_frames += white_flag

        # white_state, white_counter = update_tablet_state_and_counter(white_center, prev_mouth_rect, white_state, previous_white_state, white_counter, no_white_frames)
        # previous_white_state = white_state

        # 노란색 함수 돌리기
        yellow_center, yellow_flag = find_tablet_center(frame, yellow_Lower, yellow_Upper)
        if yellow_flag == 0:
            no_yellow_frames = 0
        else:
            no_yellow_frames += yellow_flag

        yellow_state, yellow_counter = update_tablet_state_and_counter(yellow_center, prev_mouth_rect, yellow_state, previous_yellow_state, yellow_counter, no_yellow_frames)
        previous_yellow_state = yellow_state

        # # 디버그용
        # status_text = f"State (White): {white_state}"
        # counter_text = f"Count (White): {white_counter}"

        # yellow_text = f"State (Yellow): {yellow_state}"
        # yellows_text = f"Count (Yellow): {yellow_counter}"

        # center_text = f"Center (White): {white_center}"
        # frame_text = f"Count (White): {no_white_frames}"

        # cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # cv2.putText(frame, counter_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # cv2.putText(frame, yellow_text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # cv2.putText(frame, yellows_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # cv2.putText(frame, center_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # cv2.putText(frame, frame_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     break

    frame_index += 1

#cv2.destroyAllWindows()

# 종료 시간 기록
end_time = time.time()

# 걸린 시간 계산
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# # 비디오 스트림 또는 비디오 파일을 해제합니다.
# vs.release()

# JSON 형식으로 결과를 출력합니다.
white_output = white_counter
yellow_output = yellow_counter

print(json.dumps(white_output))
print(json.dumps(yellow_output))
