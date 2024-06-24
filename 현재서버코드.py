import argparse
import time
import json
import cv2
import imutils
import numpy as np
from collections import deque
from imutils.video import VideoStream

# 시작 시간 기록
start_time = time.time()

# 인수 파서 설정
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="(선택 사항) 비디오 파일 경로")
ap.add_argument("-b", "--buffer", type=int, default=64, help="최대 버퍼 크기")
args = vars(ap.parse_args())
args["video"] = "C:\\all\imgproc_project\ict_tablet\data\\eg_3.mp4"

# HSV 색 공간에서 상한 경계 정의 (노란색)
tablet_Lower = (20, 40, 100)
tablet_Upper = (50, 250, 250)

# 각 색상에 대해 별도의 점 큐 설정
tablet_pts = deque(maxlen=args["buffer"])

# 비디오 스트림 초기화
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

# # 카메라 또는 비디오 파일 예열 시간
# time.sleep(2.0)

# 입 검출을 위한 Haar cascade 로드
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
if mouth_cascade.empty():
    raise IOError('입 검출을 위한 Haar cascade 분류기 XML 파일을 로드할 수 없습니다.')

is_mouth_detected = False
prev_mouth_rect = None

tablet_counter = 0  # 입의 영역 내에 center가 들어왔다가 사라진 횟수를 세는 변수
tablet_state = "not_tablet_detected"  # 물체의 현재 상태
previous_tablet_state = "not_tablet_detected"  # 이전 상태를 추적
no_detection_frames = 0  # 물체가 감지되지 않은 프레임 수

frame_index = 0  # 프레임 인덱스



# 반복문 실행
while True:
    # 현재 프레임 가져오기
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # 비디오 끝에 도달했는지 확인
    if frame is None:
        break

    # 홀수 프레임일 때만 처리
    if frame_index % 2 == 1:
        # 프레임 크기 조정 및 블러링 후 HSV 색 공간으로 변환
        frame = imutils.resize(frame, width=1200)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 노란색 마스크 생성 및 노이즈 제거
        tablet_mask = cv2.inRange(hsv, tablet_Lower, tablet_Upper)
        tablet_mask = cv2.erode(tablet_mask, None, iterations=2)
        tablet_mask = cv2.dilate(tablet_mask, None, iterations=2)

        # 윤곽 찾기
        tablet_cnts = cv2.findContours(tablet_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tablet_cnts = imutils.grab_contours(tablet_cnts)
        tablet_center = None

        # 윤곽이 발견된 경우
        if len(tablet_cnts) > 0:
            c = max(tablet_cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            tablet_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            no_detection_frames = 0
        else:
            no_detection_frames += 1

        # 큐 업데이트
        tablet_pts.appendleft(tablet_center)

        # 입 검출을 위해 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)

        if len(mouth_rects) > 0:
            is_mouth_detected = True
            prev_mouth_rect = mouth_rects[0]
        else:
            if is_mouth_detected and prev_mouth_rect is not None:
                x, y, w, h = prev_mouth_rect

        # 상태 업데이트
        if no_detection_frames > 30:
            tablet_state = "not_tablet_detected"
        else:
            if tablet_center is not None and prev_mouth_rect is not None:
                x, y, w, h = prev_mouth_rect
                y = int(y - 0.15 * h)
                if x <= tablet_center[0] <= x + w and y <= tablet_center[1] <= y + h:
                    tablet_state = "tablet_detected_inside_mouth"
                else:
                    tablet_state = "tablet_detected_outside_mouth"

        # 상태 전환 시 카운터 증가
        if previous_tablet_state == "tablet_detected_inside_mouth" and tablet_state == "not_tablet_detected":
            tablet_counter += 1

        previous_tablet_state = tablet_state

        # 프레임 출력 (옵션)
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     break

    frame_index += 1

# # 모든 창 닫기
# cv2.destroyAllWindows()

# 종료 시간 기록
end_time = time.time()

# 걸린 시간 계산
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# 비디오 스트림 중지
if not args.get("video", False):
    vs.stop()
else:
    output = tablet_counter
    print(json.dumps(output))
    vs.release()
