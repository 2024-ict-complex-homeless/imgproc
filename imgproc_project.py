
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# 칼만 필터 초기화
kf_orange = cv2.KalmanFilter(4, 2)  # 4개의 상태 변수, 2개의 측정 변수
kf_orange.transitionMatrix = np.array([[1, 0, 1, 0],
                                       [0, 1, 0, 1],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], dtype=np.float32)  # 상태 전이 행렬 (x, y, dx, dy)

# 측정 함수 설정
kf_orange.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], dtype=np.float32)  # 측정 행렬 (x, y)

# 프로세스 노이즈 공분산 설정
kf_orange.processNoiseCov = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], dtype=np.float32) * 1e-2

# 측정 노이즈 공분산 설정
kf_orange.measurementNoiseCov = np.array([[1, 0],
                                          [0, 1]], dtype=np.float32) * 1e-1

# 초기 상태 설정 (x, y, dx, dy)
kf_orange.statePost = np.array([[0],
                                [0],
                                [0],
                                [0]], dtype=np.float32)



# 초록색

# 칼만 필터 초기화
kf_green = cv2.KalmanFilter(4, 2)  # 4개의 상태 변수, 2개의 측정 변수
kf_green.transitionMatrix = np.array([[1, 0, 1, 0],
                                       [0, 1, 0, 1],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], dtype=np.float32)  # 상태 전이 행렬 (x, y, dx, dy)

# 측정 함수 설정
kf_green.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], dtype=np.float32)  # 측정 행렬 (x, y)

# 프로세스 노이즈 공분산 설정
kf_green.processNoiseCov = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], dtype=np.float32) * 1e-2 

# 측정 노이즈 공분산 설정
kf_green.measurementNoiseCov = np.array([[1, 0],
                                          [0, 1]], dtype=np.float32) * 1e-1 * 10

# 초기 상태 설정 (x, y, dx, dy)
kf_green.statePost = np.array([[0],
                                [0],
                                [0],
                                [0]], dtype=np.float32)


# 노란색

# 칼만 필터 초기화
kf_yellow = cv2.KalmanFilter(4, 2)  # 4개의 상태 변수, 2개의 측정 변수
kf_yellow.transitionMatrix = np.array([[1, 0, 1, 0],
                                       [0, 1, 0, 1],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], dtype=np.float32)  # 상태 전이 행렬 (x, y, dx, dy)

# 측정 함수 설정
kf_yellow.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], dtype=np.float32)  # 측정 행렬 (x, y)

# 프로세스 노이즈 공분산 설정

kf_yellow.processNoiseCov = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], dtype=np.float32) * 1e-2

# 측정 노이즈 공분산 설정
kf_yellow.measurementNoiseCov = np.array([[1, 0],
                                          [0, 1]], dtype=np.float32) * 1e-1 * 100

# 초기 상태 설정 (x, y, dx, dy)
kf_yellow.statePost = np.array([[0],
                                [0],
                                [0],
                                [0]], dtype=np.float32)

# 핑크

# 칼만 필터 초기화
kf_pink = cv2.KalmanFilter(4, 2)  # 4개의 상태 변수, 2개의 측정 변수
kf_pink.transitionMatrix = np.array([[1, 0, 1, 0],
                                       [0, 1, 0, 1],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], dtype=np.float32)  # 상태 전이 행렬 (x, y, dx, dy)

# 측정 함수 설정
kf_pink.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], dtype=np.float32)  # 측정 행렬 (x, y)

# 프로세스 노이즈 공분산 설정
kf_pink.processNoiseCov = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]], dtype=np.float32) * 1e-2 

# 측정 노이즈 공분산 설정
kf_pink.measurementNoiseCov = np.array([[1, 0],
                                          [0, 1]], dtype=np.float32) * 1e-1 * 10

# 초기 상태 설정 (x, y, dx, dy)
kf_pink.statePost = np.array([[0],
                                [0],
                                [0],
                                [0]], dtype=np.float32)




# 인수 파서를 구성하고 인수를 구문 분석합니다.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="(선택 사항) 비디오 파일 경로")
ap.add_argument("-b", "--buffer", type=int, default=64, help="최대 버퍼 크기")
args = vars(ap.parse_args())

# HSV 색 공간에서 상한 경계를 정의합니다.
orange_Lower = (20, 90, 180)
orange_Upper = (25, 150, 255)  # 주황색

green_Lower = (70, 50, 50)
green_Upper = (90, 100, 175)  # 초록색


yellow_Lower = (30, 40, 100)
yellow_Upper = (40, 250, 250) #노란색멘토스

pink_Lower = (10, 50, 50)
pink_Upper = (30, 150, 150) #진한 핑크색

# pink_Lower = (150, 100, 100)
# pink_Upper = (165, 255, 255) #진한 핑크색

# 각 색상에 대해 별도의 점 큐를 설정합니다.
orange_pts = deque(maxlen=args["buffer"])
green_pts = deque(maxlen=args["buffer"])
yellow_pts = deque(maxlen=args["buffer"])
pink_pts = deque(maxlen=args["buffer"])

# 비디오 경로가 제공되지 않은 경우 웹캠 참조를 가져옵니다.
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

# 카메라 또는 비디오 파일이 예열되도록 허용합니다.
time.sleep(2.0)

# 입 검출을 위한 Haar cascade 로드
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
if mouth_cascade.empty():
    raise IOError('입 검출을 위한 Haar cascade 분류기 XML 파일을 로드할 수 없습니다.')

is_mouth_detected = False
prev_mouth_rect = None


orange_counter = 0  # 입의 영역 내에 center가 들어왔다가 사라진 횟수를 세는 변수
green_counter = 0 
yellow_counter = 0
pink_counter = 0

orange_state = "not_orange_detected"  # 물체의 현재 상태
previous_orange_state = "not_orange_detected"  # 이전 상태를 추적

green_state = "not_green_detected"  # 물체의 현재 상태
previous_green_state = "not_green_detected"  # 이전 상태를 추적

yellow_state = "not_yellow_detected"  # 물체의 현재 상태
previous_yellow_state = "not_yellow_detected"  # 이전 상태를 추적

pink_state = "not_pink_detected"  # 물체의 현재 상태
previous_pink_state = "not_pink_detected"  # 이전 상태를 추적


no_detection_frames = 0  # 물체가 감지되지 않은 프레임 수

# 반복문을 계속 실행합니다.
while True:
    # 현재 프레임을 가져옵니다.
    frame = vs.read()

    # VideoCapture 또는 VideoStream에서 프레임을 처리합니다.
    frame = frame[1] if args.get("video", False) else frame

    # 비디오를 보고 있고 프레임을 가져오지 않은 경우 비디오의 끝에 도달했습니다.
    if frame is None:
        break

    # 프레임 크기를 조정하고 블러를 적용한 후 HSV 색 공간으로 변환합니다.
    frame = imutils.resize(frame, width=1200)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 노란색과 핑크색 각각에 대한 마스크를 구성합니다.
    orange_mask = cv2.inRange(hsv, orange_Lower, orange_Upper)
    green_mask = cv2.inRange(hsv, green_Lower, green_Upper)
    yellow_mask = cv2.inRange(hsv, yellow_Lower, yellow_Upper)
    pink_mask = cv2.inRange(hsv, pink_Lower, pink_Upper)

    # 각 마스크에 대해 침식과 팽창을 수행하여 잡음을 제거합니다.
    orange_mask = cv2.erode(orange_mask, None, iterations=2)
    orange_mask = cv2.dilate(orange_mask, None, iterations=2)
    green_mask = cv2.erode(green_mask, None, iterations=2)
    green_mask = cv2.dilate(green_mask, None, iterations=2)
    yellow_mask = cv2.erode(yellow_mask, None, iterations=2)
    yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)
    pink_mask = cv2.erode(pink_mask, None, iterations=2)
    pink_mask = cv2.dilate(pink_mask, None, iterations=2)

    # 오렌지 마스크에서 윤곽을 찾습니다.
    oragne_cnts = cv2.findContours(orange_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    oragne_cnts = imutils.grab_contours(oragne_cnts)
    orange_center = None

    # 초록색 마스크에서 윤곽을 찾습니다.
    green_cnts = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_cnts = imutils.grab_contours(green_cnts)
    green_center = None

    # 노란색 마스크에서 윤곽을 찾습니다.
    yellow_cnts = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_cnts = imutils.grab_contours(yellow_cnts)
    yellow_center = None

    # 핑크 마스크에서 윤곽을 찾습니다.
    pink_cnts = cv2.findContours(pink_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pink_cnts = imutils.grab_contours(pink_cnts)
    pink_center = None

    # 주황색 윤곽이 발견된 경우
    if len(oragne_cnts) > 0:
        # 마스크에서 가장 큰 윤곽을 찾아 최소 외접원을 계산하고 중심을 찾는 데 사용합니다.
        # 면적이라는 key를 가지고 최대 컨투어를 리턴
        c = max(oragne_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        #M은 모먼트 값에 대한 딕셔너리
        M = cv2.moments(c)
        # m10은 x좌표의 합 / m00은 전체 갯수에 대한 모먼트
        orange_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # 칼만 필터 업데이트
        if orange_center is not None:
            kf_orange.correct(np.array([[orange_center[0]], [orange_center[1]]], dtype=np.float32))
            prediction = kf_orange.predict()

            orange_center = (int(prediction[0]), int(prediction[1]))

        # 반지름이 최소 크기를 충족하는 경우에만 진행합니다.
        if radius > 2:
            # 프레임에 원과 중심을 그리고 추적된 점 목록을 업데이트합니다.
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 167, 255), 2)
            cv2.circle(frame, orange_center, 5, (0, 167, 255), -1)
        no_detection_frames = 0
    else:
        no_detection_frames += 1

    # 초록색 윤곽이 발견된 경우
    if len(green_cnts) > 0:
        # 마스크에서 가장 큰 윤곽을 찾아 최소 외접원을 계산하고 중심을 찾는 데 사용합니다.
        c = max(green_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        
        # 칼만 필터 업데이트
        if green_center is not None:
            kf_green.correct(np.array([[green_center[0]], [green_center[1]]], dtype=np.float32))
            prediction = kf_green.predict()

            green_center = (int(prediction[0]), int(prediction[1]))


        # 반지름이 최소 크기를 충족하는 경우에만 진행합니다.
        if radius > 2:
            # 프레임에 원과 중심을 그리고 추적된 점 목록을 업데이트합니다.
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, green_center, 5, (0, 255, 0), -1)
        no_detection_frames = 0
    else:
        no_detection_frames += 1

    # 옐로우 윤곽이 발견된 경우
    if len(yellow_cnts) > 0:
        # 마스크에서 가장 큰 윤곽을 찾아 최소 외접원을 계산하고 중심을 찾는 데 사용합니다.
        c = max(yellow_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        yellow_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # 칼만 필터 업데이트
        if yellow_center is not None:
            kf_yellow.correct(np.array([[yellow_center[0]], [yellow_center[1]]], dtype=np.float32))
            prediction = kf_yellow.predict()

            yellow_center = (int(prediction[0]), int(prediction[1]))


        # 반지름이 최소 크기를 충족하는 경우에만 진행합니다.
        if radius > 2:
            # 프레임에 원과 중심을 그리고 추적된 점 목록을 업데이트합니다.
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, yellow_center, 5, (0, 255, 255), -1)
        no_detection_frames = 0
    else:
        no_detection_frames += 1

    # 핑크 윤곽이 발견된 경우
    if len(pink_cnts) > 0:
        # 마스크에서 가장 큰 윤곽을 찾아 최소 외접원을 계산하고 중심을 찾는 데 사용합니다.
        c = max(pink_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        pink_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # 칼만 필터 업데이트
        if pink_center is not None:
            kf_pink.correct(np.array([[pink_center[0]], [pink_center[1]]], dtype=np.float32))
            prediction = kf_pink.predict()

            pink_center = (int(prediction[0]), int(prediction[1]))

        # 반지름이 최소 크기를 충족하는 경우에만 진행합니다.
        if radius > 2:
            # 프레임에 원과 중심을 그리고 추적된 점 목록을 업데이트합니다.
            #cv2.circle(frame, (int(x), int(y)), int(radius), (203, 192, 255), 2)
            cv2.circle(frame, pink_center, 5, (203, 192, 255), -1)
        no_detection_frames = 0
    else:
        no_detection_frames += 1

    # 큐를 업데이트합니다.
    orange_pts.appendleft(orange_center)
    green_pts.appendleft(green_center)
    yellow_pts.appendleft(yellow_center)
    pink_pts.appendleft(pink_center)

    # 입 검출을 위해 프레임을 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 프레임에서 입을 검출합니다.
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)

    if len(mouth_rects) > 0:
        is_mouth_detected = True
        prev_mouth_rect = mouth_rects[0]
    else:
        if is_mouth_detected:
            # 현재 프레임에서 입이 검출되지 않은 경우 이전 프레임에서의 rect를 그림
            x, y, w, h = prev_mouth_rect
            #y = int(y - 0.15 * h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            pass

    # 검출된 입에 대해 사각형 그리기
    for (x, y, w, h) in mouth_rects:
        #y = int(y - 0.15 * h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        break

    # 상태 업데이트
    if no_detection_frames > 30:
        orange_state = "not_orange_detected"
    else:
        if orange_center is not None:
            if prev_mouth_rect is not None:
                x, y, w, h = prev_mouth_rect
                y = int(y - 0.15 * h)
                if x <= orange_center[0] <= x + w and y <= orange_center[1] <= y + h:
                    orange_state = "orange_detected_inside_mouth"
                else:
                    orange_state = "orange_detected_outside_mouth"
    # 상태 전환 시 카운터 증가
    if previous_orange_state == "orange_detected_inside_mouth" and orange_state == "not_orange_detected":
        orange_counter += 1
    if no_detection_frames > 30:
        orange_state = "not_orange_detected"
    
    # 이전 상태 업데이트
    previous_orange_state = orange_state

    if no_detection_frames > 30:
        green_state = "not_green_detected"
    else:
        if green_center is not None:
            if prev_mouth_rect is not None:
                x, y, w, h = prev_mouth_rect
                y = int(y - 0.15 * h)
                if x <= green_center[0] <= x + w and y <= green_center[1] <= y + h:
                    green_state = "green_detected_inside_mouth"
                else:
                    green_state = "green_detected_outside_mouth"
    # 상태 전환 시 카운터 증가
    if previous_green_state == "green_detected_inside_mouth" and green_state == "not_green_detected":
        green_counter += 1
    if no_detection_frames > 30:
        green_state = "not_green_detected"
    
    # 이전 상태 업데이트
    previous_green_state = green_state
    
    # 옐로우

    if no_detection_frames > 30:
        yellow_state = "not_yellow_detected"
    else:
        if yellow_center is not None:
            if prev_mouth_rect is not None:
                x, y, w, h = prev_mouth_rect
                y = int(y - 0.15 * h)
                if x <= yellow_center[0] <= x + w and y <= yellow_center[1] <= y + h:
                    yellow_state = "yellow_detected_inside_mouth"
                else:
                    yellow_state = "yellow_detected_outside_mouth"
    # 상태 전환 시 카운터 증가
    if previous_yellow_state == "yellow_detected_inside_mouth" and yellow_state == "not_yellow_detected":
        yellow_counter += 1
    if no_detection_frames > 30:
        yellow_state = "not_yellow_detected"
    
    # 이전 상태 업데이트
    previous_yellow_state = yellow_state

    # 핑크
    if no_detection_frames > 30:
        pink_state = "not_pink_detected"
    else:
        if pink_center is not None:
            if prev_mouth_rect is not None:
                x, y, w, h = prev_mouth_rect
                y = int(y - 0.15 * h)
                if x <= pink_center[0] <= x + w and y <= pink_center[1] <= y + h:
                    pink_state = "pink_detected_inside_mouth"
                else:
                    pink_state = "pink_detected_outside_mouth"
    # 상태 전환 시 카운터 증가
    if previous_pink_state == "pink_detected_inside_mouth" and pink_state == "not_pink_detected":
        pink_counter += 1
    if no_detection_frames > 30:
        pink_state = "not_pink_detected"
    
    # 이전 상태 업데이트
    previous_pink_state = pink_state


    # # 상태와 카운터를 프레임에 표시
    # status_text = f"State: {orange_state, green_state, yellow_state, pink_state}"
    # counter_text = f"Count: {orange_counter, green_counter, yellow_counter, pink_counter}"
    # cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # cv2.putText(frame, counter_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


    # 프레임을 출력합니다.
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 'q' 키를 누르면 반복문을 중지합니다.
    if key == ord("q"):
        break

# 모든 창을 닫습니다.
cv2.destroyAllWindows()

# 비디오 스트림을 중지합니다.
if not args.get("video", False):
    vs.stop()
else:
    vs.release()
