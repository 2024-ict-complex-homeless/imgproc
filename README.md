//0624

저거 white erode 저게 배경 지워서 
흰색 물체가 배경에 있더라도 감지 못하게하는 부분을 넣은거임

흰색 감지할때는 커널크기 10x10 이터레이터 4 로 설정했었는데

저 패러미터로 노란색은 감지를 못함 / 5x5에 이터레이터 2로 해야 원래대로 나옴

또 백그라운드 지우는 과정 넣으면 +5초 정도 10 > 15초

커널크기에 따른 연산속도 변화는 없음

#내일 할거 : 멀티쓰레드로 다른색깔도 추가하기 

// 0625


이거 다른색깔 추가했음 >> functuon.py에서 노란색만하면 9초 흰색도 넣으면 13초

멀티스레드 추가할려는데 지피티 4.0 맛보기가 끝남

multithred.py << 지피티로 멀티스레드 추가한건데 33초로 늘어남;;;
videocapture 랑 finalversion이랑 같은 파일 그냥 영상출력만 지운거 비디오스트림 >> 비디오캡쳐로 바꿈 큰 차이는 없음 13초
