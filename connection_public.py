from ultralytics import YOLO
import cv2
import requests
import time
import os

from datetime import datetime


CLASS_CONF = {
    "fire": 0.10,
    "smoke": 0.20,
    "cigarette_butt": 0.05
}

MODEL_PATH = "./best.engine"   # 현재 모델 경로
BOT_TOKEN = "--" #봇 토큰
CHAT_ID = "--" #chat ID

CONF_THRES = 0.01             # 일단 낮춰두고 코드로 제어
COOLDOWN = 10                 # 알림 간격, 초

TARGET_CLASSES = ["fire", "smoke", "cigarette_butt"]  # 네 모델 class 이름에 맞게 수정

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1 sync=false"
    )

#이하 : 텔레그램 함수

def send_telegram_message(text): #메세지 보내는 함수
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID, #받을 사람
        "text": text #보낼 메세지 내용
    }
    try:
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print("텔레그램 전송 실패:", e)
        return False


def send_telegram_photo(image_path, caption=""): #사진 전송 함수, no caption
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    try:

        with open(image_path, "rb") as photo: #보낼 이미지 경로(image_path)
            files = {"photo": photo}
            data = {
                "chat_id": CHAT_ID, #보낼 사람
                "caption": caption #사진 밑 설명글
            }

            response = requests.post(url, files=files, data=data, timeout=20)
            #서버 응답 20초까지 기다려봄

            response.raise_for_status()

        return True
    
    except(requests.RequestException,OSError)as e:
        print("사진 전송 실패: ",e);
        return False
    


#YOLO 함수

model = YOLO(MODEL_PATH, task="detect") #yolo 모델 불러오기

cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER) #카메라 열기

if not cap.isOpened():
    print("카메라를 열지 못했습니다.")
    exit()
    
last_t= 0 #마지막으로 알림 보낸 시간 저장

while True: #실시간으로 카메라 확인
    ret, frame = cap.read() #현재 화면 한 장 읽고
    if not ret: #읽기 성공했나 확인
        print("카메라 프레임을 읽지 못했습니다.")
        break #못 읽으면 메세지 출력 후 반복문 종료

    results = model(frame, conf=CONF_THRES) #현재 카메라 화면을 모델에 넣어 탐지

    best_detections = {} #바운딩 박스가 여러개일 경우 가장 높은 신뢰도만 저장

    for result in results:
        for box in result.boxes: #바운딩 박스 하나씩 확인
            cls_id = int(box.cls[0]) #감지된 객체 클래스 번호 가져옴
            conf = float(box.conf[0]) #확률 가져옴
            class_name = model.names[cls_id] #번호와 실제 이름 매핑

            if class_name not in TARGET_CLASSES: 
                continue
            
            required_conf = CLASS_CONF.get(class_name, 0.25) #각 클래스에 필요한 최소 확률 가져옴

            if conf < required_conf: #최소 확률 이하면
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            #바운딩 박스 좌표

            if(
                class_name not in best_detections
                or conf>best_detections[class_name]
            ):
                best_detections[class_name]={
                    "conf": conf,
                    "box" : (x1,y1,x2,y2)
                }
            #각 클래스 별로 가장 신뢰도 높은 박스만 기록

    detected_classes = list(best_detections.keys())

    detected_names = [
        f"{class_name} {conf:.2f}"
        for class_name, conf in best_detections.items()
    ]

    detected = len(best_detections) > 0
                    
    #이미지에 바운딩 박스 표시
    annotated_frame = results[0].plot()

    now = time.time() #현재 시간 초단위로 가져오기
    if detected and now - last_t >= COOLDOWN: #마지막 알림 이후 10초가 지났는지 확인
        message_sent=False
        photo_sent=False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") #파일 이름에 현재 시간 기록
        image_path = f"alert_{timestamp}.jpg" #저장할 이미지 파일 이름 만들기

        image_saved=cv2.imwrite(image_path, annotated_frame) #이미지 jpg로 저장
        
        if image_saved:
            photo_sent = send_telegram_photo(
                image_path,
                caption="감지 이미지"
            )
        else:
            print("이미지 저장 실패:", image_path)
            
        if "fire" in detected_classes or "smoke" in detected_classes:
            text = "화재 의심 상황 감지!\n" + "\n".join(detected_names)
        elif "cigarette_butt" in detected_classes:
            text = "위험상황 감지!\n" + "\n".join(detected_names)
        else:
            text = "객체 감지!\n" + "\n".join(detected_names)

            #각 상황에 맞게 메세지 만들기

        message_sent = send_telegram_message(text) #메세지 보내고

        if message_sent or photo_sent:
            last_t = now 
            print("텔레그램 알림 전송 완료:", detected_names)
        else:
            print("텔레그램 알림 전송 실패")

        if photo_sent:
            try:
                os.remove(image_path)
                print("이미지 삭제 완료:", image_path)

            except OSError as e:
                print("이미지 삭제 실패:", e)
    

        #계속 사진이 쌓이면 저장공간이 부족해지니 사진 전송 후 삭제하도록 
    ##cv2.imshow("YOLO Fire Detection", annotated_frame) #PC화면에 실시간 감지 화면 띄우기

    #if cv2.waitKey(1) & 0xFF == ord("q"): #q를 누르면 반복문 종료
    #    break

cap.release() #카메라 사용 종료
##cv2.destroyAllWindows() #openCV 창 모두 종료
##주석처리된 건 PC 테스트용
