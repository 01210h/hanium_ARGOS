from ultralytics import YOLO
import cv2
import requests
import time
import os
import serial
import threading
import re
from pathlib import Path
from datetime import datetime


CLASS_CONF = {
    "fire": 0.10,
    "smoke": 0.20,
    "cigarette_butt": 0.07
}

MODEL_PATH = "./best.engine"   # 현재 모델 경로
BOT_TOKEN = "--" #봇 토큰
CHAT_ID = "--" #chat ID
SAVE_DIR = Path("./fire_alerts")
ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUDRATE = 115200

CONF_THRES = 0.01             # 일단 낮춰두고 코드로 제어
COOLDOWN = 10                 # 알림 간격, 초
ALERT_THRESHOLD = 70
ALERT_CONFIRM_FRAMES = 3
ALERT_COOLDOWN_SECONDS = 60
TARGET_CLASSES = ["fire", "smoke", "cigarette_butt"]
CAMERA_SCORES = {
    "fire": 75,
    "smoke": 15,
    "cigarette_butt": 10,
}

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

CSV_PATTERN = re.compile(
    rf"^\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*$"
)

GAS_PATTERN = re.compile(
    rf"(?:GAS(?:_RAW)?|가스(?:값)?)"
    rf"\s*[:=]\s*({NUMBER_PATTERN})",
    re.IGNORECASE,
)

IR_PATTERN = re.compile(
    rf"(?:IR(?:_TEMP(?:ERATURE)?)?|"
    rf"TEMP(?:ERATURE)?|"
    rf"적외선(?:\s*온도)?|온도)"
    rf"\s*[:=]\s*({NUMBER_PATTERN})",
    re.IGNORECASE,
)


def parse_sensor_line(line):
    """
    지원 형식:

    15,28.7
    GAS_RAW=15,IR_TEMP=28.7
    GAS_RAW=15
    IR_TEMP=28.7
    가스=15
    온도=28.7
    """

    # 가장 단순한 CSV 형식: 15,28.7
    csv_match = CSV_PATTERN.fullmatch(line)

    if csv_match:
        gas_raw = int(float(csv_match.group(1)))
        ir_temperature = float(csv_match.group(2))

        return gas_raw, ir_temperature

    gas_raw = None
    ir_temperature = None

    gas_match = GAS_PATTERN.search(line)

    if gas_match:
        gas_raw = int(float(gas_match.group(1)))

    ir_match = IR_PATTERN.search(line)

    if ir_match:
        ir_temperature = float(ir_match.group(1))

    return gas_raw, ir_temperature


def read_arduino(stop_event):
    # 가스와 온도가 서로 다른 줄로 들어오는 경우를 위한 임시 저장값
    pending_gas = None
    pending_ir = None

    pending_gas_time = 0.0
    pending_ir_time = 0.0

    while not stop_event.is_set():
        try:
            print(f"[Arduino] {ARDUINO_PORT} 연결 중")

            with serial.Serial(
                port=ARDUINO_PORT,
                baudrate=ARDUINO_BAUDRATE,
                timeout=1,
            ) as arduino:

                # 포트를 열면 보드가 재시작할 수 있으므로 대기
                time.sleep(2)

                # 여기서 reset_input_buffer()는 사용하지 않음
                # 초기 센서 메시지가 삭제될 수 있기 때문
                print("[Arduino] 연결 완료")

                while not stop_event.is_set():
                    raw_data = arduino.readline()

                    if not raw_data:
                        continue

                    line = raw_data.decode(
                        "utf-8",
                        errors="ignore",
                    ).strip()

                    if not line:
                        continue

                    print(f"[Arduino RAW] {line!r}")

                    gas_value, ir_value = parse_sensor_line(line)

                    current_time = time.monotonic()

                    # 이번 줄에서 가스값을 찾은 경우
                    if gas_value is not None:
                        # 비정상적인 값 방지
                        if 0 <= gas_value <= 16383:
                            pending_gas = gas_value
                            pending_gas_time = current_time
                        else:
                            print(
                                "[Arduino] 비정상 가스값:",
                                gas_value,
                            )

                    # 이번 줄에서 적외선 온도를 찾은 경우
                    if ir_value is not None:
                        # 비정상적인 온도 방지
                        if -40.0 <= ir_value <= 300.0:
                            pending_ir = ir_value
                            pending_ir_time = current_time
                        else:
                            print(
                                "[Arduino] 비정상 온도값:",
                                ir_value,
                            )

                    # 아직 둘 중 하나라도 받은 적이 없으면 대기
                    if pending_gas is None or pending_ir is None:
                        continue

                    # 서로 다른 줄에서 받은 값이라면
                    # 두 값의 수신 시점이 3초 이내인지 확인
                    sensor_time_difference = abs(
                        pending_gas_time - pending_ir_time
                    )

                    if sensor_time_difference > 3.0:
                        continue

                    with sensor_lock:
                        sensor_data["gas_raw"] = pending_gas
                        sensor_data["ir_temperature"] = pending_ir
                        sensor_data["last_update"] = current_time

                    print(
                        "[Arduino 정상 수신] "
                        f"gas={pending_gas}, "
                        f"ir={pending_ir:.1f}"
                    )

        except serial.SerialException as error:
            print(f"[Arduino] 연결 실패: {error}")

            if not stop_event.is_set():
                time.sleep(2)

        except Exception as error:
            print(
                "[Arduino] 예상하지 못한 오류:",
                repr(error),
            )

            if not stop_event.is_set():
                time.sleep(1)
            
def get_sensor_data():
    with sensor_lock:
        gas_raw = sensor_data["gas_raw"]
        ir_temperature = sensor_data["ir_temperature"]
        last_update = sensor_data["last_update"]

    # 3초 이상 센서값이 들어오지 않으면 연결 끊김으로 판단
    sensor_connected = (
        last_update > 0
        and time.monotonic() - last_update <= 3
    )

    if not sensor_connected:
        return 0, 0.0, False

    return gas_raw, ir_temperature, True

def calculate_gas_score(gas_raw):

    gas_ratio = gas_raw / 1023.0
    gas_ratio = max(0.0, min(1.0, gas_ratio))

    gas_score = gas_ratio * 60

    return gas_score, gas_ratio


def calculate_ir_score(temperature):

    if temperature >= 60:
        return 30

    if temperature >= 20:
        return 20

    if temperature >= 10:
        return 10

    return 0


def calculate_camera_score(detected_classes):
    score = 0

    for class_name in detected_classes:
        score += CAMERA_SCORES.get(class_name, 0)

    return score


sensor_lock = threading.Lock()

sensor_data = {
    "gas_raw": 0,
    "ir_temperature": 0.0,
    "last_update": 0.0,
}


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
        response = requests.post(
            url,
            data=data,
            timeout=10,
        )

        if not response.ok:
            print("상태 코드:", response.status_code)
            print("텔레그램 응답:", response.text)
            return False

        return True

    except requests.RequestException as error:
        print("텔레그램 연결 오류:", error)
        return False

def send_telegram_photo(image_path, caption=""): #사진 전송 함수, no caption
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    try:
        with open(image_path, "rb") as photo:
            response = requests.post(
                url,
                files={"photo": photo},
                data={
                    "chat_id": CHAT_ID,
                    "caption": caption,
                },
                timeout=20,
            )

        if not response.ok:
            print("사진 상태 코드:", response.status_code)
            print("사진 텔레그램 응답:", response.text)
            return False

        return True

    except (requests.RequestException, OSError) as error:
        print("사진 전송 오류:", error)
        return False
    
def get_class_name(model, class_id):
    if isinstance(model.names, dict):
        return str(model.names[class_id])

    return str(model.names[class_id])


#YOLO 함수

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model = YOLO(
    MODEL_PATH,
    task="detect",
)

# 아두이노 센서 수신 스레드
# 종료를 위한 신호
stop_event = threading.Event()

arduino_thread = threading.Thread(
    target=read_arduino,
    #스레드 실행 함수 저장
    args=(stop_event,),
    #아두이노에 전달할 인자
    daemon=True,
    #메인 종료 시 스레드도 함께 종료됨
)

arduino_thread.start()
#아두이노 스레드 실행(read_arduino함수가 별도 작동)

cap = cv2.VideoCapture(
    gstreamer_pipeline(sensor_id=0),
    cv2.CAP_GSTREAMER,
)
#파이캠 실행

if not cap.isOpened():
    stop_event.set()
    raise RuntimeError("카메라를 열지 못했습니다.")
#카메라 에러 체크

last_t = 0
#마지막으로 텔레그램 알림 보낸 시간 저장

alert_frame_count = 0
#이상 프레임 갯수


try:
    while True:
        ret, frame = cap.read()
        #프레임 단위로 읽어오기
        
        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            break

        results = model(
            frame,
            conf=CONF_THRES,
            verbose=False,
        )
        #현재 프레임을 YOLO 모델에 넣음

        # 각 클래스에서 신뢰도가 높은 결과만 저장
        best_detections = {}

        for result in results:
        #결과 하나씩 확인
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                #결과 신뢰도 가져오기
                class_name = get_class_name(
                    model,
                    cls_id,
                )
                required_confidence = CLASS_CONF.get(
                    class_name,
                    0.25,
                )

                if confidence < required_confidence:
                    continue
                #낮은 신뢰도는 무시
                
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist(),
                )
                #바운딩 박스 좌표 받아오기

                if (
                    class_name not in best_detections #첫감지 그대로 저장
                    or confidence
                    > best_detections[class_name]["conf"] #또는 가장 높으면 덮어쓰기
                ):
                    best_detections[class_name] = {
                        "conf": confidence,
                        "box": (x1, y1, x2, y2),
                    }
                    
                    #각 클래스 최고 신뢰도만 저장
        
        #딕셔너리 키 갖고오기
        detected_classes = list(
            best_detections.keys()
        )
        #카메라 점수 계산
        camera_score = calculate_camera_score(
            detected_classes
        )

        detected_names = [
            f"{class_name} {data['conf']:.2f}"
            for class_name, data
            in best_detections.items()
        ]

        (
            gas_raw,
            ir_temperature,
            sensor_connected,
        ) = get_sensor_data()
        # 센서값 저장
        
        #점수계산
        if sensor_connected:
            gas_score, gas_ratio = calculate_gas_score(
                gas_raw
            )

            ir_score = calculate_ir_score(
                ir_temperature
            )
        else:
            gas_score = 0
            gas_ratio = 0
            ir_score = 0

        #최종 점수계산
        total_score = (
            camera_score
            + gas_score
            + ir_score
        )


        #바운딩 박스 표시
        annotated_frame = frame.copy()

        for class_name, data in best_detections.items():
            confidence = data["conf"]
            x1, y1, x2, y2 = data["box"]

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2,
            )

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )


        #센서 연결 상태 표시
        sensor_status = (
            "OK"
            if sensor_connected
            else "DISCONNECTED"
        )
        #총점 화면 표시
        cv2.putText(
            annotated_frame,
            f"TOTAL: {total_score:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        
        #센서별 점수 표시
        cv2.putText(
            annotated_frame,
            (
                f"Camera: {camera_score:.1f}  "
                f"Gas: {gas_score:.1f}  "
                f"IR: {ir_score:.1f}"
            ),
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        #센서 원시값 표시
        cv2.putText(
            annotated_frame,
            (
                f"Gas raw: {gas_raw}  "
                f"IR: {ir_temperature:.1f} C  "
                f"Sensor: {sensor_status}"
            ),
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )


        #이상 프레임 수 계산

        if total_score >= ALERT_THRESHOLD:
            alert_frame_count += 1
        else:
            alert_frame_count = 0

        now = time.time()

        alert_confirmed = (
            alert_frame_count
            >= ALERT_CONFIRM_FRAMES
        )

        #마지막 알림 이후 지난 시간 계산
        cooldown_finished = (
            now - last_t
            >= ALERT_COOLDOWN_SECONDS
        )

        #두 조건 모두 만족 시 알림 전송
        if alert_confirmed and cooldown_finished:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            image_path = (
                SAVE_DIR
                / f"alert_{timestamp}.jpg"
            )
            
            #이미지 저장
            image_saved = cv2.imwrite(
                str(image_path),
                annotated_frame,
            )

            if detected_names:
                detected_text = "\n".join(
                    detected_names
                )
            else:
                detected_text = "카메라 감지 없음"
            
            #텔레그램 메세지 생성
            text = (
                "🔥 화재 위험 상황 감지!\n\n"
                f"최종 점수: {total_score:.1f}점\n"
                f"경보 기준: {ALERT_THRESHOLD}점\n\n"
                f"카메라 점수: {camera_score:.1f}점\n"
                f"가스 점수: {gas_score:.1f}점\n"
                f"적외선 점수: {ir_score:.1f}점\n\n"
                f"가스 측정값: {gas_raw}\n"
                f"적외선 온도: "
                f"{ir_temperature:.1f}°C\n\n"
                f"카메라 감지:\n"
                f"{detected_text}"
            )

            message_sent = send_telegram_message(
                text
            )

            photo_sent = False

            if image_saved:
                photo_sent = send_telegram_photo(
                    image_path,
                    caption=(
                        f"화재 위험 감지 "
                        f"{total_score:.1f}점"
                    ),
                )
            else:
                print(
                    "이미지 저장 실패:",
                    image_path,
                )
                
            #알림 전송
            if message_sent or photo_sent:
                last_t = now

                print(
                    "텔레그램 알림 전송 완료:",
                    f"{total_score:.1f}점",
                )
            else:
                print("텔레그램 알림 전송 실패")
            
            #이미지 삭제
            if photo_sent:
                try:
                    os.remove(image_path)
                    print(
                        "이미지 삭제 완료:",
                        image_path,
                    )

                except OSError as error:
                    print(
                        "이미지 삭제 실패:",
                        error,
                    )

            alert_frame_count = 0


        cv2.imshow(
            "Fire Detection",
            annotated_frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


except KeyboardInterrupt:
    print("\n프로그램을 종료합니다.")

#프로그램 종료
finally:
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
