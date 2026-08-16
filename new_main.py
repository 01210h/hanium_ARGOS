from ultralytics import YOLO
import cv2
import requests
import time
import os
import serial
import threading
import re
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import deque

#YOLO TensorRT 모델
MODEL_PATH = "./best.engine"

#MLP 모델
MLP_MODEL_PATH = "./fire_mlp.pkl"

#Telegram 봇
BOT_TOKEN = "--"
CHAT_ID = "--"

SAVE_DIR = Path("./fire_alerts")

#Arduino
ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUDRATE = 115200

#YOLO 설정
CONF_THRES = 0.01

#화면 박스 표시용
DISPLAY_CONF = 0.10

#YOLO 클래스
TARGET_CLASSES = ["fire", "smoke", "cigarette_butt", "spark",]

#MLP 설정
#순서 중요
FEATURE_COLUMNS = [
    "fire_conf",
    "smoke_conf",
    "cigarette_conf",
    "spark_conf",
    "temperature",
    "gas",
    "temp_change",
    "gas_change"
]

sensor_history = deque()
CHANGE_WINDOW_SECONDS = 2.0

#위험 판단 기준
FIRE_PROB_THRESHOLD = 0.70

#3 프레임 연속 시 위험상황으로 인식 
ALERT_CONFIRM_FRAMES = 3

#텔레그램 재전송 제한(60s)
ALERT_COOLDOWN_SECONDS = 60

#센서 데이터 파싱
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

#센서 공유 데이터
sensor_lock = threading.Lock()

sensor_data = {
    "gas_raw": 0,
    "ir_temperature": 0.0,
    "last_update": 0.0,
}

#Arduino 데이터 한 줄 해석
def parse_sensor_line(line):
    csv_match = CSV_PATTERN.fullmatch(line)

    if csv_match:
        gas_raw = int(
            float(csv_match.group(1))
        )

        ir_temperature = float(
            csv_match.group(2)
        )
        return gas_raw, ir_temperature

    gas_raw = None
    ir_temperature = None

    gas_match = GAS_PATTERN.search(line)

    if gas_match:
        gas_raw = int(float(gas_match.group(1)))

    ir_match = IR_PATTERN.search(line)

    if ir_match:
        ir_temperature = float(
            ir_match.group(1)
        )
    return gas_raw, ir_temperature

#Arduino 수신 Thread
def read_arduino(stop_event):
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
            )as arduino:
                time.sleep(2)
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

                    gas_value, ir_value = (parse_sensor_line(line))
                    current_time = time.monotonic()

                    #가스
                    if gas_value is not None:
                        if 0 <= gas_value <= 16383:
                            pending_gas = gas_value
                            pending_gas_time = current_time
                        else:
                            print("[Arduino] 비정상 가스값:", gas_value,)

                    #온도
                    if ir_value is not None:
                        if -40.0 <= ir_value <= 300.0:
                            pending_ir = ir_value
                            pending_ir_time = current_time
                        else:
                            print("[Arduino] 비정상 온도값:",ir_value,)

                    #둘 중 하나라도 없으면 기다림
                    if (
                        pending_gas is None
                        or pending_ir is None
                    ):
                        continue

                    #두 센서 값이 너무 오래 차이 나면 사용 X
                    sensor_time_difference = abs(
                        pending_gas_time
                        - pending_ir_time
                    )

                    if sensor_time_difference > 3.0:
                        continue

                    #공유 데이터 업데이트
                    with sensor_lock:
                        sensor_data["gas_raw"] = (
                            pending_gas
                        )
                        sensor_data["ir_temperature"] = (
                            pending_ir
                        )
                        sensor_data["last_update"] = (
                            current_time
                        )

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
            print("[Arduino] 예상하지 못한 오류:", repr(error),)
            if not stop_event.is_set():
                time.sleep(1)

#현재 센서값 가져오기
def get_sensor_data():
    with sensor_lock:
        gas_raw = sensor_data["gas_raw"]

        ir_temperature = (
            sensor_data["ir_temperature"]
        )
        last_update = (
            sensor_data["last_update"]
        )

    #3초 이상 새 데이터가 없으면 연결 끊김
    sensor_connected = (
        last_update > 0
        and
        time.monotonic() - last_update <= 3
    )

    if not sensor_connected:
        return 0, 0.0, False

    return (gas_raw, ir_temperature, True,)

#Jetson CSI Camera Pipeline
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
        f"nvarguscamerasrc "
        f"sensor-id={sensor_id} ! "

        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, "
        f"height=(int){capture_height}, "
        f"format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "

        f"nvvidconv "
        f"flip-method={flip_method} ! "

        f"video/x-raw, "
        f"width=(int){display_width}, "
        f"height=(int){display_height}, "
        f"format=(string)BGRx ! "

        f"videoconvert ! "

        f"video/x-raw, "
        f"format=(string)BGR ! "

        f"appsink drop=1 sync=false"
    )

#Telegram 메시지
def send_telegram_message(text):
    url = (
        f"https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )
    data = {"chat_id": CHAT_ID, "text": text,}

    try:
        response = requests.post(url, data=data, timeout=10)
        if not response.ok:
            print("상태 코드:", response.status_code)
            print("텔레그램 응답:", response.text)
            return False
        return True

    except requests.RequestException as error:
        print("텔레그램 연결 오류:", error)
        return False

#Telegram 사진
def send_telegram_photo(image_path,caption="",):
    url = (
        f"https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendPhoto"
    )
    try:
        with open(image_path, "rb",) as photo:
            response = requests.post(
                url,
                files={
                    "photo": photo
                },
                data={
                    "chat_id": CHAT_ID,
                    "caption": caption,
                },
                timeout=20,
            )

        if not response.ok:
            print("사진 상태 코드:", response.status_code,)
            print("사진 텔레그램 응답:", response.text,)
            return False
        return True
    except (
        requests.RequestException, OSError,
    ) as error:
        print("사진 전송 오류:", error,)
        return False

#YOLO Class 이름 얻기
def get_class_name(model, class_id,):
    if isinstance(model.names, dict,):
        return str(model.names[class_id])

    return str(model.names[class_id])

#폴더 생성
SAVE_DIR.mkdir(parents=True, exist_ok=True,)

#YOLO 모델 Load
print("[YOLO] 모델 로딩 중...")
model = YOLO(MODEL_PATH, task="detect",)

print("[YOLO] 모델 로딩 완료")
print("[YOLO] classes:", model.names,)

#MLP 모델 Load
print("[MLP] 모델 로딩 중...")

fire_mlp = joblib.load(
    MLP_MODEL_PATH
)

print("[MLP] 모델 로딩 완료")

#Arduino Thread 시작
stop_event = threading.Event()

arduino_thread = threading.Thread(
    target=read_arduino,
    args=(stop_event,),
    daemon=True,
)

arduino_thread.start()

#Camera 시작
cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    stop_event.set()
    raise RuntimeError("카메라를 열지 못했습니다.")

#Alert 상태값
last_alert_time = 0
alert_frame_count = 0

#Main Loop
try:
    while True:
        #Camera
        ret, frame = cap.read()

        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            break

        #YOLO
        results = model(
            frame,
            conf=CONF_THRES,
            verbose=False,
        )

        # 클래스별 최고 confidence for MLP
        yolo_confs = {
            "fire": 0.0,
            "smoke": 0.0,
            "cigarette_butt": 0.0,
            "spark": 0.0,
        }

        #화면 표시용 detection
        best_detections = {}

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                class_name = get_class_name(model,cls_id,)

                #필요 없는 클래스는 무시
                if class_name not in TARGET_CLASSES:
                    continue

                # MLP용 confidence
                # 0.01 이상 YOLO 결과 그대로 사용
                yolo_confs[class_name] = max(yolo_confs[class_name], confidence,)

                #화면 표시용
                if confidence < DISPLAY_CONF:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist(),)

                if (
                    class_name
                    not in best_detections
                    or
                    confidence > best_detections[class_name]["conf"]
                ):
                    best_detections[class_name] = {
                        "conf":
                            confidence,
                        "box":
                            (x1, y1, x2, y2,),
                    }

        #Arduino sensor
        (gas_raw, ir_temperature, sensor_connected,) = get_sensor_data()

        # MLP
        fire_probability = 0.0
        temp_change = 0.0
        gas_change = 0.0

        if sensor_connected:
            current_time = time.monotonic()
            sensor_history.append(
                (
                    current_time,
                    ir_temperature,
                    gas_raw,
                )
            )

            while (
                sensor_history
                and current_time - sensor_history[0][0] > CHANGE_WINDOW_SECONDS
            ):
                sensor_history.popleft()

            if len(sensor_history) >= 2:
                old_time, old_temp, old_gas = sensor_history[0]
                temp_change = (ir_temperature - old_temp)
                gas_change = (gas_raw - old_gas)

        else:
            sensor_history.clear()

        if sensor_connected:
            # 학습할 때 사용한 Feature 이름과
            # 순서를 반드시 동일하게 맞춰야 함

            mlp_input = pd.DataFrame(
                [[
                    yolo_confs["fire"],
                    yolo_confs["smoke"],
                    yolo_confs["cigarette_butt"],
                    yolo_confs["spark"],
                    ir_temperature,
                    gas_raw,
                    temp_change,
                    gas_change,
                ]],

                columns=FEATURE_COLUMNS,
            )

            #최종 위험상황 판단 확률
            probabilities = (
                fire_mlp.predict_proba(mlp_input)[0]
            )

            # class 1 = 위험상황
            fire_class_index = list(fire_mlp.classes_).index(1)
            fire_probability = float(probabilities[fire_class_index])

        #터미널 출력
        print(
            f"Fire={yolo_confs['fire']:.3f} | "
            f"Smoke={yolo_confs['smoke']:.3f} | "
            f"Cigarette="
            f"{yolo_confs['cigarette_butt']:.3f} | "
            f"Spark={yolo_confs['spark']:.3f} | "
            f"Temp={ir_temperature:.1f} | "
            f"Gas={gas_raw} | "
            f"dTemp={temp_change:+.1f} | "
            f"dGas={gas_change:+.0f} | "
            f"MLP={fire_probability:.3f}"
        )

        #화면 만들기
        annotated_frame = frame.copy()

        #YOLO Bounding Box
        for (class_name, data,) in best_detections.items():
            confidence = data["conf"]
            x1, y1, x2, y2 = (data["box"])

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2,)

            label = (f"{class_name} " f"{confidence:.2f}")

            cv2.putText(
                annotated_frame,
                label,
                (x1,max(25, y1 - 10,),),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        #Sensor Status
        sensor_status = (
            "OK"
            if sensor_connected
            else
            "DISCONNECTED"
        )

        #위험상황 확률 표시
        if sensor_connected:
            if (fire_probability>= FIRE_PROB_THRESHOLD):
                probability_color = (0, 0, 255,)
                probability_text = (
                    f"FIRE PROB: "
                    f"{fire_probability * 100:.1f}%"
                )

            else:
                probability_color = (0, 255,0,)
                probability_text = (
                    f"FIRE PROB: "
                    f"{fire_probability * 100:.1f}%"
                )

        else:
            probability_color = (0, 255, 255,)
            probability_text = ("FIRE PROB: SENSOR ERROR")

        cv2.putText(
            annotated_frame,
            probability_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            probability_color,
            2,
        )

        #YOLO conf 표시
        cv2.putText(
            annotated_frame,
            (
                f"Fire:{yolo_confs['fire']:.2f}  "
                f"Smoke:{yolo_confs['smoke']:.2f}  "
                f"Cig:{yolo_confs['cigarette_butt']:.2f}  "
                f"Spark:{yolo_confs['spark']:.2f}"
            ),
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        #센서 표시
        cv2.putText(
            annotated_frame,
            (
                f"Gas:{gas_raw}  "
                f"Temp:{ir_temperature:.1f}C  "
                f"Sensor:{sensor_status}"
            ),
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        #화재 판단
        if (sensor_connected
            and
            fire_probability>= FIRE_PROB_THRESHOLD
        ):
            alert_frame_count += 1
        else:
            alert_frame_count = 0

        #연속 프레임 표시
        cv2.putText(
            annotated_frame,
            (
                f"Confirm: "
                f"{alert_frame_count}"
                f"/{ALERT_CONFIRM_FRAMES}"
            ),
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        #Telegram Alert
        now = time.time()
        alert_confirmed = (alert_frame_count>= ALERT_CONFIRM_FRAMES)
        cooldown_finished = (now - last_alert_time>= ALERT_COOLDOWN_SECONDS)

        if (alert_confirmed and cooldown_finished):
            timestamp = (
                datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
            )

            image_path = (
                SAVE_DIR
                /
                f"alert_{timestamp}.jpg"
            )

            #사진 저장
            image_saved = cv2.imwrite(
                str(image_path),
                annotated_frame,
            )

            #Telegram 메시지
            text = (
                "🔥 화재 위험 상황 감지!\n\n"

                f"🔥 최종 위험상황 확률: "
                f"{fire_probability * 100:.1f}%\n"

                f"경보 기준: "
                f"{FIRE_PROB_THRESHOLD * 100:.0f}%\n\n"

                "📷 카메라 발견\n"

                f"불: "
                f"{yolo_confs['fire']:.3f}\n"

                f"연기: "
                f"{yolo_confs['smoke']:.3f}\n"

                f"담배: "
                f"{yolo_confs['cigarette_butt']:.3f}\n"

                f"스파크: "
                f"{yolo_confs['spark']:.3f}\n\n"

                "🌡 센서\n"

                f"온도: "
                f"{ir_temperature:.1f}°C\n"

                f"가스: "
                f"{gas_raw}"
            )

            message_sent = (
                send_telegram_message(
                    text
                )
            )

            #사진 전송
            photo_sent = False

            if image_saved:
                photo_sent = (
                    send_telegram_photo(
                        image_path,
                        caption=(
                            "🔥 화재 위험 감지 "
                            f"{fire_probability * 100:.1f}%"
                        ),
                    )
                )
            else:
                print("이미지 저장 실패:", image_path,)

            #알림 성공
            if (message_sent or photo_sent):
                last_alert_time = now
                print("텔레그램 알림 전송 완료:", f"{fire_probability * 100:.1f}%")

            else:
                print("텔레그램 알림 전송 실패")

            #전송 후 사진 삭제 (저장공간 관리)
            if photo_sent:
                try:
                    os.remove(image_path)
                    print("이미지 삭제 완료:", image_path,)

                except OSError as error:
                    print("이미지 삭제 실패:", error,)

            #다시 처음부터 연속 프레임 검사
            alert_frame_count = 0

        cv2.imshow("Fire Detection", annotated_frame,)

        if (cv2.waitKey(1)& 0xFF== ord("q")):
            break

except KeyboardInterrupt:
    print("\n프로그램을 종료합니다.")

finally:
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()