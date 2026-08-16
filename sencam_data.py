from ultralytics import YOLO
import cv2
import time
import serial
import threading
import re
import csv
from pathlib import Path
from datetime import datetime
from collections import deque

"""
0  → 현재 상황을 정상(label=0)으로 지정
1  → 현재 상황을 화재(label=1)로 지정

S  → 데이터 저장 시작
S  → 다시 누르면 일시정지

X  → label 초기화 + 저장 정지
Q  → 프로그램 종료

프로그램 실행
      ↓
0 누르기
      ↓
S 누르기
      ↓
평상시 환경 촬영
      ↓
S 눌러 정지

1 누르기
    ↓
S 누르기
    ↓
화재 상황 데이터 수집
    ↓
S 누르기

"""

#YOLO TensorRT 모델
MODEL_PATH = "./best.engine"

#수집 데이터 저장 파일
CSV_PATH = Path("./fire_training_data.csv")

#Arduino
ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUDRATE = 115200


#YOLO 설정

#MLP에는 가능한 한 원래 confidence를 전달
CONF_THRES = 0.01

#화면 Bounding Box 표시용
#MLP 데이터 수집에는 영향 없음
DISPLAY_CONF = 0.10

TARGET_CLASSES = [
    "fire",
    "smoke",
    "cigarette_butt",
    "spark",
]

#센서 변화량 설정

#최근 2초간의 변화량
CHANGE_WINDOW_SECONDS = 2.0
sensor_history = deque()

#CSV 저장 설정

#0.2초마다 1개 저장
SAVE_INTERVAL_SECONDS = 0.2

#프로그램 한 번 실행할 때 하나의 session ID
SESSION_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

CSV_COLUMNS = [
    "timestamp",
    "session_id",

    "fire_conf",
    "smoke_conf",
    "cigarette_conf",
    "spark_conf",

    "temperature",
    "gas",

    "temp_change",
    "gas_change",

    "label",
]

# 센서 데이터 파싱
NUMBER_PATTERN = (
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
)

CSV_PATTERN = re.compile(
    rf"^\s*({NUMBER_PATTERN})"
    rf"\s*,\s*"
    rf"({NUMBER_PATTERN})\s*$"
)

GAS_PATTERN = re.compile(
    rf"(?:GAS(?:_RAW)?|가스(?:값)?)"
    rf"\s*[:=]\s*({NUMBER_PATTERN})",
    re.IGNORECASE,
)

IR_PATTERN = re.compile(
    rf"(?:"
    rf"IR(?:_TEMP(?:ERATURE)?)?"
    rf"|TEMP(?:ERATURE)?"
    rf"|적외선(?:\s*온도)?"
    rf"|온도"
    rf")"
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

#Arduino 한 줄 데이터 분석
def parse_sensor_line(line):
    csv_match = CSV_PATTERN.fullmatch(line)

    if csv_match:
        gas_raw = int(
            float(csv_match.group(1))
        )
        ir_temperature = float(
            csv_match.group(2)
        )
        return (
            gas_raw,
            ir_temperature,
        )

    gas_raw = None
    ir_temperature = None

    gas_match = GAS_PATTERN.search(line)

    if gas_match:
        gas_raw = int(
            float(gas_match.group(1))
        )

    ir_match = IR_PATTERN.search(line)
    if ir_match:

        ir_temperature = float(
            ir_match.group(1)
        )

    return (
        gas_raw,
        ir_temperature,
    )

def read_arduino(stop_event):
    pending_gas = None
    pending_ir = None

    pending_gas_time = 0.0
    pending_ir_time = 0.0

    while not stop_event.is_set():
        try:
            print(
                f"[Arduino] "
                f"{ARDUINO_PORT} 연결 중"
            )

            with serial.Serial(
                port=ARDUINO_PORT,
                baudrate=ARDUINO_BAUDRATE,
                timeout=1,
            ) as arduino:
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
                    (gas_value, ir_value,) = parse_sensor_line(line)

                    current_time = (time.monotonic())

                    #가스
                    if gas_value is not None:
                        if (0<= gas_value<= 16383):
                            pending_gas = (gas_value)

                            pending_gas_time = (current_time)

                        else:
                            print("[Arduino] " "비정상 가스값:", gas_value,)

                    #온도
                    if ir_value is not None:
                        if (-40.0<= ir_value<= 300.0
                        ):
                            pending_ir = (ir_value)

                            pending_ir_time = (current_time)

                        else:
                            print("[Arduino] " "비정상 온도값:",ir_value,)


                    #둘 다 받아야 사용
                    if (pending_gas is None or pending_ir is None):
                        continue

                    #가스/온도 측정 시간 차이
                    sensor_time_difference = abs(
                        pending_gas_time
                        - pending_ir_time
                    )

                    #센서간 너무 오래 차이나면 사용하지 않음
                    if (sensor_time_difference> 3.0):
                        continue


                    # 공유 데이터 업데이트
                    with sensor_lock:
                        sensor_data[
                            "gas_raw"
                        ] = pending_gas
                        sensor_data[
                            "ir_temperature"
                        ] = pending_ir
                        sensor_data[
                            "last_update"
                        ] = current_time

                    print(
                        "[Arduino 정상 수신] "
                        f"gas={pending_gas}, "
                        f"ir={pending_ir:.1f}"
                    )


        except serial.SerialException as error:
            print("[Arduino] 연결 실패:", error,)

            if not stop_event.is_set():
                time.sleep(2)


        except Exception as error:
            print("[Arduino] " "예상하지 못한 오류:", repr(error),)

            if not stop_event.is_set():
                time.sleep(1)

#현재 센서값 가져오기

def get_sensor_data():
    with sensor_lock:
        gas_raw = (
            sensor_data["gas_raw"]
        )
        ir_temperature = (
            sensor_data[
                "ir_temperature"
            ]
        )
        last_update = (
            sensor_data["last_update"]
        )

    sensor_connected = (
        last_update > 0 and time.monotonic() - last_update <= 3.0
    )

    if not sensor_connected:
        return (0, 0.0, False,)

    return (gas_raw, ir_temperature, True,)

#Jetson CSI Camera

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

        f"appsink "
        f"drop=1 "
        f"sync=false"
    )

#YOLO 클래스 이름
def get_class_name(model, class_id,):
    if isinstance(model.names, dict,):
        return str(model.names[class_id])

    return str(model.names[class_id])

#CSV 준비
#파일이 없거나 비어있으면 header 생성
write_header = (not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0)

csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8",)

csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS,)

if write_header:
    csv_writer.writeheader()
    csv_file.flush()

print(
    f"[CSV] 저장 파일: "
    f"{CSV_PATH.resolve()}"
)

print(
    f"[CSV] Session ID: "
    f"{SESSION_ID}"
)

#YOLO 모델
print("[YOLO] 모델 로딩 중...")

model = YOLO(MODEL_PATH, task="detect",)

print("[YOLO] 모델 로딩 완료")

print("[YOLO] classes:", model.names,)

#Arduino Thread
stop_event = threading.Event()

arduino_thread = threading.Thread(target=read_arduino, args=(stop_event,), daemon=True,)
arduino_thread.start()

#Camera
cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER,)

if not cap.isOpened():
    stop_event.set()
    csv_file.close()

    raise RuntimeError("카메라를 열지 못했습니다.")

#데이터 수집 상태

#None = 아직 label 선택 안함
current_label = None

#처음에는 저장 정지 상태
recording = False

#마지막 저장 시각
last_save_time = 0.0

#이번 실행에서 저장한 데이터 개수
saved_count = 0

#사용법 출력
print()
print("=" * 55)
print("MLP 학습 데이터 수집")
print("=" * 55)

print("[0] 정상 상태 (label = 0)")

print("[1] 화재 상태 (label = 1)")

print("[S] CSV 저장 시작 / 일시정지")

print("[X] 현재 label 해제")

print("[Q] 종료")

print("=" * 55)
print()

#Main Loop
try:
    while True:
        #카메라
        ret, frame = cap.read()

        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            break

        #YOLO
        results = model(frame, conf=CONF_THRES, verbose=False,)

        #클래스별 최고 confidence
        yolo_confs = {
            "fire": 0.0,
            "smoke": 0.0,
            "cigarette_butt": 0.0,
            "spark": 0.0,
        }

        #화면 표시용
        best_detections = {}

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())

                confidence = float(box.conf[0].item())

                class_name = (get_class_name(model, cls_id,))

                #관심 없는 클래스 제외
                if(class_name not in TARGET_CLASSES):
                    continue

                #MLP 학습 데이터용 confidence
                yolo_confs[class_name] = max(yolo_confs[class_name], confidence,)

                #화면 표시용 threshold
                if (confidence< DISPLAY_CONF):
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist(),)

                if (class_name not in best_detections or confidence > best_detections[class_name]["conf"]):
                    best_detections[class_name] = {"conf":confidence, "box":(x1, y1, x2, y2,),}

        #Arduino Sensor
        (gas_raw, ir_temperature, sensor_connected,) = get_sensor_data()

        #최근 약 2초 센서 변화량

        temp_change = 0.0
        gas_change = 0.0

        if sensor_connected:
            current_time = (time.monotonic())
            sensor_history.append((current_time, ir_temperature, gas_raw,))

            #2초보다 오래된 값 제거
            while(
                sensor_history
                and
                current_time - sensor_history[0][0] > CHANGE_WINDOW_SECONDS
            ):

                sensor_history.popleft()

            #가장 오래된 값과 현재값 비교
            if len(sensor_history) >= 2:
                (
                    old_time,
                    old_temp,
                    old_gas,
                ) = sensor_history[0]

                temp_change = (
                    ir_temperature
                    - old_temp
                )

                gas_change = (
                    gas_raw
                    - old_gas
                )
        else:
            #센서 재연결 시 초기화
            sensor_history.clear()

        #CSV 저장
        now_monotonic = (
            time.monotonic()
        )

        should_save = (
            recording
            and
            current_label is not None
            and
            sensor_connected
            and
            (now_monotonic - last_save_time >= SAVE_INTERVAL_SECONDS)
        )

        if should_save:
            timestamp = (
                datetime.now().isoformat(
                    timespec="milliseconds"
                )
            )

            row = {
                "timestamp":
                    timestamp,
                "session_id":
                    SESSION_ID,
                "fire_conf":
                    yolo_confs["fire"],
                "smoke_conf":
                    yolo_confs["smoke"],
                "cigarette_conf":
                    yolo_confs[
                        "cigarette_butt"
                    ],
                "spark_conf":
                    yolo_confs["spark"],
                "temperature":
                    ir_temperature,
                "gas":
                    gas_raw,
                "temp_change":
                    temp_change,
                "gas_change":
                    gas_change,
                "label":
                    current_label,
            }

            csv_writer.writerow(
                row
            )

            #중간 종료돼도 데이터 최대한 보존
            csv_file.flush()

            saved_count += 1
            last_save_time = (now_monotonic)

            print(
                f"[SAVE #{saved_count}] "
                f"label={current_label} | "
                f"fire={yolo_confs['fire']:.3f} | "
                f"smoke={yolo_confs['smoke']:.3f} | "
                f"cig={yolo_confs['cigarette_butt']:.3f} | "
                f"spark={yolo_confs['spark']:.3f} | "
                f"temp={ir_temperature:.1f} | "
                f"gas={gas_raw} | "
                f"dTemp={temp_change:+.1f} | "
                f"dGas={gas_change:+.0f}"
            )

        #화면
        annotated_frame = (frame.copy())

        #Bounding Box
        for (class_name, data,) in best_detections.items():
            confidence = (data["conf"])
            (x1, y1, x2, y2,) = data["box"]

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                (
                    f"{class_name} "
                    f"{confidence:.2f}"
                ),
                (
                    x1,
                    max(
                        25,
                        y1 - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        #상태 표시
        sensor_status = (
            "OK"
            if sensor_connected
            else
            "DISCONNECTED"
        )

        if current_label is None:
            label_text = (
                "LABEL: NOT SELECTED"
            )

        elif current_label == 0:
            label_text = (
                "LABEL: 0 NORMAL"
            )

        else:
            label_text = (
                "LABEL: 1 FIRE"
            )

        recording_text = (
            "RECORDING"
            if recording
            else
            "PAUSED"
        )

        #라벨
        cv2.putText(
            annotated_frame,
            label_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (
                0,
                0,
                255,
            )
            if current_label == 1
            else
            (
                0,
                255,
                0,
            ),
            2,
        )

        #저장 상태
        cv2.putText(
            annotated_frame,
            (
                f"STATE: "
                f"{recording_text} "
                f"| Saved:{saved_count}"
            ),
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        #YOLO confidence
        cv2.putText(
            annotated_frame,
            (
                f"Fire:{yolo_confs['fire']:.2f}  "
                f"Smoke:{yolo_confs['smoke']:.2f}  "
                f"Cig:{yolo_confs['cigarette_butt']:.2f}  "
                f"Spark:{yolo_confs['spark']:.2f}"
            ),
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        #센서
        cv2.putText(
            annotated_frame,
            (
                f"Temp:{ir_temperature:.1f}C  "
                f"Gas:{gas_raw}  "
                f"dTemp:{temp_change:+.1f}  "
                f"dGas:{gas_change:+.0f}"
            ),
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        #센서 상태
        cv2.putText(
            annotated_frame,
            (
                f"Sensor: "
                f"{sensor_status}"
            ),
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                255,
                0,
            )
            if sensor_connected
            else
            (
                0,
                0,
                255,
            ),
            2,
        )


        #키 도움말
        cv2.putText(
            annotated_frame,
            (
                "0=Normal  "
                "1=Fire  "
                "S=Record/Pause  "
                "X=Clear  "
                "Q=Quit"
            ),
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "MLP Training Data Collector",
            annotated_frame,
        )

        #키 입력
        key = (cv2.waitKey(1)& 0xFF)

        #정상 label
        if key == ord("0"):
            current_label = 0
            print()
            print(">>> Label = 0 " "(NORMAL)")
            print()


        #화재 label
        elif key == ord("1"):
            current_label = 1
            print()
            print(
                ">>> Label = 1 "
                "(WARNING)"
            )
            print()

        #저장 시작/정지
        elif key in (
            ord("s"),
            ord("S"),
        ):
            if current_label is None:
                print("먼저 0 또는 1로 " "label을 선택하세요.")
            else:
                recording = not recording
                print()
                if recording:
                    print(">>> CSV 저장 시작")
                    print(
                        f">>> 현재 label = "
                        f"{current_label}"
                    )
                else:
                    print(">>> CSV 저장 일시정지")
                print()


        #label 초기화
        elif key in (
            ord("x"),
            ord("X"),
        ):
            recording = False
            current_label = None
            print(">>> Label 초기화 / " "저장 정지")

        #종료
        elif key in (
            ord("q"),
            ord("Q"),
        ):
            break

except KeyboardInterrupt:
    print("\n프로그램을 종료합니다.")

#종료
finally:
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print()
    print("=" * 55)

    print(
        f"총 저장 데이터: "
        f"{saved_count}개"
    )
    print(
        f"CSV 위치: "
        f"{CSV_PATH.resolve()}"
    )
    print(
        f"Session ID: "
        f"{SESSION_ID}"
    )
    print("=" * 55)