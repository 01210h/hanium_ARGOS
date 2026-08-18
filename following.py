import cv2
import numpy as np
import time
import serial

# 아두이노 USB 시리얼
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

TARGET_DISTANCE = 0.9

# 보정 예외
DIST_DEADBAND = 0.08
X_DEADBAND = 0.03
# 거리 ±8cm, 좌우 ±3cm

# 속도 제한
MAX_FORWARD = 140
MAX_TURN = 100
MAX_PWM = 255

# 아두이노로 보내는 주기
SEND_INTERVAL = 0.05

# ArUco 마커 ID
TARGET_MARKER_ID = 5

# P 제어
KP_DISTANCE = 200
KP_TURN = 400

# 범위 제한
def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

# 거리/좌우 제어
def calculate_motor_speed(pos_x, pos_z):
    distance_error = pos_z - TARGET_DISTANCE

    # 거리 오차가 예외 범위라면 보정 안 함
    if abs(distance_error) <= DIST_DEADBAND:
        forward = 0
    else:
        forward = KP_DISTANCE * distance_error

    # 좌우 오차
    if abs(pos_x) <= X_DEADBAND:
        turn = 0
    else:
        turn = KP_TURN * pos_x

    # 전진/회전 각각 제한
    forward = clamp(forward, -MAX_FORWARD, MAX_FORWARD)
    turn = clamp(turn, -MAX_TURN, MAX_TURN)

    # 차동구동
    left_speed = forward + turn
    right_speed = forward - turn

    # 최종 PWM 제한
    left_speed = clamp(left_speed, -MAX_PWM, MAX_PWM)
    right_speed = clamp(right_speed, -MAX_PWM, MAX_PWM)

    return int(left_speed), int(right_speed)

# 아두이노에 전송
def send_motor(arduino, left_speed, right_speed):
    message = f"{left_speed},{right_speed}\n"
    arduino.write(message.encode())

def live_aruco_detection(arduino):
    # ArUco 설정
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_250
    )
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        aruco_params
    )

    # 실제 마커 크기: 11.5cm
    marker_size = 0.115

    # 카메라 설정
    frame_width = 640
    frame_height = 480

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # 임시 카메라 내부 파라미터
    fx = 640.0
    fy = 640.0
    cx = frame_width / 2
    cy = frame_height / 2

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # 렌즈 왜곡이 없다고 임시 가정
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    # 카메라 초기화 대기
    time.sleep(2)
    last_send_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # 기본값은 정지
            left_speed = 0
            right_speed = 0

            # 마커 검출
            corners, ids, rejected = detector.detectMarkers(frame)

            if ids is not None:
                # 검출된 마커 표시
                cv2.aruco.drawDetectedMarkers(
                    frame,
                    corners,
                    ids
                )
                # 마커 위치/자세 추정
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    marker_size,
                    camera_matrix,
                    dist_coeffs
                )
                target_index = None
                # ID 5 마커 탐색
                for i in range(len(ids)):
                    if ids[i][0] == TARGET_MARKER_ID:
                        target_index = i
                        break

                if target_index is not None:
                    i = target_index

                    # 마커 위치
                    pos_x = tvecs[i][0][0]
                    pos_y = tvecs[i][0][1]
                    pos_z = tvecs[i][0][2]

                    # 모터 속도 계산
                    left_speed, right_speed = calculate_motor_speed(
                        pos_x,
                        pos_z
                    )

                    # 좌표축 표시
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvecs[i],
                        tvecs[i],
                        marker_size / 2
                    )

                    # 회전 벡터 -> 오일러 각도
                    rot_matrix, _ = cv2.Rodrigues(rvecs[i])

                    euler_angles = cv2.RQDecomp3x3(
                        rot_matrix
                    )[0]

                    # 마커 중앙 위치
                    corner = corners[i][0]

                    center_x = int(
                        np.mean(corner[:, 0])
                    )

                    center_y = int(
                        np.mean(corner[:, 1])
                    )

                    # ID 표시
                    cv2.putText(
                        frame,
                        f"ID: {ids[i][0]}",
                        (center_x, center_y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )

                    # 위치 표시
                    cv2.putText(
                        frame,
                        f"Pos: ({pos_x:.2f}, "
                        f"{pos_y:.2f}, "
                        f"{pos_z:.2f})m",
                        (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )

                    # 회전 표시
                    cv2.putText(
                        frame,
                        f"Rot: "
                        f"({euler_angles[0]:.1f}, "
                        f"{euler_angles[1]:.1f}, "
                        f"{euler_angles[2]:.1f})deg",
                        (center_x, center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )

                    # 코너 표시
                    for point in corner:
                        x = int(point[0])
                        y = int(point[1])

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 0, 255),
                            -1
                        )

                    # 모터 PWM 화면 표시
                    cv2.putText(
                        frame,
                        f"Motor: L={left_speed} R={right_speed}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

            # 아두이노에게 명령 전송
            current_time = time.monotonic()
            if current_time - last_send_time >= SEND_INTERVAL:

                send_motor(
                    arduino,
                    left_speed,
                    right_speed
                )

                last_send_time = current_time
            cv2.imshow(
                "ArUco Marker Detection",
                frame
            )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 프로그램 종료 시 무조건 정지
        send_motor(
            arduino,
            0,
            0
        )
        cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        arduino = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=0.1
        )
        time.sleep(2)
        print(
            f"Arduino connected: {SERIAL_PORT}"
        )
    except Exception as e:
        print(
            f"Arduino connection failed: {e}"
        )
        return
    print(
        "Starting ArUco marker following..."
    )
    try:
        live_aruco_detection(
            arduino
        )
    finally:
        arduino.close()


if __name__ == "__main__":
    main()