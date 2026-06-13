import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'debug_front_frame': None,  # Front frame with five-lane token labels
    'steering_input' : 0.0,
    'acceleration_input' : 1.0,
    'target_lane': 1,        # 0: Left, 1: Middle, 2: Right
    'current_lane': 1,
    'low_brightness': False,  # Event flag
    # NEW: Darkness Improvement
    'front_brightness': 255.0,
    'darkness_enter_count': 0,
    'darkness_exit_count': 0,
    'lights_on': False,
    'light_signal_timer': 0,
    'tap_timer': 0,
    'cooldown_timer': 0,
    'tap_steering': 0.0,
    'police_active': False,   # Dynamic tracking state for police event
    'red_detect_count': 0,
    'yellow_detect_count': 0
}
data_lock = threading.Lock()
is_running = True

# ---------------------------------------------------------
# Debug Helper Configuration: five perspective-aware road lanes
# These values match the trapezoid used by the existing road ROI.
# The helper is visualization-only: it does not change steering decisions.
# ---------------------------------------------------------
NUM_LANES = 5
ROAD_TOP_Y = 70
ROAD_BOTTOM_Y = 239
ROAD_LEFT_TOP = 115
ROAD_RIGHT_TOP = 205
ROAD_LEFT_BOTTOM = 40
ROAD_RIGHT_BOTTOM = 280

# Tighter token-detection ROI. This excludes the right road shoulder/curb and
# the bottom player-car area while keeping the wider debug lane guides above.
DETECT_ROAD_TOP_Y = 78
DETECT_ROAD_BOTTOM_Y = 239
DETECT_ROAD_LEFT_TOP = 120
DETECT_ROAD_RIGHT_TOP = 200
DETECT_ROAD_LEFT_BOTTOM = 52
DETECT_ROAD_RIGHT_BOTTOM = 248
EGO_IGNORE_LEFT = 102
EGO_IGNORE_RIGHT = 218
EGO_IGNORE_TOP = 185

# Token-shape filters. They reject grass bands, lane markings, road shoulder
# fragments, and the player's own car while still accepting round tokens.
TOKEN_MIN_AREA = 8
TOKEN_MAX_AREA = 5200
TOKEN_MAX_DIMENSION = 96
TOKEN_MIN_ASPECT = 0.35
TOKEN_MAX_ASPECT = 2.6
TOKEN_MIN_FILL_RATIO = 0.20
RIGHT_SHOULDER_REJECT_MARGIN = 18

# ---------------------------------------------------------
# NEW: Darkness Improvement Configuration
# ---------------------------------------------------------
# Enter/exit Darkness Mode only after several frames so it does not flicker.
DARKNESS_ENTER_THRESHOLD = 60.0
DARKNESS_ENTER_FRAMES = 4
DARKNESS_EXIT_THRESHOLD = 72.0
DARKNESS_EXIT_FRAMES = 8

# Extremely dark scene: avoid danger first, skip optional green chasing.
CRITICAL_DARKNESS_THRESHOLD = 38.0

# Slow down and reduce unnecessary steering during low visibility.
DARKNESS_ACCELERATION = 0.76
CRITICAL_DARKNESS_ACCELERATION = 0.62
LIGHT_TOGGLE_ACCELERATION = -1.0
LIGHT_TOGGLE_FRAMES = 6
NORMAL_TAP_FRAMES = 12
DARKNESS_TAP_FRAMES = 9
NORMAL_COOLDOWN_FRAMES = 10
DARKNESS_COOLDOWN_FRAMES = 12

# Improve token detection in dark scenes.
DARK_KERNEL = np.ones((3, 3), np.uint8)
GAMMA_TABLE = np.array([
    ((value / 255.0) ** 0.62) * 255
    for value in range(256)
]).astype('uint8')
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception as e:
            pass

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None

def setup_cameras():
    global front_camera_sock, back_camera_sock
    
    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False
    
    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass
                
        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass
                
        if not (front_connected and back_connected):
            time.sleep(1)

def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")
    
    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue

# ---------------------------------------------------------
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------

def read_single_camera(sock, window_name, data_key):
    #This function reads the latest frame from the camera socket and stores it in the shared data
    if sock is None:
        return
        
    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return
            
        image_length = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet
            
        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes
            
        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break
                
            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet
                
            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes
                
        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with data_lock:
                    shared_data[data_key] = frame
                
                # Disabled GUI rendering in background threads to significantly improve FPS
                # frame_resized = cv2.resize(frame, (640, 480))
                # cv2.imshow(window_name, frame_resized)
                # cv2.waitKey(1)
                
    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')

# ---------------------------------------------------------
# Debug helpers: detect a token's perspective-correct lane and
# build compact color observations for the front-camera overlay.
# ---------------------------------------------------------
def road_edges_at_y(pixel_y):
    """Return the road's left and right edges at a vertical pixel position."""
    pixel_y = float(np.clip(pixel_y, ROAD_TOP_Y, ROAD_BOTTOM_Y))
    scale = (pixel_y - ROAD_TOP_Y) / float(ROAD_BOTTOM_Y - ROAD_TOP_Y)
    left = ROAD_LEFT_TOP + (ROAD_LEFT_BOTTOM - ROAD_LEFT_TOP) * scale
    right = ROAD_RIGHT_TOP + (ROAD_RIGHT_BOTTOM - ROAD_RIGHT_TOP) * scale
    return left, right


def pixel_x_to_debug_lane(pixel_x, pixel_y):
    """Map a token center to L0..L4 while accounting for road perspective."""
    left, right = road_edges_at_y(pixel_y)
    road_width = right - left
    if road_width <= 0:
        return NUM_LANES // 2

    relative_x = (float(pixel_x) - left) / road_width
    lane = int(np.floor(relative_x * NUM_LANES))
    return int(np.clip(lane, 0, NUM_LANES - 1))


def debug_lane_divider_x(divider_index, pixel_y):
    """Return the x-position of a white divider between two debug lanes."""
    left, right = road_edges_at_y(pixel_y)
    return int(left + (right - left) * divider_index / float(NUM_LANES))


def detection_edges_at_y(pixel_y):
    """Return the tighter token-detection area's left and right edges."""
    pixel_y = float(np.clip(pixel_y, DETECT_ROAD_TOP_Y, DETECT_ROAD_BOTTOM_Y))
    scale = (pixel_y - DETECT_ROAD_TOP_Y) / float(DETECT_ROAD_BOTTOM_Y - DETECT_ROAD_TOP_Y)
    left = DETECT_ROAD_LEFT_TOP + (DETECT_ROAD_LEFT_BOTTOM - DETECT_ROAD_LEFT_TOP) * scale
    right = DETECT_ROAD_RIGHT_TOP + (DETECT_ROAD_RIGHT_BOTTOM - DETECT_ROAD_RIGHT_TOP) * scale
    return left, right


def is_token_like_contour(contour, color_code=None):
    """Reject road art, lane lines, grass bands, and the player's own car."""
    area = cv2.contourArea(contour)
    if area < TOKEN_MIN_AREA or area > TOKEN_MAX_AREA:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    if max(w, h) > TOKEN_MAX_DIMENSION:
        return False

    aspect_ratio = w / float(max(h, 1))
    fill_ratio = area / float(max(w * h, 1))
    if (
        aspect_ratio < TOKEN_MIN_ASPECT
        or aspect_ratio > TOKEN_MAX_ASPECT
        or fill_ratio < TOKEN_MIN_FILL_RATIO
    ):
        return False

    center_x = x + (w // 2)
    bottom_y = y + h

    if bottom_y >= EGO_IGNORE_TOP and EGO_IGNORE_LEFT <= center_x <= EGO_IGNORE_RIGHT:
        return False

    # The red-white right road shoulder often breaks into compact red blobs.
    # Reject red detections hugging the detection boundary in the lower half.
    if color_code == 'R':
        _, right_edge = detection_edges_at_y(bottom_y)
        if bottom_y > 120 and center_x > right_edge - RIGHT_SHOULDER_REJECT_MARGIN:
            return False

    return True


def collect_token_observations(contours, color_code, max_area=None, max_dimension=None):
    """Convert contours into compact token records used only by the overlay."""
    observations = []

    for contour in contours:
        if not is_token_like_contour(contour, color_code):
            continue

        area = cv2.contourArea(contour)

        x, y, w, h = cv2.boundingRect(contour)
        if max_area is not None and area > max_area:
            continue
        if max_dimension is not None and max(w, h) > max_dimension:
            continue

        center_x = x + w // 2
        bottom_y = y + h
        observations.append({
            'color': color_code,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'lane': pixel_x_to_debug_lane(center_x, bottom_y),
        })

    return observations


# ---------------------------------------------------------
# NEW: Darkness Improvement Helper Functions
# ---------------------------------------------------------
def set_vehicle_light(is_on, brightness):
    """
    Track the vehicle light state when Darkness Mode changes and schedule the
    Unity-recognized light command. The game logs show that acceleration -1.0
    during darkness restores the lights.
    """
    if shared_data['lights_on'] == is_on:
        return

    shared_data['lights_on'] = is_on
    if is_on:
        shared_data['light_signal_timer'] = LIGHT_TOGGLE_FRAMES
    light_state = "ON" if is_on else "OFF"
    print(f"Vehicle Light {light_state} | brightness={brightness:.1f}")


def update_darkness_state(brightness):
    """
    Prevent Darkness Mode from switching on/off because of one unusual frame.
    """
    with data_lock:
        shared_data['front_brightness'] = float(brightness)

        if not shared_data['low_brightness']:
            shared_data['darkness_exit_count'] = 0
            if brightness < DARKNESS_ENTER_THRESHOLD:
                shared_data['darkness_enter_count'] += 1
            else:
                shared_data['darkness_enter_count'] = 0

            if shared_data['darkness_enter_count'] >= DARKNESS_ENTER_FRAMES:
                shared_data['low_brightness'] = True
                shared_data['darkness_enter_count'] = 0
                set_vehicle_light(True, brightness)
                print(f"Darkness Mode ON | brightness={brightness:.1f}")
        else:
            shared_data['darkness_enter_count'] = 0
            if brightness > DARKNESS_EXIT_THRESHOLD:
                shared_data['darkness_exit_count'] += 1
            else:
                shared_data['darkness_exit_count'] = 0

            if shared_data['darkness_exit_count'] >= DARKNESS_EXIT_FRAMES:
                shared_data['low_brightness'] = False
                shared_data['darkness_exit_count'] = 0
                set_vehicle_light(False, brightness)
                print(f"Darkness Mode OFF | brightness={brightness:.1f}")

        return shared_data['low_brightness']


def enhance_low_light_frame(frame):
    """
    Brighten the camera frame before token detection.
    Gamma correction brightens dark pixels; CLAHE improves local contrast.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    value = cv2.LUT(value, GAMMA_TABLE)
    value = CLAHE.apply(value)
    enhanced_hsv = cv2.merge((hue, saturation, value))
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)


def processing_task():
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data['latest_back_frame']
        police_mode = shared_data['police_active']
    
    steering_target = 0.0
    frame_center = 160
    
    # ---------------------------------------------------------
    # BACK CAMERA ENVIRONMENT ANALYSIS (Trailing Car Evasion) [cite: 31]
    # ---------------------------------------------------------
    evade_back_car = False
    back_steering_escape = 0.0
    
    if back_frame is not None:
        small_back = cv2.resize(back_frame, (320, 240))
        gray_back = cv2.cvtColor(small_back, cv2.COLOR_BGR2GRAY)
        
        # Check for proximity of a trailing vehicle via structural changes or looming bounding boxes
        # Assuming vehicles from behind appear prominently in a specific mid-lane mask area
        back_roi = gray_back[120:240, 60:260]
        avg_intensity = np.mean(back_roi)
        contrast = np.std(back_roi)
        
        # Brightness alone was too sensitive, so require stronger brightness + contrast.
        if (avg_intensity > 225 or avg_intensity < 25) and contrast > 18:
            evade_back_car = True
            # Check which side has more space or default a structural escape jump
            back_steering_escape = 1.0  # Tap right to clear the lane [cite: 153, 156]

    # ---------------------------------------------------------
    # FRONT CAMERA ENVIRONMENT ANALYSIS (Token Processing)
    # ---------------------------------------------------------
    if front_frame is not None:
        small_frame = cv2.resize(front_frame, (320, 240))
        
        # ---------------------------------------------------------
        # NEW: Darkness Improvement
        # ---------------------------------------------------------
        gray_front = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray_front))
        low_brightness = update_darkness_state(brightness)
        critical_darkness = (
            low_brightness and brightness < CRITICAL_DARKNESS_THRESHOLD
        )

        if low_brightness:
            analysis_frame = enhance_low_light_frame(small_frame)
        else:
            analysis_frame = small_frame

        hsv_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2HSV)
        
        # Extended Range Color Masks [cite: 140, 141]
        # NEW: Darkness Improvement - widen HSV ranges only in dark scenes.
        if low_brightness:
            lower_green = np.array([32, 28, 28])
            upper_green = np.array([92, 255, 255])
            lower_red1 = np.array([0, 35, 28])
            upper_red1 = np.array([13, 255, 255])
            lower_red2 = np.array([167, 35, 28])
            upper_red2 = np.array([180, 255, 255])
            lower_yellow = np.array([12, 35, 30])
            upper_yellow = np.array([35, 255, 255])
        else:
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            lower_yellow = np.array([15, 50, 50])
            upper_yellow = np.array([30, 255, 255])

        mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
        
        mask_red = cv2.bitwise_or(cv2.inRange(hsv_frame, lower_red1, upper_red1), 
                                  cv2.inRange(hsv_frame, lower_red2, upper_red2))
                                  
        mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # Grey helper mask: useful for visual checks after a yellow effect or
        # during low brightness. Compact filtering below rejects large road areas.
        lower_grey = np.array([0, 0, 80])
        upper_grey = np.array([180, 48, 205])
        mask_grey = cv2.inRange(hsv_frame, lower_grey, upper_grey)

        # Road Region of Interest (ROI): only detect tokens inside the visible road.
        # The polygon is tuned for the resized 320 x 240 front-camera frame.
        road_mask = np.zeros((240, 320), dtype=np.uint8)
        road_polygon = np.array([
            [DETECT_ROAD_LEFT_BOTTOM, DETECT_ROAD_BOTTOM_Y],
            [DETECT_ROAD_RIGHT_BOTTOM, DETECT_ROAD_BOTTOM_Y],
            [DETECT_ROAD_RIGHT_TOP, DETECT_ROAD_TOP_Y],
            [DETECT_ROAD_LEFT_TOP, DETECT_ROAD_TOP_Y]
        ], dtype=np.int32)
        cv2.fillPoly(road_mask, [road_polygon], 255)

        # Ignore the player's own red car/hood at the bottom of the camera.
        road_mask[EGO_IGNORE_TOP:240, EGO_IGNORE_LEFT:EGO_IGNORE_RIGHT] = 0

        # Apply the road ROI before contour detection so off-road colors are ignored.
        mask_green = cv2.bitwise_and(mask_green, road_mask)
        mask_red = cv2.bitwise_and(mask_red, road_mask)
        mask_yellow = cv2.bitwise_and(mask_yellow, road_mask)
        mask_grey = cv2.bitwise_and(mask_grey, road_mask)

        # NEW: Darkness Improvement - reconnect small broken token regions.
        if low_brightness:
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, DARK_KERNEL)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, DARK_KERNEL)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, DARK_KERNEL)
        
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_grey, _ = cv2.findContours(mask_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # These observations are used only for the debug frame. Existing steering
        # continues using the original green/red/yellow contour logic.
        debug_greens = collect_token_observations(contours_green, 'G')
        debug_reds = collect_token_observations(contours_red, 'R')
        debug_yellows = collect_token_observations(contours_yellow, 'Y')
        debug_greys = collect_token_observations(
            contours_grey,
            'X',
            max_area=1500,
            max_dimension=70,
        )
        filtered_red_contours = [
            contour for contour in contours_red
            if is_token_like_contour(contour, 'R')
        ]

        # Select the most relevant token instead of blindly choosing
        # the largest contour. Nearby tokens and tokens closer to the
        # vehicle's driving path receive a higher score.
        def select_relevant_contour(contours, danger_only=False, color_code=None):
            candidates = []

            for contour in contours:
                if not is_token_like_contour(contour, color_code):
                    continue

                area = cv2.contourArea(contour)

                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + (w // 2)
                bottom_y = y + h
                horizontal_distance = abs(center_x - frame_center)

                # Red and yellow tokens outside the current driving
                # corridor should not trigger unnecessary avoidance.
                if danger_only and horizontal_distance > 75:
                    continue

                # Prefer nearby tokens, while still considering their
                # visible size and horizontal relevance.
                score = (
                    bottom_y * 2.0
                    + min(area, 120)
                    - horizontal_distance * 0.35
                )

                candidates.append((score, contour))

            if not candidates:
                return None

            return max(candidates, key=lambda item: item[0])[1]

        # Decision State Hierarchy [cite: 15]
        if evade_back_car:
            # Priority 1: Do not get wrecked from behind [cite: 31]
            steering_target = back_steering_escape
            print(f"Trailing Car Alert! Escaping to steering: {steering_target}")
            
        elif police_mode and filtered_red_contours:
            # Priority 2: Catch red token intentionally if Police are chasing [cite: 33, 213]
            largest_red = max(filtered_red_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > 5:
                M = cv2.moments(largest_red)
                if M['m00'] > 0:
                    rx = int(M['m10'] / M['m00'])
                    error = rx - frame_center
                    steering_target = -1.0 if error < -15 else (1.0 if error > 15 else 0.0)
                    if abs(error) <= 15:
                        with data_lock:
                            shared_data['police_active'] = False # Event successfully clear [cite: 17, 201]

        else:
            # Priority 3: Standard Navigation (Seek Green [cite: 19, 204], Dodge Red [cite: 20, 204], Dodge Yellow [cite: 21, 205])
            red_detected = False
            yellow_detected = False
            
            # Evade Red Tokens [cite: 20, 204]
            best_red = select_relevant_contour(contours_red, danger_only=True, color_code='R')

            if best_red is not None:
                largest_red = best_red

                rx, ry, rw, rh = cv2.boundingRect(largest_red)
                red_bottom = ry + rh
                red_area = cv2.contourArea(largest_red)

                if red_area > 5 and red_bottom > 60:
                    with data_lock:
                        shared_data['red_detect_count'] += 1

                    if shared_data['red_detect_count'] >= 2:
                        M = cv2.moments(largest_red)
                        if M['m00'] > 0:
                            rx = int(M['m10'] / M['m00'])
                            steering_target = -1.0 if rx > frame_center else 1.0
                            red_detected = True
                            print("Evading Red Token Early!")
                else:
                    with data_lock:
                        shared_data['red_detect_count'] = 0
            else:
                with data_lock:
                    shared_data['red_detect_count'] = 0

            # Evade Yellow Corruption Fields [cite: 21, 205]
            best_yellow = select_relevant_contour(contours_yellow, danger_only=True, color_code='Y')

            if not red_detected and best_yellow is not None:
                largest_yellow = best_yellow

                yx, yy, yw, yh = cv2.boundingRect(largest_yellow)
                yellow_bottom = yy + yh
                yellow_area = cv2.contourArea(largest_yellow)

                if yellow_area > 5 and yellow_bottom > 60:
                    with data_lock:
                        shared_data['yellow_detect_count'] += 1

                    if shared_data['yellow_detect_count'] >= 2:
                        M = cv2.moments(largest_yellow)
                        if M['m00'] > 0:
                            yx = int(M['m10'] / M['m00'])
                            steering_target = -1.0 if yx > frame_center else 1.0
                            yellow_detected = True
                            print("Evading Corruptive Yellow Token Early!")
                else:
                    with data_lock:
                        shared_data['yellow_detect_count'] = 0
            else:
                with data_lock:
                    shared_data['yellow_detect_count'] = 0

            # Collect Speed Upgrades [cite: 19, 204]
            best_green = select_relevant_contour(contours_green, color_code='G')

            # NEW: Darkness Improvement
            # During critical darkness, avoid danger first and skip optional green tokens.
            if not red_detected and not yellow_detected and not critical_darkness and best_green is not None:
                largest_green = best_green
                
                gx, gy, gw, gh = cv2.boundingRect(largest_green)
                green_bottom = gy + gh
                if low_brightness:
                    minimum_green_area = 48
                    minimum_green_bottom = 125
                else:
                    minimum_green_area = 35
                    minimum_green_bottom = 105

                if cv2.contourArea(largest_green) > minimum_green_area and green_bottom > minimum_green_bottom:
                    M = cv2.moments(largest_green)
                    if M['m00'] > 0:
                        gx = int(M['m10'] / M['m00'])
                        error = gx - frame_center
                        if error < -35:
                            steering_target = -1.0
                        elif error > 35:
                            steering_target = 1.0
                        print(f"Targeting Green Token! Error: {error}") 

        # Commit decision to shared resources safely using a Tap Sequence [cite: 152]
        with data_lock:
            if shared_data['tap_timer'] > 0:
                # State 1: Actively tapping [cite: 156]
                shared_data['tap_timer'] -= 1
                shared_data['steering_input'] = shared_data['tap_steering']
                if shared_data['tap_timer'] == 0:
                    # NEW: Darkness Improvement
                    # Wait slightly longer during darkness to reduce random left-right movement.
                    if shared_data['low_brightness']:
                        shared_data['cooldown_timer'] = DARKNESS_COOLDOWN_FRAMES
                    else:
                        shared_data['cooldown_timer'] = NORMAL_COOLDOWN_FRAMES
                    shared_data['steering_input'] = 0.0
            elif shared_data['cooldown_timer'] > 0:
                # State 2: Waiting for car to settle [cite: 157]
                shared_data['cooldown_timer'] -= 1
                shared_data['steering_input'] = 0.0
            else:
                # State 0: Ready for a new tap [cite: 154]
                if steering_target != 0.0:
                    shared_data['tap_steering'] = steering_target
                    # NEW: Darkness Improvement - use shorter steering taps in darkness.
                    if shared_data['low_brightness']:
                        shared_data['tap_timer'] = DARKNESS_TAP_FRAMES
                    else:
                        shared_data['tap_timer'] = NORMAL_TAP_FRAMES
                    shared_data['steering_input'] = steering_target
                    print(f"Initiating Tap! Steering: {steering_target}")
                else:
                    shared_data['steering_input'] = 0.0

        # ---------------------------------------------------------
        # Front-camera debug frame: white five-lane guides plus token labels.
        # G = green, R = red, Y = yellow, X = grey.
        # This display does not alter the driving controller.
        # ---------------------------------------------------------
        # NEW: Darkness Improvement - display enhanced frame during Darkness Mode.
        debug_frame = analysis_frame.copy()

        for divider_index in range(1, NUM_LANES):
            top_x = debug_lane_divider_x(divider_index, ROAD_TOP_Y)
            bottom_x = debug_lane_divider_x(divider_index, ROAD_BOTTOM_Y)
            cv2.line(
                debug_frame,
                (top_x, ROAD_TOP_Y),
                (bottom_x, ROAD_BOTTOM_Y),
                (255, 255, 255),
                1,
            )

        cv2.polylines(debug_frame, [road_polygon], True, (255, 255, 255), 1)

        debug_colors = {
            'G': (0, 255, 0),
            'R': (0, 0, 255),
            'Y': (0, 255, 255),
            'X': (180, 180, 180),
        }

        for token in debug_greens + debug_reds + debug_yellows + debug_greys:
            draw_color = debug_colors[token['color']]
            cv2.rectangle(
                debug_frame,
                (token['x'], token['y']),
                (token['x'] + token['w'], token['y'] + token['h']),
                draw_color,
                1,
            )
            cv2.putText(
                debug_frame,
                f"{token['color']} L{token['lane']}",
                (token['x'], max(12, token['y'] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                draw_color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            debug_frame,
            (
                f"mode={'DARKNESS' if low_brightness else 'NORMAL'} | "
                f"light={'SIGNAL' if shared_data['light_signal_timer'] > 0 else ('ON' if shared_data['lights_on'] else 'OFF')} | "
                f"brightness={brightness:.1f}"
            ),
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            debug_frame,
            f"lanes={NUM_LANES} | G=green R=red Y=yellow X=grey",
            (5, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        with data_lock:
            shared_data['debug_front_frame'] = debug_frame


def send_controls_task():
    global control_conn
    if control_conn is None:
        # Workaround: Safely restart the locked server function if it died
        is_server_running = any(t.name == "ControlServerRecovery" for t in threading.enumerate())
        if not is_server_running:
            print("Connection missing. Restarting control server...")
            threading.Thread(target=setup_control_server, name="ControlServerRecovery", daemon=True).start()
        return
    
    with data_lock:
        steering_to_send = shared_data['steering_input']
        # NEW: Darkness Improvement
        # Unity treats acceleration -1.0 during darkness as the light recovery command.
        if shared_data['light_signal_timer'] > 0:
            shared_data['light_signal_timer'] -= 1
            steering_to_send = 0.0
            acceleration_to_send = LIGHT_TOGGLE_ACCELERATION
        elif shared_data['low_brightness']:
            if shared_data['front_brightness'] < CRITICAL_DARKNESS_THRESHOLD:
                acceleration_to_send = CRITICAL_DARKNESS_ACCELERATION
            else:
                acceleration_to_send = DARKNESS_ACCELERATION
        else:
            acceleration_to_send = 1.0 # Default: full gas ahead [cite: 174]

    try:
        # Pack and send the control command to Unity [cite: 177]
        data = struct.pack('ff', steering_to_send, acceleration_to_send)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}") 
        control_conn = None


# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")
    
    # Initialize network connections
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")
    
    # This is where you define tasks with explicit Scheduling parameters (Concurrency, Priority, Period)
    # Period refers to the period of execution of the task in seconds
    # Priority refers to the priority of the task, higher priority means higher priority
    # Concurrency refers to the number of instances of the task that can run at the same time
    t_front_camera = RTTask("ReadFrontCamera", period=0.033, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.033, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.033, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.033, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    
    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()
    
    try:
        # You need this to keep the main thread alive, otherwise the program will exit immediately
        while is_running:
            with data_lock:
                front = shared_data.get('latest_front_frame')
                debug_front = shared_data.get('debug_front_frame')
                back = shared_data.get('latest_back_frame')

            front_to_show = debug_front if debug_front is not None else front
            if front_to_show is not None:
                cv2.imshow("Front Camera - AI Driving", cv2.resize(front_to_show, (640, 480)))
            if back is not None:
                cv2.imshow("Back Camera", cv2.resize(back, (640, 480)))

            if cv2.waitKey(33) & 0xFF == ord('q'):
                is_running = False
                break
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    # This is to make sure that the tasks are terminated cleanly
    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()
    
    # This is to close all the connections
    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
