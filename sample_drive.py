import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes
import os

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
    'last_front_brightness': 255.0,
    'darkness_enter_count': 0,
    'darkness_exit_count': 0,
    'lights_on': False,
    'light_signal_timer': 0,
    'tap_timer': 0,
    'cooldown_timer': 0,
    'tap_steering': 0.0,
    'police_active': False,
    'police_timer': 0,
    'police_detect_count': 0,
    'police_rearm_timer': 0,
    'police_memory_timer': 0,
    'police_memory_score': 0.0,
    'police_last_position': None,
    'police_velocity': None,
    'police_avoid_timer': 0,
    'police_avoid_steering': 0.0,
    'police_avoid_acceleration': 0.25,
    'police_red_attempted': False,
    'police_debug_capture_cooldown': 0,
    'red_detect_count': 0,
    'yellow_detect_count': 0,
    'trailing_detect_count': 0,
    'trailing_escape_cooldown': 0,
    'trailing_escape_send_timer': 0,
    'trailing_debug_tick': 0,
    'race_elapsed': 0.0,
    'race_phase': 0.0,
    'race_game_state': 'WAIT',  # WAIT / RUN / DONE
    'race_last_timestamp': None,
    'race_prev_motion_frame': None,
    'race_motion_score': 0.0,
    'race_motion_active': False,
    'race_move_frames': 0,
    'race_still_frames': 0,
    'debug_screenshot_count': 0
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

PROXIMITY_SCAN_TOP = 165
PROXIMITY_SCAN_BOTTOM = 185
PROXIMITY_PIXEL_THRESHOLD = 15

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
# Police Event Configuration
# ---------------------------------------------------------
TASK_PERIOD_SECONDS = 0.033

# ---------------------------------------------------------
# Game Day Race Clock Configuration
# ---------------------------------------------------------
RACE_DURATION_SECONDS = 180.0
RACE_PHASE_SECONDS = 60.0
RACE_MOTION_THRESHOLD = 1.15
RACE_MOTION_ENTER_FRAMES = 3
RACE_MOTION_EXIT_FRAMES = 8
RACE_CLOCK_MAX_DELTA = 0.20
RACE_AUTO_RESET_STILL_FRAMES = 90
RACE_AUTO_RESET_MIN_ELAPSED = 1.0
RACE_CLOCK_ROI_TOP = 78
RACE_CLOCK_ROI_BOTTOM = 190
RACE_CLOCK_ROI_LEFT = 42
RACE_CLOCK_ROI_RIGHT = 278
POLICE_PHASE_START_SECONDS = 28.0
POLICE_PHASE_END_SECONDS = 52.0
POLICE_EVENT_SECONDS = 5.0
POLICE_EVENT_FRAMES = max(1, int(POLICE_EVENT_SECONDS / TASK_PERIOD_SECONDS))
POLICE_CONFIRM_FRAMES = 2
POLICE_REARM_FRAMES = 30
POLICE_MIN_COLOR_AREA = 85
POLICE_MIN_COMBINED_AREA = 350
POLICE_MIN_BOTTOM_Y = 102
POLICE_MAX_CENTER_GAP_X = 120
POLICE_MAX_CENTER_GAP_Y = 70
POLICE_MIN_WIDTH = 36
POLICE_MIN_HEIGHT = 16
POLICE_MAX_ASPECT = 4.0
POLICE_MIN_FILL_RATIO = 0.08
POLICE_MIN_LANE_COVERAGE = 0.10
POLICE_STRONG_LANE_COVERAGE = 0.25
POLICE_BLUE_MAX_VALUE = 210
POLICE_DEBUG_DIR = 'police_debug'
POLICE_DEBUG_CAPTURE_COOLDOWN_FRAMES = 12
POLICE_DEBUG_CAPTURE_SCORE_MIN = 30.0
# Code-only switch for automatic police screenshots.
# True  = save a screenshot whenever the detector sees a police candidate.
# False = disable all automatic police screenshots for better FPS.
POLICE_DEBUG_CAPTURE_ENABLED = False
POLICE_LANE_GUIDE_DILATE_ITERS = 3
POLICE_MEMORY_FRAMES = 12
POLICE_MEMORY_MIN_SCORE = 9999.0
POLICE_CAPTURE_MIN_BOTTOM = 175
POLICE_CAPTURE_MAX_ERROR = 14
POLICE_MIN_VERTICAL_OVERLAP_RATIO = 0.02
POLICE_EVADE_BOTTOM_Y = 120
POLICE_EVADE_WIDTH = 40
POLICE_EVADE_CENTER_MARGIN = 30
POLICE_AVOID_HOLD_FRAMES = 14
POLICE_AVOID_ACCELERATION = 0.18
POLICE_HARD_DODGE_SCORE = 120.0
POLICE_HARD_DODGE_BOTTOM_Y = 112
POLICE_HARD_DODGE_WIDTH = 30
POLICE_HARD_DODGE_HOLD_FRAMES = 24
POLICE_HARD_DODGE_ACCELERATION = 0.68
POLICE_COLLISION_AVOID_HOLD_FRAMES = 18
POLICE_COLLISION_BRAKE_ACCELERATION = -0.35
POLICE_COLLISION_BOTTOM_Y = 122
POLICE_COLLISION_WIDTH = 22
POLICE_COLLISION_CENTER_MARGIN = 34
POLICE_INSTANT_SCORE = 50.0   # Much lower: very easy to trigger
POLICE_INSTANT_BOTTOM_Y = 85   # Much higher: detect further away
POLICE_INSTANT_WIDTH = 20      # Much smaller: detect tiny distant police
POLICE_INSTANT_HEIGHT = 10     # Much smaller: detect tiny distant police
POLICE_INSTANT_CENTER_MARGIN = 35  # More tolerance at distance
POLICE_EMERGENCY_SCORE = 110.0  # Much lower: easier detection
POLICE_EMERGENCY_BOTTOM_Y = 105  # Higher position for earlier detection
POLICE_EMERGENCY_WIDTH = 32     # Smaller detection size
POLICE_SUDDEN_SCORE = 75.0      # Much lower: easier trigger
POLICE_SUDDEN_BOTTOM_Y = 90     # Much higher: earlier detection
POLICE_SUDDEN_WIDTH = 24        # Smaller size
POLICE_SUDDEN_HEIGHT = 12       # Smaller size
POLICE_SUDDEN_CENTER_MARGIN = 35  # More tolerance
POLICE_BOX_PAD_X = 0.45
POLICE_BOX_PAD_TOP = 0.35
POLICE_BOX_PAD_BOTTOM = 0.55
POLICE_ROI_TOP = 100     
POLICE_ROI_BOTTOM = 220  
POLICE_ROI_LEFT = 12    
POLICE_ROI_RIGHT = 308    

POLICE_KERNEL = np.ones((5, 5), np.uint8)


# ---------------------------------------------------------
# NEW: Darkness Improvement Configuration
# ---------------------------------------------------------
# Enter/exit Darkness Mode only after several frames so it does not flicker.
DARKNESS_ENTER_THRESHOLD = 42.0
DARKNESS_ENTER_FRAMES = 2
DARKNESS_EXIT_THRESHOLD = 58.0
DARKNESS_EXIT_FRAMES = 10
DARKNESS_EMERGENCY_THRESHOLD = 24.0
DARKNESS_DROP_THRESHOLD = 16.0
DARKNESS_DROP_TARGET_THRESHOLD = 35.0

CRITICAL_DARKNESS_THRESHOLD = 35.0

DARKNESS_ACCELERATION = 0.35
CRITICAL_DARKNESS_ACCELERATION = 0.15
LIGHT_TOGGLE_ACCELERATION = -1.0
LIGHT_TOGGLE_FRAMES = 6
NORMAL_TAP_FRAMES = 14          # Increased: Stronger steering duration
DARKNESS_TAP_FRAMES = 11        # Increased: Longer dodge in darkness
TRAILING_TAP_FRAMES = 24
TRAILING_CONFIRM_FRAMES = 1
TRAILING_ESCAPE_COOLDOWN_FRAMES = 45
NORMAL_COOLDOWN_FRAMES = 8      # Reduced: Faster recovery for repeated dodges
DARKNESS_COOLDOWN_FRAMES = 12
TRAILING_DEBUG_VERSION = "trailing-v5-state"

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
    Detect EV1 Darkness reliably.

    Fix: when the camera brightness suddenly drops from around 35+ to 10+,
    do not wait for several frames. Enter Darkness Mode immediately and keep
    sending acceleration_input = -1.0 for enough frames for Unity to open light.
    """
    with data_lock:
        previous_brightness = float(shared_data.get('front_brightness', brightness))
        brightness = float(brightness)
        brightness_drop = previous_brightness - brightness

        shared_data['last_front_brightness'] = previous_brightness
        shared_data['front_brightness'] = brightness

        instant_darkness = (
            brightness <= DARKNESS_EMERGENCY_THRESHOLD
            or (
                previous_brightness >= DARKNESS_DROP_TARGET_THRESHOLD
                and brightness <= DARKNESS_DROP_TARGET_THRESHOLD
                and brightness_drop >= DARKNESS_DROP_THRESHOLD
            )
        )

        if not shared_data['low_brightness']:
            shared_data['darkness_exit_count'] = 0

            if instant_darkness:
                shared_data['darkness_enter_count'] = DARKNESS_ENTER_FRAMES
            elif brightness < DARKNESS_ENTER_THRESHOLD:
                shared_data['darkness_enter_count'] += 1
            else:
                shared_data['darkness_enter_count'] = 0

            if shared_data['darkness_enter_count'] >= DARKNESS_ENTER_FRAMES:
                shared_data['low_brightness'] = True
                shared_data['darkness_enter_count'] = 0
                set_vehicle_light(True, brightness)
                shared_data['light_signal_timer'] = max(
                    shared_data['light_signal_timer'],
                    LIGHT_TOGGLE_FRAMES
                )
                print(
                    f"Darkness Mode ON | brightness={brightness:.1f} | "
                    f"prev={previous_brightness:.1f} | drop={brightness_drop:.1f}"
                )
        else:
            shared_data['darkness_enter_count'] = 0

            # Do not keep resetting light_signal_timer while it is dark.
            # If we reset it every frame, acceleration_input stays -1.0 forever
            # and the car will reverse.

            if brightness > DARKNESS_EXIT_THRESHOLD:
                shared_data['darkness_exit_count'] += 1
            else:
                shared_data['darkness_exit_count'] = 0

            if shared_data['darkness_exit_count'] >= DARKNESS_EXIT_FRAMES:
                shared_data['low_brightness'] = False
                shared_data['darkness_exit_count'] = 0
                shared_data['light_signal_timer'] = 0
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


def extract_race_motion_frame(small_frame):
    """Return a compact road ROI used only to decide whether the simulator moves."""
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    roi = gray[
        RACE_CLOCK_ROI_TOP:RACE_CLOCK_ROI_BOTTOM,
        RACE_CLOCK_ROI_LEFT:RACE_CLOCK_ROI_RIGHT,
    ]
    roi = cv2.resize(roi, (96, 48), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(roi, (5, 5), 0)


def update_race_clock_from_frame(small_frame):
    """
    Advance the 180s Game Day clock only while the simulator image is moving.
    If the simulator stays stopped after a run/reset, automatically restart
    the Python race clock so the next moving run begins from 0.0s.
    """
    now = time.monotonic()
    current_motion_frame = extract_race_motion_frame(small_frame)

    with data_lock:
        previous_motion_frame = shared_data['race_prev_motion_frame']
        last_timestamp = shared_data['race_last_timestamp']
        elapsed = shared_data['race_elapsed']
        motion_active = shared_data['race_motion_active']
        move_frames = shared_data['race_move_frames']
        still_frames = shared_data['race_still_frames']

    if previous_motion_frame is None:
        motion_score = 0.0
        raw_moving = False
    else:
        motion_score = float(
            np.mean(cv2.absdiff(previous_motion_frame, current_motion_frame))
        )
        raw_moving = motion_score >= RACE_MOTION_THRESHOLD

    if raw_moving:
        move_frames += 1
        still_frames = 0
    else:
        still_frames += 1
        move_frames = 0

    if not motion_active and move_frames >= RACE_MOTION_ENTER_FRAMES:
        motion_active = True
    elif motion_active and still_frames >= RACE_MOTION_EXIT_FRAMES:
        motion_active = False

    if last_timestamp is None:
        dt = 0.0
    else:
        dt = max(0.0, min(now - last_timestamp, RACE_CLOCK_MAX_DELTA))

    if elapsed < RACE_DURATION_SECONDS and motion_active:
        elapsed = min(RACE_DURATION_SECONDS, elapsed + dt)

    if elapsed >= RACE_DURATION_SECONDS:
        game_state = 'DONE'
        motion_active = False
    elif motion_active:
        game_state = 'RUN'
    else:
        game_state = 'WAIT'

    auto_reset_due = (
        elapsed >= RACE_AUTO_RESET_MIN_ELAPSED
        and not motion_active
        and still_frames >= RACE_AUTO_RESET_STILL_FRAMES
    )

    if auto_reset_due:
        elapsed = 0.0
        phase = 0.0
        game_state = 'WAIT'
        motion_active = False
        move_frames = 0
        still_frames = 0
    else:
        phase = elapsed % RACE_PHASE_SECONDS

    with data_lock:
        if auto_reset_due:
            # Reset Python-side driving/event state only after the simulator has
            shared_data.update({
                'steering_input': 0.0,
                'acceleration_input': 1.0,
                'target_lane': 1,
                'current_lane': 1,
                'low_brightness': False,
                'front_brightness': 255.0,
                'darkness_enter_count': 0,
                'darkness_exit_count': 0,
                'lights_on': False,
                'light_signal_timer': 0,
                'tap_timer': 0,
                'cooldown_timer': 0,
                'tap_steering': 0.0,
                'police_active': False,
                'police_timer': 0,
                'police_detect_count': 0,
                'police_rearm_timer': 0,
                'police_memory_timer': 0,
                'police_memory_score': 0.0,
                'police_last_position': None,
                'police_velocity': None,
                'police_avoid_timer': 0,
                'police_avoid_steering': 0.0,
                'police_avoid_acceleration': POLICE_AVOID_ACCELERATION,
                'police_red_attempted': False,
                'police_debug_capture_cooldown': 0,
                'red_detect_count': 0,
                'yellow_detect_count': 0,
                'trailing_detect_count': 0,
                'trailing_escape_cooldown': 0,
                'trailing_escape_send_timer': 0,
                'trailing_debug_tick': 0,
            })

        shared_data['race_prev_motion_frame'] = current_motion_frame
        shared_data['race_last_timestamp'] = now
        shared_data['race_elapsed'] = elapsed
        shared_data['race_phase'] = phase
        shared_data['race_game_state'] = game_state
        shared_data['race_motion_active'] = motion_active
        shared_data['race_motion_score'] = motion_score
        shared_data['race_move_frames'] = move_frames
        shared_data['race_still_frames'] = still_frames

    if auto_reset_due:
        print(
            "Race clock auto reset after simulator stopped. "
            "Next moving run will start from 0.0s."
        )

    return game_state, elapsed, phase


def reset_runtime_state(reason='manual'):
    """
    Reset Python-side timers/states after restarting the simulator.
    Screenshot enable/disable is controlled only by POLICE_DEBUG_CAPTURE_ENABLED.
    """
    with data_lock:
        screenshot_count = shared_data.get('debug_screenshot_count', 0)

        shared_data.update({
            # Control state
            'steering_input': 0.0,
            'acceleration_input': 1.0,
            'target_lane': 1,
            'current_lane': 1,

            # Darkness / light state
            'low_brightness': False,
            'front_brightness': 255.0,
            'darkness_enter_count': 0,
            'darkness_exit_count': 0,
            'lights_on': False,
            'light_signal_timer': 0,

            # Steering tap / cooldown state
            'tap_timer': 0,
            'cooldown_timer': 0,
            'tap_steering': 0.0,

            # Police event state
            'police_active': False,
            'police_timer': 0,
            'police_detect_count': 0,
            'police_rearm_timer': 0,
            'police_memory_timer': 0,
            'police_memory_score': 0.0,
            'police_last_position': None,
            'police_velocity': None,
            'police_avoid_timer': 0,
            'police_avoid_steering': 0.0,
            'police_avoid_acceleration': POLICE_AVOID_ACCELERATION,
            'police_red_attempted': False,
            'police_debug_capture_cooldown': 0,

            # Token / rear-car state
            'red_detect_count': 0,
            'yellow_detect_count': 0,
            'trailing_detect_count': 0,
            'trailing_escape_cooldown': 0,
            'trailing_escape_send_timer': 0,
            'trailing_debug_tick': 0,

            # 180s Game Day race clock state
            'race_elapsed': 0.0,
            'race_phase': 0.0,
            'race_game_state': 'WAIT',
            'race_last_timestamp': None,
            'race_prev_motion_frame': None,
            'race_motion_score': 0.0,
            'race_motion_active': False,
            'race_move_frames': 0,
            'race_still_frames': 0,

        })

        # Preserve total saved screenshot count only.
        shared_data['debug_screenshot_count'] = screenshot_count

    print(f"Python runtime state reset ({reason}). game=WAIT 0.0s | phase=0.0s")


def evaluate_trailing_signal(
    avg_intensity,
    contrast,
    edge_density,
    lower_edge_density,
    upper_edge_density,
    largest_edge_blob,
    lowest_blob_bottom,
):
    """Return (signal, reason) for rear-car danger from compact ROI metrics."""
    triggers = {
        'NORMAL': (
            contrast > 34
            and lower_edge_density > 0.070
            and lower_edge_density > upper_edge_density * 1.10
            and largest_edge_blob > 1400
            and lowest_blob_bottom > 92
        ),
        'HUGE_APPROACH': (
            largest_edge_blob > 8000
            and lowest_blob_bottom > 85
            and edge_density > 0.040
        ),
        'DIM_CLOSE': (
            avg_intensity < 35
            and contrast > 28
            and largest_edge_blob > 4200
            and lowest_blob_bottom > 100
        ),
        'WEAK_CLOSE': (
            contrast > 38
            and edge_density > 0.018
            and largest_edge_blob > 900
            and lowest_blob_bottom > 112
        ),
        'VERY_DARK': (
            avg_intensity < 14
            and largest_edge_blob > 650
            and lowest_blob_bottom > 96
        ),
        'APPROACH': (
            contrast > 32
            and edge_density > 0.035
            and largest_edge_blob > 5000
            and 78 < lowest_blob_bottom <= 112
        ),
    }

    for reason, is_triggered in triggers.items():
        if is_triggered:
            return True, reason

    return False, 'NONE'


def choose_trailing_escape_direction(current_lane):
    """Choose one lane-change direction for a confirmed rear-car event."""
    if current_lane >= 2:
        return -1.0
    return 1.0


def detect_front_police_car(base_frame, road_mask):
    """
    Detect the front police car as one large object that contains both
    prominent red and blue regions inside the road area.
    """
    hsv_frame = cv2.cvtColor(base_frame, cv2.COLOR_BGR2HSV)

    # Aggressive HSV ranges: catch police cars in all lighting conditions
    police_red_mask = cv2.bitwise_or(
        cv2.inRange(hsv_frame, np.array([0, 60, 50]), np.array([15, 255, 255])),
        cv2.inRange(hsv_frame, np.array([165, 60, 50]), np.array([180, 255, 255])),
    )
    blue_mask = cv2.inRange(
        hsv_frame,
        np.array([100, 110, 80]),
        np.array([140, 255, POLICE_BLUE_MAX_VALUE]),
    )

    # Detect the grey asphalt/lane surface instead of the whole road trapezoid.
    lane_mask = cv2.inRange(
        hsv_frame,
        np.array([0, 0, 28]),
        np.array([180, 105, 205]),
    )

    police_roi_mask = road_mask.copy()
    police_roi_mask[:POLICE_ROI_TOP, :] = 0
    police_roi_mask[POLICE_ROI_BOTTOM:, :] = 0
    police_roi_mask[:, :POLICE_ROI_LEFT] = 0
    police_roi_mask[:, POLICE_ROI_RIGHT:] = 0

    lane_mask = cv2.bitwise_and(lane_mask, police_roi_mask)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, POLICE_KERNEL)
    lane_mask = cv2.dilate(lane_mask, POLICE_KERNEL, iterations=1)
    lane_guide_mask = cv2.dilate(
        lane_mask,
        POLICE_KERNEL,
        iterations=POLICE_LANE_GUIDE_DILATE_ITERS,
    )

    police_red_mask = cv2.bitwise_and(police_red_mask, police_roi_mask)
    police_blue_mask = cv2.bitwise_and(blue_mask, lane_guide_mask)

    # Enhanced morphological operations: remove noise and connect fragments
    police_red_mask = cv2.morphologyEx(police_red_mask, cv2.MORPH_CLOSE, POLICE_KERNEL)
    police_blue_mask = cv2.morphologyEx(police_blue_mask, cv2.MORPH_CLOSE, POLICE_KERNEL)
    police_red_mask = cv2.morphologyEx(police_red_mask, cv2.MORPH_OPEN, POLICE_KERNEL)
    police_blue_mask = cv2.morphologyEx(police_blue_mask, cv2.MORPH_OPEN, POLICE_KERNEL)
    # Additional erosion to remove small noise pixels
    police_red_mask = cv2.erode(police_red_mask, POLICE_KERNEL, iterations=1)
    police_blue_mask = cv2.erode(police_blue_mask, POLICE_KERNEL, iterations=1)

    red_contours, _ = cv2.findContours(
        police_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    blue_contours, _ = cv2.findContours(
        police_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    large_reds = [
        contour for contour in red_contours
        if cv2.contourArea(contour) >= POLICE_MIN_COLOR_AREA
    ]
    large_blues = [
        contour for contour in blue_contours
        if cv2.contourArea(contour) >= POLICE_MIN_COLOR_AREA
    ]

    debug_info = {
        'seen': False,
        'red_boxes': [cv2.boundingRect(contour) for contour in large_reds],
        'blue_boxes': [cv2.boundingRect(contour) for contour in large_blues],
        'candidate_box': None,
        'raw_candidate_box': None,
        'combined_area': 0,
        'fill_ratio': 0.0,
        'score': 0.0,
        'lane_coverage': 0.0,
        'roi_box': (
            POLICE_ROI_LEFT,
            POLICE_ROI_TOP,
            POLICE_ROI_RIGHT - POLICE_ROI_LEFT,
            POLICE_ROI_BOTTOM - POLICE_ROI_TOP,
        ),
    }

    best_candidate = None
    combined_mask = cv2.bitwise_or(police_red_mask, police_blue_mask)
    roi_center_x = (POLICE_ROI_LEFT + POLICE_ROI_RIGHT) // 2

    for red_contour in large_reds:
        rx, ry, rw, rh = cv2.boundingRect(red_contour)
        red_center = (rx + (rw // 2), ry + (rh // 2))
        red_bottom = ry + rh

        for blue_contour in large_blues:
            bx, by, bw, bh = cv2.boundingRect(blue_contour)
            blue_center = (bx + (bw // 2), by + (bh // 2))
            blue_bottom = by + bh

            if abs(red_center[0] - blue_center[0]) > POLICE_MAX_CENTER_GAP_X:
                continue
            if abs(red_center[1] - blue_center[1]) > POLICE_MAX_CENTER_GAP_Y:
                continue
            if max(red_bottom, blue_bottom) < POLICE_MIN_BOTTOM_Y:
                continue

            # NEW: Prefer red-on-top, blue-on-bottom pattern (typical police car)
            # But don't reject if pattern is reversed - allows flexibility
            red_on_top = ry <= by  # Bonus if red is truly above blue

            vertical_overlap = max(
                0,
                min(ry + rh, by + bh) - max(ry, by),
            )
            min_color_height = max(1, min(rh, bh))
            overlap_ratio = vertical_overlap / float(min_color_height)
            if overlap_ratio < POLICE_MIN_VERTICAL_OVERLAP_RATIO:
                continue

            union_left = min(rx, bx)
            union_top = min(ry, by)
            union_right = max(rx + rw, bx + bw)
            union_bottom = max(ry + rh, by + bh)
            union_width = union_right - union_left
            union_height = union_bottom - union_top

            if union_width < POLICE_MIN_WIDTH or union_height < POLICE_MIN_HEIGHT:
                continue

            aspect_ratio = union_width / max(1, union_height)
            if aspect_ratio > POLICE_MAX_ASPECT:
                continue

            union_patch = combined_mask[union_top:union_bottom, union_left:union_right]
            if union_patch.size == 0:
                continue

            filled_area = int(cv2.countNonZero(union_patch))
            fill_ratio = filled_area / float(union_width * union_height)
            lane_patch = lane_mask[union_top:union_bottom, union_left:union_right]
            if lane_patch.size == 0:
                continue

            lane_coverage = cv2.countNonZero(lane_patch) / float(union_width * union_height)
            if (
                filled_area < POLICE_MIN_COMBINED_AREA
                or fill_ratio < POLICE_MIN_FILL_RATIO
                or lane_coverage < POLICE_MIN_LANE_COVERAGE
            ):
                continue

            candidate_center_x = union_left + (union_width // 2)
            center_offset = abs(candidate_center_x - roi_center_x)
            color_balance = min(cv2.contourArea(red_contour), cv2.contourArea(blue_contour))
            
            # Balanced scoring: prioritize detection while using fill_ratio for noise rejection
            red_blue_vertical_distance = abs(red_center[1] - blue_center[1])
            red_above_bonus = max(0, 20 - red_blue_vertical_distance) * 5.0 if red_on_top else 0  # Modest bonus for proper pattern
            
            score = (
                filled_area * 1.2  # Substantial detection priority
                + fill_ratio * 140.0  # Prefer solid objects but not over-penalizing
                + max(red_bottom, blue_bottom) * 3.8
                + color_balance * 0.40
                + lane_coverage * 280.0  # Road presence is important
                + red_above_bonus  # Reward proper red-blue stacking
                - center_offset * 1.6
                - red_blue_vertical_distance * 2.8  # Reasonable alignment penalty
            )

            if lane_coverage < POLICE_STRONG_LANE_COVERAGE:
                score -= (POLICE_STRONG_LANE_COVERAGE - lane_coverage) * 180.0  # Standard penalty

            pad_x = max(16, int(union_width * POLICE_BOX_PAD_X))
            pad_top = max(10, int(union_height * POLICE_BOX_PAD_TOP))
            pad_bottom = max(14, int(union_height * POLICE_BOX_PAD_BOTTOM))

            expanded_left = max(POLICE_ROI_LEFT, union_left - pad_x)
            expanded_top = max(POLICE_ROI_TOP, union_top - pad_top)
            expanded_right = min(POLICE_ROI_RIGHT, union_right + pad_x)
            expanded_bottom = min(POLICE_ROI_BOTTOM, union_bottom + pad_bottom)
            expanded_box = (
                expanded_left,
                expanded_top,
                expanded_right - expanded_left,
                expanded_bottom - expanded_top,
            )

            candidate = {
                'score': score,
                'raw_box': (union_left, union_top, union_width, union_height),
                'expanded_box': expanded_box,
                'combined_area': filled_area,
                'fill_ratio': fill_ratio,
                'lane_coverage': lane_coverage,
            }

            if best_candidate is None or score > best_candidate['score']:
                best_candidate = candidate

    if best_candidate is not None:
        debug_info['seen'] = True
        debug_info['candidate_box'] = best_candidate['expanded_box']
        debug_info['raw_candidate_box'] = best_candidate['raw_box']
        debug_info['combined_area'] = best_candidate['combined_area']
        debug_info['fill_ratio'] = best_candidate['fill_ratio']
        debug_info['score'] = best_candidate['score']
        debug_info['lane_coverage'] = best_candidate['lane_coverage']
        return True, debug_info

    return False, debug_info


def boxes_overlap(box_a, box_b, pad=0):
    """Return True when two bounding boxes overlap. Used to ignore police-body red pixels."""
    if box_a is None or box_b is None:
        return False
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax1, ay1 = ax - pad, ay - pad
    ax2, ay2 = ax + aw + pad, ay + ah + pad
    bx1, by1 = bx, by
    bx2, by2 = bx + bw, by + bh
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def is_strong_police_event_candidate(police_debug):
    """
    Strict gate for starting the formal EV2 5s red-token mission.
    policeTime=ON only means the event can happen; police=ON starts only when
    a strong red+blue police-car candidate is visible.
    """
    if not police_debug.get('seen'):
        return False

    raw_box = police_debug.get('raw_candidate_box')
    candidate_box = police_debug.get('candidate_box')
    if raw_box is None or candidate_box is None:
        return False

    _, raw_y, raw_w, raw_h = raw_box
    _, cand_y, cand_w, cand_h = candidate_box
    score = float(police_debug.get('score', 0.0))
    area = int(police_debug.get('combined_area', 0))
    fill = float(police_debug.get('fill_ratio', 0.0))
    lane = float(police_debug.get('lane_coverage', 0.0))
    cand_bottom = cand_y + cand_h
    raw_bottom = raw_y + raw_h

    # Main gate: strong enough for real police-car detection but still catches
    # fast/near police cars like score≈360, area≈360, fill≈0.34.
    strong_shape = (
        score >= 130.0
        and area >= 250
        and fill >= 0.10
        and lane >= 0.10
        and cand_bottom >= 104
        and raw_bottom >= 100
        and cand_w >= 30
        and raw_w >= 20
        and raw_h >= 10
    )

    # Backup gate for very confident but partially clipped close police cars.
    very_confident_close = (
        score >= 220.0
        and area >= 160
        and fill >= 0.08
        and lane >= 0.08
        and cand_bottom >= 96
        and raw_w >= 18
        and raw_h >= 8
    )

    return strong_shape or very_confident_close


def choose_police_red_token(filtered_red_contours, police_raw_box, police_candidate_box, frame_center):
    """
    Pick the real EV2 red token during police=ON.
    Important: do not use largest_red. That often chooses police lights,
    red road shoulder, or red noise. This function rejects red contours that
    overlap the police box and scores token-like blobs by bottom/area/path.
    """
    best = None

    for contour in filtered_red_contours:
        area = cv2.contourArea(contour)
        if area <= 5:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        box = (x, y, w, h)

        # Red pixels inside/near the police bbox are part of the police car,
        # not the red token we need to catch.
        if boxes_overlap(box, police_raw_box, pad=10):
            continue
        if boxes_overlap(box, police_candidate_box, pad=8):
            continue

        M = cv2.moments(contour)
        if M['m00'] <= 0:
            continue

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        bottom = y + h
        center_error = abs(cx - frame_center)

        score = (
            area * 0.85
            + bottom * 2.25
            + h * 1.30
            - center_error * 0.35
            - max(0, 95 - bottom) * 1.4
        )

        item = {
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': cx,
            'cy': cy,
            'bottom': bottom,
            'area': area,
            'score': score,
        }

        if best is None or item['score'] > best['score']:
            best = item

    return best


def save_police_debug_capture(debug_frame, police_debug, reason):
    """Save an automatic police debug screenshot when police detection triggers."""
    if debug_frame is None:
        return

    os.makedirs(POLICE_DEBUG_DIR, exist_ok=True)
    now = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    milliseconds = int((now % 1) * 1000)
    score = int(float(police_debug.get('score', 0.0)))
    base_name = f"{timestamp}_{milliseconds:03d}_{reason}_s{score}"
    debug_path = os.path.join(POLICE_DEBUG_DIR, f"{base_name}_debug.png")

    if cv2.imwrite(debug_path, debug_frame):
        with data_lock:
            shared_data['debug_screenshot_count'] += 1
        print(f"Police debug capture saved: {debug_path}")



def update_police_state(police_seen, police_score=0.0, red_token_captured=False, police_position=None, allow_activation=True):
    """
    Keep the police event self-contained so the rest of the driving logic
    does not need to change.
    NEW: Added trajectory prediction for high-speed scenarios to bridge detection gaps.
    """
    with data_lock:
        # NEW: Update trajectory for prediction at high speed
        if police_position is not None and police_seen:
            if shared_data['police_last_position'] is not None:
                # Calculate simple velocity vector
                last_x, last_y = shared_data['police_last_position']
                curr_x, curr_y = police_position
                shared_data['police_velocity'] = (curr_x - last_x, curr_y - last_y)
            shared_data['police_last_position'] = police_position
        elif not police_seen and shared_data['police_last_position'] is not None:
            # Extrapolate position using velocity to bridge detection gaps in high-speed
            if shared_data['police_active'] and shared_data['police_velocity'] is not None:
                vx, vy = shared_data['police_velocity']
                # If we lose detection but velocity suggests police is still in frame, keep active
                if abs(vx) > 1 or abs(vy) > 1:
                    police_seen = True
                    police_score = shared_data['police_memory_score']
        
        if shared_data['police_rearm_timer'] > 0:
            shared_data['police_rearm_timer'] -= 1

        if red_token_captured:
            if shared_data['police_active']:
                shared_data['police_red_attempted'] = True
                print("Police red token reached; keeping police=ON until timer ends.")
                return True
            return False

        if shared_data['police_active']:
            if shared_data['police_timer'] > 0:
                shared_data['police_timer'] -= 1
            else:
                shared_data['police_active'] = False
                shared_data['police_detect_count'] = 0
                shared_data['police_rearm_timer'] = POLICE_REARM_FRAMES
                shared_data['police_memory_timer'] = 0
                shared_data['police_memory_score'] = 0.0
                shared_data['police_red_attempted'] = False
                print(f"Police Event ended after {POLICE_EVENT_SECONDS:.0f} seconds.")
            return shared_data['police_active']

        if shared_data['police_rearm_timer'] > 0:
            shared_data['police_detect_count'] = 0
            shared_data['police_memory_timer'] = 0
            shared_data['police_memory_score'] = 0.0
            return False

        # 30s-50s phase window (with margin). Detection may still run
        if not allow_activation:
            shared_data['police_detect_count'] = 0
            shared_data['police_memory_timer'] = 0
            shared_data['police_memory_score'] = 0.0
            return False

        if police_seen:
            shared_data['police_memory_timer'] = POLICE_MEMORY_FRAMES
            shared_data['police_memory_score'] = max(
                shared_data['police_memory_score'],
                float(police_score),
            )
            shared_data['police_detect_count'] += 1
            if shared_data['police_detect_count'] >= POLICE_CONFIRM_FRAMES:
                shared_data['police_active'] = True
                shared_data['police_timer'] = POLICE_EVENT_FRAMES
                shared_data['police_detect_count'] = 0
                shared_data['police_memory_timer'] = 0
                shared_data['police_memory_score'] = 0.0
                shared_data['police_red_attempted'] = False
                print("Police Event detected in front camera.")
        else:
            if shared_data['police_memory_timer'] > 0:
                shared_data['police_memory_timer'] -= 1
                if shared_data['police_detect_count'] > 0:
                    shared_data['police_detect_count'] = 1
                shared_data['police_memory_score'] *= 0.92
            else:
                shared_data['police_detect_count'] = 0
                shared_data['police_memory_score'] = 0.0

        return shared_data['police_active']

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
        edge_density = 0.0
        lower_edge_density = 0.0
        upper_edge_density = 0.0
        contours = []

        # Brightness alone was too sensitive, so require stronger brightness + contrast.
        if (avg_intensity > 225 or avg_intensity < 25) and contrast > 18:
        
            blurred_back_roi = cv2.GaussianBlur(back_roi, (5, 5), 0)
            rear_edges = cv2.Canny(blurred_back_roi, 45, 135)
            edge_density = cv2.countNonZero(rear_edges) / float(back_roi.size)
            lower_edges = rear_edges[60:120, :]
            upper_edges = rear_edges[0:60, :]
            lower_edge_density = cv2.countNonZero(lower_edges) / float(lower_edges.size)
            upper_edge_density = cv2.countNonZero(upper_edges) / float(upper_edges.size)

            edge_kernel = np.ones((5, 5), np.uint8)
            connected_edges = cv2.dilate(rear_edges, edge_kernel, iterations=2)
            contours, _ = cv2.findContours(
                connected_edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
        )

        largest_edge_blob = 0
        lowest_blob_bottom = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / float(max(h, 1))
            if 0.45 <= aspect_ratio <= 4.5 and area > largest_edge_blob:
                largest_edge_blob = area
                lowest_blob_bottom = y + h

        rear_car_signal, rear_signal_reason = evaluate_trailing_signal(
            avg_intensity,
            contrast,
            edge_density,
            lower_edge_density,
            upper_edge_density,
            largest_edge_blob,
            lowest_blob_bottom,
        )

        with data_lock:
            if shared_data['trailing_escape_cooldown'] > 0:
                shared_data['trailing_escape_cooldown'] -= 1

            if rear_car_signal:
                shared_data['trailing_detect_count'] += 1
            else:
                shared_data['trailing_detect_count'] = 0

            shared_data['trailing_debug_tick'] = (shared_data['trailing_debug_tick'] + 1) % 30
            should_print_watch = shared_data['trailing_debug_tick'] == 0
            confirmed_rear_car = (
                shared_data['trailing_detect_count'] >= TRAILING_CONFIRM_FRAMES
                and shared_data['trailing_escape_cooldown'] == 0
            )

        if should_print_watch:
            print(
                "Trailing Watch: "
                f"{TRAILING_DEBUG_VERSION} "
                f"avg={avg_intensity:.1f}, contrast={contrast:.1f}, "
                f"edges={edge_density:.3f}, lower={lower_edge_density:.3f}, "
                f"blob={largest_edge_blob:.0f}, bottom={lowest_blob_bottom}, "
                f"reason={rear_signal_reason}, "
                f"signal={rear_car_signal}, cooldown={shared_data['trailing_escape_cooldown']}"
            )

        if confirmed_rear_car:
            evade_back_car = True
            # Check which side has more space or default a structural escape jump
            back_steering_escape = 1.0  # Tap right to clear the lane [cite: 153, 156]
            with data_lock:
                current_lane_for_escape = shared_data.get('current_lane', 1)
            back_steering_escape = choose_trailing_escape_direction(current_lane_for_escape)
            print(
                "Trailing Car Alert! "
                f"{TRAILING_DEBUG_VERSION} "
                f"avg={avg_intensity:.1f}, contrast={contrast:.1f}, "
                f"edges={edge_density:.3f}, lower={lower_edge_density:.3f}, "
                f"blob={largest_edge_blob:.0f}, bottom={lowest_blob_bottom}, "
                f"reason={rear_signal_reason}, steering={back_steering_escape}"
            )

    # ---------------------------------------------------------
    # FRONT CAMERA ENVIRONMENT ANALYSIS (Token Processing)
    # ---------------------------------------------------------
    if front_frame is not None:
        small_frame = cv2.resize(front_frame, (320, 240))
        game_state, race_elapsed, race_phase = update_race_clock_from_frame(small_frame)
        
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

        police_seen, police_debug = detect_front_police_car(small_frame, road_mask)
        police_phase_window = (
            POLICE_PHASE_START_SECONDS <= race_phase <= POLICE_PHASE_END_SECONDS
        )
        police_position = None
        if police_debug.get('raw_candidate_box') is not None:
            x, y, w, h = police_debug['raw_candidate_box']
            police_position = (x + w // 2, y + h)  # (center_x, bottom_y)

        police_event_seen = (
            police_phase_window
            and is_strong_police_event_candidate(police_debug)
        )
        police_mode = update_police_state(
            police_event_seen,
            police_score=police_debug.get('score', 0.0),
            police_position=police_position if police_event_seen else None,
            allow_activation=police_phase_window,
        )

        # NEW: Darkness Improvement - reconnect small broken token regions.
        if low_brightness:
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, DARK_KERNEL)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, DARK_KERNEL)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, DARK_KERNEL)
        
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_red_contours = [
            contour for contour in contours_red
            if is_token_like_contour(contour, 'R')
        ]

        # ---------------------------------------------------------
        # FIXED: Proximity Scan Line - 5 Bucket Bounding Boxes
        # ---------------------------------------------------------
        scan_y_center = (PROXIMITY_SCAN_TOP + PROXIMITY_SCAN_BOTTOM) // 2
        left_edge, right_edge = road_edges_at_y(scan_y_center)
        
        # Pull inward slightly to avoid picking up outer grass borders/sidewalk fragments
        buffer_margin = 8  
        left_edge += buffer_margin
        right_edge -= buffer_margin
        
        lane_width = (right_edge - left_edge) / float(NUM_LANES)
        
        scan_boxes = []
        lane_states = ['Clear'] * NUM_LANES
        
        for i in range(NUM_LANES):
            x1 = int(left_edge + i * lane_width)
            x2 = int(left_edge + (i + 1) * lane_width)
            scan_boxes.append((x1, PROXIMITY_SCAN_TOP, x2, PROXIMITY_SCAN_BOTTOM))
            
            red_roi = mask_red[PROXIMITY_SCAN_TOP:PROXIMITY_SCAN_BOTTOM, x1:x2]
            yellow_roi = mask_yellow[PROXIMITY_SCAN_TOP:PROXIMITY_SCAN_BOTTOM, x1:x2]
            green_roi = mask_green[PROXIMITY_SCAN_TOP:PROXIMITY_SCAN_BOTTOM, x1:x2]
            
            # --- Anti-Grass Noise Validation ---
            # Instead of a blind pixel count, check if the matching pixels form a structural object 
            # rather than a thin, scattered edge slice or large ground mass.
            valid_green = False
            if cv2.countNonZero(green_roi) > PROXIMITY_PIXEL_THRESHOLD:
                # Find local contours within this lane segment slice to verify shape integrity
                sub_contours, _ = cv2.findContours(green_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for sc in sub_contours:
                    sc_area = cv2.contourArea(sc)
                    # Legitimate tokens at this perspective scale have structured footprint areas
                    if 6 <= sc_area <= 450:
                        valid_green = True
                        break

            valid_danger = False
            if cv2.countNonZero(red_roi) > PROXIMITY_PIXEL_THRESHOLD or cv2.countNonZero(yellow_roi) > PROXIMITY_PIXEL_THRESHOLD:
                # Validate Danger tokens similarly to ensure it's not the curb track border line splitting
                combined_danger_mask = cv2.bitwise_or(red_roi, yellow_roi)
                sub_contours_d, _ = cv2.findContours(combined_danger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for sc in sub_contours_d:
                    if 6 <= cv2.contourArea(sc) <= 450:
                        valid_danger = True
                        break
            
            # Assign states based on filtered structural results
            if valid_danger:
                lane_states[i] = 'Danger'
            elif not critical_darkness and valid_green:
                lane_states[i] = 'Target'
        police_candidate_box = police_debug.get('candidate_box')
        police_raw_box = police_debug.get('raw_candidate_box')
        police_detected_now = False
        police_obstacle_close = False
        police_emergency_close = False
        police_sudden_risk = False
        police_alert_active = False
        police_collision_imminent = False
        police_hard_immediate_dodge = False
        police_escape_steering = 0.0

        if police_raw_box is not None:
            if police_candidate_box is not None:
                px, py, pw, ph = police_candidate_box
            else:
                px, py, pw, ph = police_raw_box

            rx, ry, rw, rh = police_raw_box
            police_score = float(police_debug.get('score', 0.0))

            police_center_x = px + (pw // 2)
            police_bottom = py + ph
            police_left = px
            police_right = px + pw

            ego_lane = pixel_x_to_debug_lane(frame_center, police_bottom)
            police_lane_left = pixel_x_to_debug_lane(police_left, police_bottom)
            police_lane_right = pixel_x_to_debug_lane(police_right, police_bottom)
            police_lane_min = min(police_lane_left, police_lane_right)
            police_lane_max = max(police_lane_left, police_lane_right)
            police_center_error = police_center_x - frame_center
            police_center_margin = max(34, min(72, (pw // 2) + 26))
            police_center_in_corridor = abs(police_center_error) <= police_center_margin
            police_blocks_ego_lane = (
                police_lane_min <= ego_lane <= police_lane_max
                or police_center_in_corridor
            )

            police_alert_active = (
                police_blocks_ego_lane
                and police_bottom >= 72
                and police_score >= 35
            )

            police_escape_steering = -1.0 if police_center_x >= frame_center else 1.0

            # HARD POLICE DODGE:
            if (
                police_phase_window
                and police_score >= POLICE_HARD_DODGE_SCORE
                and police_bottom >= POLICE_HARD_DODGE_BOTTOM_Y
                and pw >= POLICE_HARD_DODGE_WIDTH
            ):
                police_hard_immediate_dodge = True
                police_escape_steering = 1.0 if police_center_x < frame_center else -1.0
                print(
                    "HARD Police Immediate Dodge Trigger! "
                    f"bottom={police_bottom}, width={pw}, score={police_score:.0f}, "
                    f"err={police_center_error}, steering={police_escape_steering}"
                )

            if (
                police_score >= 35
                and police_bottom >= 108
                and pw >= 24
                and police_center_in_corridor
            ):
                police_collision_imminent = True


            if not police_collision_imminent and police_blocks_ego_lane and police_bottom >= 74 and police_score >= 35:
                police_detected_now = True
                print(
                    "CRITICAL: Police in corridor! "
                    f"bottom={police_bottom}, width={pw}, score={police_score:.0f}, "
                    f"err={police_center_error}, steering={police_escape_steering}"
                )
            elif police_blocks_ego_lane and police_bottom >= 66 and police_score >= 35:
                # Pre-emptive slight steering if still far away.
                steering_target = -0.5 if police_center_x >= frame_center else 0.5
                print(
                    "Early Warning: Police approaching corridor, "
                    f"bottom={police_bottom}, score={police_score:.0f}, pre-steering={steering_target}"
                )

            if (
                police_score >= POLICE_INSTANT_SCORE
                and police_bottom >= 74
                and pw >= POLICE_INSTANT_WIDTH
                and ph >= POLICE_INSTANT_HEIGHT
                and police_blocks_ego_lane
                and police_center_in_corridor
            ):
                police_detected_now = True

            if (
                police_bottom >= 100
                and pw >= 28
                and police_blocks_ego_lane
                and police_center_in_corridor
            ):
                police_obstacle_close = True
            elif (
                police_score >= POLICE_EMERGENCY_SCORE
                and police_bottom >= 90
                and pw >= 26
                and police_blocks_ego_lane
                and police_center_in_corridor
            ):
                police_emergency_close = True
            elif (
                police_score >= POLICE_SUDDEN_SCORE
                and police_bottom >= 82
                and pw >= 22
                and ph >= POLICE_SUDDEN_HEIGHT
                and police_blocks_ego_lane
                and police_center_in_corridor
            ):
                police_sudden_risk = True

        police_urgent_avoidance = False
        police_red_chase_active = False

        # Decision State Hierarchy [cite: 15]
        if (
            evade_back_car
            and not (
                police_hard_immediate_dodge
                or police_collision_imminent
                or police_detected_now
                or police_sudden_risk
                or police_emergency_close
                or (police_mode and police_obstacle_close)
            )
        ):
            steering_target = back_steering_escape
            print(f"Trailing Car Alert! Escaping to steering: {steering_target}")
        elif police_hard_immediate_dodge:
            # Absolute priority: police is already close/low in the front camera.
            # Bypass red-token chasing and normal cooldown immediately.
            steering_target = police_escape_steering
            police_urgent_avoidance = True
            with data_lock:
                shared_data['police_avoid_timer'] = POLICE_HARD_DODGE_HOLD_FRAMES
                shared_data['police_avoid_steering'] = steering_target
                shared_data['police_avoid_acceleration'] = POLICE_HARD_DODGE_ACCELERATION
            print(f"HARD Police Dodge NOW! Steering: {steering_target}")
        elif police_collision_imminent:
            # Priority 1: Do not hit the police car head-on
            steering_target = police_escape_steering
            police_urgent_avoidance = True
            with data_lock:
                shared_data['police_avoid_timer'] = POLICE_COLLISION_AVOID_HOLD_FRAMES
                shared_data['police_avoid_steering'] = steering_target
                shared_data['police_avoid_acceleration'] = POLICE_COLLISION_BRAKE_ACCELERATION
            print(f"Collision Imminent! Full police dodge: {steering_target}")
        elif police_detected_now:
            # Priority 2: The moment a police candidate is detected in the
            # driving corridor, evade immediately without extra delay.
            steering_target = police_escape_steering
            police_urgent_avoidance = True
            with data_lock:
                shared_data['police_avoid_acceleration'] = POLICE_AVOID_ACCELERATION
            print(f"Instant Police Avoidance! Steering: {steering_target}")
        elif police_sudden_risk:
            # Priority 3: Crest/visibility-change protection. If a strong
            # police-like obstacle suddenly appears ahead, evade immediately.
            steering_target = police_escape_steering
            police_urgent_avoidance = True
            with data_lock:
                shared_data['police_avoid_acceleration'] = POLICE_AVOID_ACCELERATION
            print(f"Sudden Police Risk! Steering: {steering_target}")
        elif police_emergency_close:
            # Priority 4: Even before the police event fully confirms, treat a
            # strong close police candidate as an obstacle to avoid game over.
            steering_target = police_escape_steering
            police_urgent_avoidance = True
            with data_lock:
                shared_data['police_avoid_acceleration'] = POLICE_AVOID_ACCELERATION
            print(f"Emergency Police Avoidance! Steering: {steering_target}")
            
        elif police_mode:
            # Priority 5: During a confirmed police event, avoid the police car
            # first. Only when it is safe, chase one real red token within the
            # 5s window. Do not use largest_red because police lights/curb can
            # be larger than the actual EV2 red token.
            if police_obstacle_close:
                steering_target = police_escape_steering
                police_urgent_avoidance = True
                with data_lock:
                    shared_data['police_avoid_acceleration'] = POLICE_AVOID_ACCELERATION
                print(f"Police Car Close! Evading to steering: {steering_target}")
            else:
                with data_lock:
                    red_already_attempted = shared_data.get('police_red_attempted', False)

                if not red_already_attempted:
                    red_target = choose_police_red_token(
                        filtered_red_contours,
                        police_raw_box,
                        police_candidate_box,
                        frame_center,
                    )

                    if red_target is not None:
                        error = red_target['cx'] - frame_center
                        steering_target = -1.0 if error < -12 else (1.0 if error > 12 else 0.0)
                        police_red_chase_active = True
                        print(
                            "Police Red Chase! "
                            f"cx={red_target['cx']}, err={error}, "
                            f"bottom={red_target['bottom']}, area={red_target['area']:.0f}, "
                            f"steer={steering_target}"
                        )

                        if (
                            abs(error) <= POLICE_CAPTURE_MAX_ERROR
                            and red_target['bottom'] >= POLICE_CAPTURE_MIN_BOTTOM
                        ):
                            police_mode = update_police_state(
                                False,
                                police_score=0.0,
                                red_token_captured=True,
                            )
                else:
                    steering_target = 0.0

        else:
            # Priority 3: Standard Navigation using the 5-Bucket Logic Engine
            ego_lane = 2  # The camera is fixed to the car, so the car is always in lane 2
            current_state = lane_states[ego_lane]
            best_lane = ego_lane
            
            if current_state == 'Danger':
                # Need to escape. Find nearest Target or Clear lane
                min_dist = float('inf')
                for priority in ['Target', 'Clear']:
                    for i, state in enumerate(lane_states):
                        if state == priority:
                            dist = abs(i - ego_lane)
                            if dist < min_dist:
                                min_dist = dist
                                best_lane = i
                    if best_lane != ego_lane:
                        break
            elif current_state == 'Clear':
                # Safe, but look for a Target (Green) lane
                min_dist = float('inf')
                for i, state in enumerate(lane_states):
                    if state == 'Target':
                        dist = abs(i - ego_lane)
                        if dist < min_dist:
                            min_dist = dist
                            best_lane = i
                            
            # Fire Steering Commands
            if best_lane < ego_lane:
                steering_target = -1.0
                print(f"Logic Engine: Steering LEFT to Lane {best_lane}")
            elif best_lane > ego_lane:
                steering_target = 1.0
                print(f"Logic Engine: Steering RIGHT to Lane {best_lane}")

        # Commit decision to shared resources safely using a Tap Sequence [cite: 152]
        with data_lock:
            if (
                police_alert_active
                and police_escape_steering != 0.0
                and shared_data['tap_timer'] > 0
                and np.sign(shared_data['tap_steering']) != np.sign(police_escape_steering)
            ):
                # Cancel an old token-avoidance tap if it would steer us into
                # the police car.
                shared_data['tap_timer'] = 0
                shared_data['cooldown_timer'] = 0
                shared_data['tap_steering'] = 0.0
                shared_data['steering_input'] = 0.0

            if evade_back_car and not police_urgent_avoidance:
                shared_data['tap_steering'] = back_steering_escape
                shared_data['tap_timer'] = TRAILING_TAP_FRAMES
                shared_data['cooldown_timer'] = 0
                shared_data['steering_input'] = back_steering_escape
                shared_data['trailing_detect_count'] = 0
                shared_data['trailing_escape_cooldown'] = TRAILING_ESCAPE_COOLDOWN_FRAMES
                shared_data['trailing_escape_send_timer'] = TRAILING_TAP_FRAMES
                shared_data['light_signal_timer'] = 0
                lane_delta = -1 if back_steering_escape < 0 else 1
                shared_data['target_lane'] = int(np.clip(shared_data.get('current_lane', 1) + lane_delta, 0, 2))
                shared_data['current_lane'] = shared_data['target_lane']
                print(
                    "Initiating Trailing Escape! "
                    f"Steering: {back_steering_escape}, lane={shared_data['current_lane']}"
                )
            elif police_urgent_avoidance and steering_target != 0.0:
                # Police avoidance must bypass the normal tap/cooldown rhythm.
                if shared_data['police_avoid_timer'] <= 0:
                    shared_data['police_avoid_timer'] = POLICE_AVOID_HOLD_FRAMES
                    shared_data['police_avoid_acceleration'] = POLICE_AVOID_ACCELERATION
                shared_data['police_avoid_steering'] = steering_target
                shared_data['tap_timer'] = 0
                shared_data['cooldown_timer'] = 0
                shared_data['tap_steering'] = steering_target
                shared_data['steering_input'] = steering_target
            elif shared_data['police_avoid_timer'] > 0:
                shared_data['police_avoid_timer'] -= 1
                shared_data['steering_input'] = shared_data['police_avoid_steering']
            elif police_red_chase_active and steering_target != 0.0:
                shared_data['tap_steering'] = steering_target
                shared_data['tap_timer'] = max(8, NORMAL_TAP_FRAMES // 2)
                shared_data['cooldown_timer'] = 0
                shared_data['steering_input'] = steering_target
                print(f"Initiating Police Red Chase Tap! Steering: {steering_target}")
            elif shared_data['tap_timer'] > 0:
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

        # Draw the 5-Bucket Proximity Scan Line
        # ---------------------------------------------------------
        # FIXED: Perspective-Aware 5-Bucket Mesh Overlay
        # ---------------------------------------------------------
        # Calculate boundaries at both the top and bottom of the scan zone
        top_left_edge, top_right_edge = road_edges_at_y(PROXIMITY_SCAN_TOP)
        bottom_left_edge, bottom_right_edge = road_edges_at_y(PROXIMITY_SCAN_BOTTOM)
        
        # Apply the exact same buffer margin used in your processing logic
        buffer_margin = 8
        top_left_edge += buffer_margin
        top_right_edge -= buffer_margin
        bottom_left_edge += buffer_margin
        bottom_right_edge -= buffer_margin
        
        # Determine the sliding width per lane step for top and bottom lines
        top_lane_width = (top_right_edge - top_left_edge) / float(NUM_LANES)
        bottom_lane_width = (bottom_right_edge - bottom_left_edge) / float(NUM_LANES)
        
        for i in range(NUM_LANES):
            state = lane_states[i]
            if state == 'Danger':
                color = (0, 0, 255)      # Red
            elif state == 'Target':
                color = (0, 255, 0)      # Green
            else:
                color = (255, 255, 255)  # White
                
            # Calculate the 4 corner points of the perspective polygon for lane i
            tx1 = int(top_left_edge + i * top_lane_width)
            tx2 = int(top_left_edge + (i + 1) * top_lane_width)
            bx1 = int(bottom_left_edge + i * bottom_lane_width)
            bx2 = int(bottom_left_edge + (i + 1) * bottom_lane_width)
            
            # Construct the polygon vertex array
            pts = np.array([
                [tx1, PROXIMITY_SCAN_TOP],      # Top-Left
                [tx2, PROXIMITY_SCAN_TOP],      # Top-Right
                [bx2, PROXIMITY_SCAN_BOTTOM],   # Bottom-Right
                [bx1, PROXIMITY_SCAN_BOTTOM]    # Bottom-Left
            ], dtype=np.int32)
            
            # Draw the perspective quad frame
            cv2.polylines(debug_frame, [pts], True, color, 1, lineType=cv2.LINE_AA)
            
            # Position the text dynamically near the lower center of each custom quad
            text_x = int((bx1 + bx2) / 2) - 12
            text_y = PROXIMITY_SCAN_BOTTOM - 4
            cv2.putText(
                debug_frame,
                f"L{i}:{state[0]}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )

        with data_lock:
            overlay_game_state = shared_data['race_game_state']
            overlay_race_elapsed = shared_data['race_elapsed']
            overlay_phase = shared_data['race_phase']
            overlay_police_time = (
                POLICE_PHASE_START_SECONDS <= overlay_phase <= POLICE_PHASE_END_SECONDS
            )
            overlay_police_active = shared_data['police_active']
            overlay_police_state = 'ON' if overlay_police_active else 'OFF'
            overlay_police_timer = shared_data['police_timer'] * TASK_PERIOD_SECONDS
            overlay_light_state = (
                'SIGNAL'
                if shared_data['light_signal_timer'] > 0
                else ('ON' if shared_data['lights_on'] else 'OFF')
            )

        overlay_font_scale = 0.30
        overlay_color = (255, 255, 255)

        cv2.putText(
            debug_frame,
            (
                f"game={overlay_game_state} {overlay_race_elapsed:.1f}s | "
                f"phase={overlay_phase:.1f}s"
            ),
            (5, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            overlay_font_scale,
            overlay_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            debug_frame,
            (
                f"mode={'DARKNESS' if low_brightness else 'NORMAL'} | "
                f"light={overlay_light_state} | "
                f"brightness={brightness:.1f}"
            ),
            (5, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            overlay_font_scale,
            overlay_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            debug_frame,
            f"lanes={NUM_LANES} | scan line=ACTIVE",
            (5, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            overlay_font_scale,
            overlay_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            debug_frame,
            (
                f"policeTime={'ON' if overlay_police_time else 'OFF'} | "
                f"police={overlay_police_state} | "
                f"timer={overlay_police_timer:.1f}s"
            ),
            (5, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            overlay_font_scale,
            overlay_color,
            1,
            cv2.LINE_AA,
        )

        for x, y, w, h in police_debug.get('red_boxes', []):
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        for x, y, w, h in police_debug.get('blue_boxes', []):
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        candidate_box = police_debug.get('candidate_box')
        raw_candidate_box = police_debug.get('raw_candidate_box')
        if raw_candidate_box is not None:
            x, y, w, h = raw_candidate_box
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 165, 255), 1)

        if candidate_box is not None:
            x, y, w, h = candidate_box
            candidate_color = (255, 255, 255) if police_debug.get('seen') else (0, 165, 255)
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), candidate_color, 2)
            cv2.putText(
                debug_frame,
                (
                    f"police area={police_debug['combined_area']} "
                    f"fill={police_debug['fill_ratio']:.2f} "
                    f"lane={police_debug['lane_coverage']:.2f} "
                    f"score={police_debug['score']:.0f}"
                ),
                (x, max(60, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                candidate_color,
                1,
                cv2.LINE_AA,
            )

        capture_reason = None
        candidate_score = float(police_debug.get('score', 0.0))
        if (
            police_hard_immediate_dodge
            or police_collision_imminent
            or police_detected_now
            or police_emergency_close
            or police_obstacle_close
        ):
            capture_reason = 'urgent'
        elif police_debug.get('seen'):
            capture_reason = 'confirmed'
        elif raw_candidate_box is not None and candidate_score >= POLICE_DEBUG_CAPTURE_SCORE_MIN:
            capture_reason = 'candidate'
        elif police_mode and raw_candidate_box is not None:
            capture_reason = 'active'

        should_capture = False
        with data_lock:
            if shared_data['police_debug_capture_cooldown'] > 0:
                shared_data['police_debug_capture_cooldown'] -= 1
            if (
                POLICE_DEBUG_CAPTURE_ENABLED
                and capture_reason is not None
                and shared_data['police_debug_capture_cooldown'] <= 0
            ):
                shared_data['police_debug_capture_cooldown'] = POLICE_DEBUG_CAPTURE_COOLDOWN_FRAMES
                should_capture = True
            shared_data['debug_front_frame'] = debug_frame
        if should_capture:
            save_police_debug_capture(debug_frame, police_debug, capture_reason)
    else:
        update_police_state(False, police_score=0.0, allow_activation=False)


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
        police_avoid_active = shared_data['police_avoid_timer'] > 0
        police_avoid_acceleration = shared_data['police_avoid_acceleration']
        trailing_escape_active = shared_data['trailing_escape_send_timer'] > 0
        if trailing_escape_active:
            shared_data['trailing_escape_send_timer'] -= 1
            shared_data['light_signal_timer'] = 0
        # NEW: Darkness Improvement
        # Unity treats acceleration -1.0 during darkness as the light recovery command.
        if trailing_escape_active:
            shared_data['trailing_escape_send_timer'] -= 1

        # Highest priority: EV1 requires acceleration_input = -1.0 to brake/open light.
        # Do not let trailing/police logic cancel the light command.
        if shared_data['light_signal_timer'] > 0:
            shared_data['light_signal_timer'] -= 1
            steering_to_send = 0.0
            acceleration_to_send = LIGHT_TOGGLE_ACCELERATION
        elif shared_data['low_brightness']:
            if shared_data['front_brightness'] < CRITICAL_DARKNESS_THRESHOLD:
                acceleration_to_send = CRITICAL_DARKNESS_ACCELERATION
            else:
                acceleration_to_send = DARKNESS_ACCELERATION
        elif trailing_escape_active:
            steering_to_send = shared_data['tap_steering']
            acceleration_to_send = 1.0
        elif police_avoid_active:
            acceleration_to_send = police_avoid_acceleration
        else:
            acceleration_to_send = 1.0

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
    print("Debug keys: T = reset Python timers/state, Q = quit")
    print("Note: R is reserved for the simulator only. Python will not reset its clock on R.")
    print("Auto clock reset: if simulator image stops for ~3s, Python returns to WAIT 0.0s.")
    print(f"Automatic police screenshots from code: {'ON' if POLICE_DEBUG_CAPTURE_ENABLED else 'OFF'}")

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
    
    last_runtime_reset_key_time = 0.0
    
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

            key = cv2.waitKey(33) & 0xFF

            # OpenCV catches keys only when an OpenCV window has focus.
            # The keyboard module gives a best-effort global R/T reset when the simulator window has focus.
            try:
                # Do NOT listen to global R here. R only resets the simulator when
                # the simulator window has focus. Listening to global R caused
                # Python timers to reset even when Unity did not restart.
                global_reset_pressed = keyboard.is_pressed('t')
            except Exception:
                global_reset_pressed = False

            if key == ord('q'):
                is_running = False
                break
            elif key == ord('t') or global_reset_pressed:
                now_key_time = time.monotonic()
                if now_key_time - last_runtime_reset_key_time > 0.80:
                    reset_runtime_state('manual hotkey')
                    last_runtime_reset_key_time = now_key_time
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


