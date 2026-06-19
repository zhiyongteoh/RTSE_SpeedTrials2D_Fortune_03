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
    'debug_back_frame': None,
    'chasing_car_seen': False,
    'chasing_car_score': 0.0,
    'chasing_car_box': None,
    'chasing_car_reason': 'NONE',
    'chasing_prev_area': 0.0,
    'chasing_prev_bottom_y': 0,
    'chasing_prev_center_x': None,
    'chasing_lost_frames': 0,
    'debug_front_frame': None,  # Front frame with five-lane token labels
    'steering_input' : 0.0,
    'acceleration_input' : 1.0,
    'target_lane': 2,        # Five-lane index: center is L2.
    'current_lane': 2,
    'settled_lane': 2,
    'lane_arrival_confirmed': True,
    'lane_settle_timer': 0,
    'pending_lane': None,
    'golden_lane_active': False,
    'golden_lane_target': None,
    'golden_lane_timer': 0,
    'golden_rearm_timer': 0,
    'golden_detect_score': 0.0,
    'golden_scan_counter': 0,
    'low_brightness': False,  # Event flag
    # NEW: Darkness Improvement
    'front_brightness': 255.0,
    'last_front_brightness': 255.0,
    'front_road_brightness': 255.0,
    'front_center_brightness': 255.0,
    'front_darkness_metric': 255.0,
    'light_retry_cooldown': 0,
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
TOKEN_ROI_EDGE_MARGIN = 8

PROXIMITY_SCAN_TOP = 165
PROXIMITY_SCAN_BOTTOM = 185
PROXIMITY_PIXEL_THRESHOLD = 15

# Extra red/yellow look-ahead detection.
# The original scan line only checks a thin band at y=165..185, so fast red
# tokens can be detected too late. This wider zone marks dangerous lanes earlier.
DANGER_LOOKAHEAD_TOP = 105
DANGER_LOOKAHEAD_BOTTOM = 212
DANGER_LOOKAHEAD_MIN_AREA = 7
DANGER_LOOKAHEAD_MAX_AREA = 3600
DANGER_CLOSE_BOTTOM_Y = 150
DANGER_CLOSE_ADJACENT_AREA = 120
RED_EMERGENCY_TAP_FRAMES = 18

# Token-shape filters. Tokens are small round blobs in the 320 x 240
# processing frame; circularity rejects rectangular road art and lane fragments.
TOKEN_MIN_AREA = 8
TOKEN_MAX_AREA = 1800
TOKEN_MAX_DIMENSION = 64
TOKEN_MIN_CIRCULARITY = 0.45
TOKEN_MIN_FILL_RATIO = 0.34
RIGHT_SHOULDER_REJECT_MARGIN = 18

# ---------------------------------------------------------
# Police Event Configuration
# ---------------------------------------------------------
TASK_PERIOD_SECONDS = 0.033

# Golden Lane EV5: top-bar text says "LANE X - ALL GREEN!" using 1-5.
# Internal lane labels are L0-L4, so the detector converts X -> X - 1.
GOLDEN_LANE_EVENT_SECONDS = 5.0
GOLDEN_LANE_EVENT_FRAMES = max(1, int(GOLDEN_LANE_EVENT_SECONDS / TASK_PERIOD_SECONDS))
GOLDEN_LANE_REARM_FRAMES = max(1, int(1.2 / TASK_PERIOD_SECONDS))
GOLDEN_LANE_DETECT_INTERVAL_FRAMES = 15
GOLDEN_LANE_TOP = 0
GOLDEN_LANE_BOTTOM = 90
# Matching occurs only after yellow-banner and black-text validation, allowing
# a slightly lower threshold than whole-HUD matching without false triggers.
GOLDEN_LANE_MATCH_THRESHOLD = 0.27
GOLDEN_BANNER_MIN_AREA = 450
GOLDEN_BANNER_MIN_WIDTH = 90
GOLDEN_YELLOW_LOWER = np.array([16, 70, 90])
GOLDEN_YELLOW_UPPER = np.array([40, 255, 255])
GOLDEN_BLACK_UPPER_VALUE = 160
GOLDEN_LANE_TEMPLATES = None
LANE_SETTLE_FRAMES = 10

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
# Use an effective road brightness metric, not only whole-screen mean.
# This catches very fast EV1 darkness drops even when UI/text/tokens keep
# the whole-frame average artificially high.
DARKNESS_ENTER_THRESHOLD = 48.0
DARKNESS_ENTER_FRAMES = 1
DARKNESS_EXIT_THRESHOLD = 62.0
DARKNESS_EXIT_FRAMES = 12
DARKNESS_EMERGENCY_THRESHOLD = 34.0
DARKNESS_DROP_THRESHOLD = 12.0
DARKNESS_DROP_TARGET_THRESHOLD = 50.0

CRITICAL_DARKNESS_THRESHOLD = 38.0

DARKNESS_ACCELERATION = 0.55
CRITICAL_DARKNESS_ACCELERATION = 0.35
LIGHT_TOGGLE_ACCELERATION = -1.0
LIGHT_TOGGLE_FRAMES = 2
DARKNESS_RETRY_SIGNAL_FRAMES = 1
DARKNESS_RETRY_COOLDOWN_FRAMES = 90
NORMAL_TAP_FRAMES = 16          # Stronger steering duration for token avoidance
DARKNESS_TAP_FRAMES = 14        # Longer dodge in darkness, but not too long
TRAILING_TAP_FRAMES = 34        # Chasing car needs a longer lane-change hold
TRAILING_CONFIRM_FRAMES = 2
TRAILING_ESCAPE_COOLDOWN_FRAMES = 38
TRAILING_ESCAPE_ACCELERATION = 0.95  # Do not use darkness slow speed during chasing escape
NORMAL_COOLDOWN_FRAMES = 5      # Faster recovery for repeated red-token dodges
DARKNESS_COOLDOWN_FRAMES = 8
TRAILING_DEBUG_VERSION = "trailing-v5-state"

# ---------------------------------------------------------
# Back Camera Chasing Car Detection
# Detects the cyan/turquoise chasing car from rear camera.
# ---------------------------------------------------------
TRAILING_DEBUG_VERSION = "chasing-cyan-v2-approach"

# Use a tighter central road ROI for the back camera.
# This avoids side signs, lamp posts, and off-road objects.
BACK_ROAD_TOP_Y = 76
BACK_ROAD_BOTTOM_Y = 239
BACK_ROAD_LEFT_TOP = 126
BACK_ROAD_RIGHT_TOP = 194
BACK_ROAD_LEFT_BOTTOM = 34
BACK_ROAD_RIGHT_BOTTOM = 286

# Car-shape and color thresholds.
CHASE_MIN_CYAN_PIXELS = 120
CHASE_MIN_BLUE_PIXELS = 6
CHASE_MIN_DARK_PIXELS = 22
CHASE_MIN_SCORE = 850.0
CHASE_CLOSE_SCORE = 980.0
CHASE_CLOSE_BOTTOM_Y = 138
CHASE_CLOSE_AREA = 950
CHASE_MIN_WIDTH = 36
CHASE_MIN_HEIGHT = 14
CHASE_MIN_ASPECT = 1.35
CHASE_MAX_ASPECT = 5.8
CHASE_MIN_FILL_RATIO = 0.16
CHASE_CENTER_TOLERANCE = 82

# Temporal validation: the chasing car should become larger or move lower.
CHASE_APPROACH_AREA_GAIN = 1.08
CHASE_APPROACH_BOTTOM_GAIN = 3
CHASE_MAX_CENTER_JUMP = 80
CHASE_MEMORY_LOST_FRAMES = 6
CHASE_KERNEL = np.ones((5, 5), np.uint8)

# Acceleration floors chosen by the frame arbiter. Emergency modes need enough
# forward motion for lateral steering authority, even if Darkness is active.
NORMAL_ACCELERATION = 1.0
RED_YELLOW_DODGE_ACCELERATION = 0.72
POLICE_DODGE_ACCELERATION = 0.62
POLICE_COLLISION_ACCELERATION = 0.30
POLICE_RED_COLLECTION_ACCELERATION = 0.68

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


def back_road_edges_at_y(pixel_y):
    """Return back-camera road edges for shared object gating."""
    pixel_y = float(np.clip(pixel_y, BACK_ROAD_TOP_Y, BACK_ROAD_BOTTOM_Y))
    scale = (pixel_y - BACK_ROAD_TOP_Y) / float(BACK_ROAD_BOTTOM_Y - BACK_ROAD_TOP_Y)
    left = BACK_ROAD_LEFT_TOP + (BACK_ROAD_LEFT_BOTTOM - BACK_ROAD_LEFT_TOP) * scale
    right = BACK_ROAD_RIGHT_TOP + (BACK_ROAD_RIGHT_BOTTOM - BACK_ROAD_RIGHT_TOP) * scale
    return left, right


def token_contour_metrics(contour):
    """Return contour metrics used for circular token filtering and overlay."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    (_, _), radius = cv2.minEnclosingCircle(contour)
    circularity = 0.0
    if perimeter > 0:
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
    fill_ratio = area / float(max(w * h, 1))

    M = cv2.moments(contour)
    if M['m00'] > 0:
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
    else:
        center_x = x + (w // 2)
        center_y = y + (h // 2)

    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'fill_ratio': fill_ratio,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': center_x,
        'cy': center_y,
        'bottom': y + h,
        'radius': radius,
    }


def object_gate_reason(
    metrics,
    road_mask=None,
    edge_func=detection_edges_at_y,
    edge_margin=TOKEN_ROI_EDGE_MARGIN,
    shape_kind='round',
    min_area=TOKEN_MIN_AREA,
    max_area=TOKEN_MAX_AREA,
):
    """
    Shared road/shape gate for real objects.

    Keeping this centralized prevents kerb, edge, and off-road fragments from
    being accepted differently by tokens vs. chasing-car detection.
    """
    center_x = metrics['cx']
    center_y = metrics['cy']

    if road_mask is not None:
        if center_y < 0 or center_y >= road_mask.shape[0] or center_x < 0 or center_x >= road_mask.shape[1]:
            return 'outside-road'
        if road_mask[int(center_y), int(center_x)] == 0:
            return 'outside-road'

    left_edge, right_edge = edge_func(center_y)
    if center_x < left_edge + edge_margin or center_x > right_edge - edge_margin:
        return 'edge'

    if metrics['area'] < min_area or metrics['area'] > max_area:
        return 'size'

    if max(metrics['w'], metrics['h']) > TOKEN_MAX_DIMENSION and shape_kind == 'round':
        return 'size'

    thin_ratio = min(metrics['w'], metrics['h']) / float(max(metrics['w'], metrics['h'], 1))
    if thin_ratio < 0.18:
        return 'too-thin'

    if shape_kind == 'round':
        if metrics['circularity'] < TOKEN_MIN_CIRCULARITY or metrics['fill_ratio'] < TOKEN_MIN_FILL_RATIO:
            return 'not-round'
    elif shape_kind == 'wide-car':
        aspect_ratio = metrics['w'] / float(max(metrics['h'], 1))
        if aspect_ratio < CHASE_MIN_ASPECT or aspect_ratio > CHASE_MAX_ASPECT:
            return 'bad-shape'
        if metrics['fill_ratio'] < CHASE_MIN_FILL_RATIO:
            return 'too-thin'

    return None


def is_circular_token_shape(contour, min_area=TOKEN_MIN_AREA, max_area=TOKEN_MAX_AREA):
    """Accept compact round token blobs; reject elongated or rectangular shapes."""
    metrics = token_contour_metrics(contour)
    return object_gate_reason(
        metrics,
        road_mask=None,
        edge_func=lambda _y: (-9999, 9999),
        edge_margin=0,
        shape_kind='round',
        min_area=min_area,
        max_area=max_area,
    ) is None


def is_centroid_inside_token_road_roi(center_x, center_y, road_mask=None):
    """Keep token centroids on asphalt, away from shoulder/grass ROI edges."""
    if center_y < DETECT_ROAD_TOP_Y or center_y > DETECT_ROAD_BOTTOM_Y:
        return False

    left_edge, right_edge = detection_edges_at_y(center_y)
    if (
        center_x < left_edge + TOKEN_ROI_EDGE_MARGIN
        or center_x > right_edge - TOKEN_ROI_EDGE_MARGIN
    ):
        return False

    if road_mask is not None:
        y = int(np.clip(center_y, 0, road_mask.shape[0] - 1))
        x = int(np.clip(center_x, 0, road_mask.shape[1] - 1))
        if road_mask[y, x] == 0:
            return False

    return True


def token_rejection_reason(contour, color_code=None, road_mask=None):
    metrics = token_contour_metrics(contour)
    reason = object_gate_reason(
        metrics,
        road_mask=road_mask,
        edge_func=detection_edges_at_y,
        edge_margin=TOKEN_ROI_EDGE_MARGIN,
        shape_kind='round',
    )
    if reason is not None:
        return reason

    if metrics['bottom'] >= EGO_IGNORE_TOP and EGO_IGNORE_LEFT <= metrics['cx'] <= EGO_IGNORE_RIGHT:
        return 'ego-car'

    if color_code == 'R':
        _, right_edge = detection_edges_at_y(metrics['bottom'])
        if metrics['bottom'] > 120 and metrics['cx'] > right_edge - RIGHT_SHOULDER_REJECT_MARGIN:
            return 'kerb'

    return None


def is_token_like_contour(contour, color_code=None, road_mask=None):
    """Reject road art, lane lines, grass bands, and the player's own car."""
    return token_rejection_reason(contour, color_code, road_mask) is None


def collect_token_observations(contours, color_code, road_mask=None, max_area=None, max_dimension=None):
    """Convert contours into compact token records used only by the overlay."""
    observations = []

    for contour in contours:
        if not is_token_like_contour(contour, color_code, road_mask):
            continue

        metrics = token_contour_metrics(contour)
        area = metrics['area']
        x = metrics['x']
        y = metrics['y']
        w = metrics['w']
        h = metrics['h']
        if max_area is not None and area > max_area:
            continue
        if max_dimension is not None and max(w, h) > max_dimension:
            continue

        center_x = metrics['cx']
        center_y = metrics['cy']
        bottom_y = metrics['bottom']
        observations.append({
            'color': color_code,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': center_x,
            'cy': center_y,
            'radius': max(3, int(round(metrics['radius']))),
            'confidence': min(1.0, max(0.0, metrics['circularity'])),
            'lane': pixel_x_to_debug_lane(center_x, bottom_y),
        })

    return observations


def classify_token_contours(contours, color_code, road_mask=None, max_rejected=8):
    """Return accepted contours plus low-cost rejected samples for debug labels."""
    accepted = []
    rejected = []

    for contour in contours:
        metrics = token_contour_metrics(contour)
        reason = token_rejection_reason(contour, color_code, road_mask)
        if reason is None:
            accepted.append(contour)
        elif len(rejected) < max_rejected and metrics['area'] >= TOKEN_MIN_AREA:
            rejected.append({
                'color': color_code,
                'reason': reason,
                'cx': metrics['cx'],
                'cy': metrics['cy'],
                'radius': max(3, int(round(metrics['radius']))),
            })

    return accepted, rejected


def mark_danger_lanes_from_token_contours(lane_states, contours, color_code, road_mask=None):
    """
    Mark lanes as dangerous from a wider red/yellow look-ahead zone.

    This fixes the case where the car only notices a red token at the thin
    proximity scan line and reaches it before the lane change finishes.
    """
    danger_lanes = set()

    for contour in contours:
        if not is_token_like_contour(contour, color_code, road_mask):
            continue

        area = cv2.contourArea(contour)
        if area < DANGER_LOOKAHEAD_MIN_AREA or area > DANGER_LOOKAHEAD_MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + (w // 2)
        bottom_y = y + h

        if bottom_y < DANGER_LOOKAHEAD_TOP or bottom_y > DANGER_LOOKAHEAD_BOTTOM:
            continue

        lane = pixel_x_to_debug_lane(center_x, bottom_y)
        lane_states[lane] = 'Danger'
        danger_lanes.add(lane)

        # If the red/yellow token is already very close and reasonably large,
        # also mark the neighbouring lane as not preferred. This prevents the
        # controller from changing only half a lane and still clipping the token.
        if bottom_y >= DANGER_CLOSE_BOTTOM_Y and area >= DANGER_CLOSE_ADJACENT_AREA:
            if lane > 0:
                lane_states[lane - 1] = 'Danger'
                danger_lanes.add(lane - 1)
            if lane < NUM_LANES - 1:
                lane_states[lane + 1] = 'Danger'
                danger_lanes.add(lane + 1)

    return danger_lanes


def build_golden_lane_templates():
    """Build normalized digit glyphs for lanes 1..5 without OCR deps."""
    templates = []
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
    ]
    for shown_lane in range(1, NUM_LANES + 1):
        variants = (str(shown_lane),)
        for text in variants:
            for font in fonts:
                for scale in (0.42, 0.50, 0.58):
                    thickness = 1
                    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
                    template = np.zeros((h + baseline + 10, w + 10), dtype=np.uint8)
                    cv2.putText(
                        template,
                        text,
                        (5, h + 3),
                        font,
                        scale,
                        255,
                        thickness,
                        cv2.LINE_AA,
                    )
                    points = cv2.findNonZero(template)
                    if points is not None:
                        gx, gy, gw, gh = cv2.boundingRect(points)
                        glyph = template[gy:gy + gh, gx:gx + gw]
                        glyph = cv2.resize(glyph, (18, 26), interpolation=cv2.INTER_AREA)
                        _, glyph = cv2.threshold(glyph, 80, 255, cv2.THRESH_BINARY)
                        templates.append({
                            'shown_lane': shown_lane,
                            'image': glyph,
                        })
    return templates


def detect_golden_lane_event(frame):
    """
    OpenCV-only Golden Lane reader.

    1. Find the yellow HUD banner in the top of the image.
    2. Extract only dark/black lettering inside that yellow rectangle.
    3. Template-match LANE 1 through LANE 5 against the extracted text.

    Timer anchor: detection time, because this script has no simulator event
    timestamp channel for EV5.
    """
    global GOLDEN_LANE_TEMPLATES
    if GOLDEN_LANE_TEMPLATES is None:
        GOLDEN_LANE_TEMPLATES = build_golden_lane_templates()

    top_bar = frame[GOLDEN_LANE_TOP:GOLDEN_LANE_BOTTOM, :]
    if top_bar.size == 0:
        return None, 0.0

    top_hsv = cv2.cvtColor(top_bar, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(top_hsv, GOLDEN_YELLOW_LOWER, GOLDEN_YELLOW_UPPER)
    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 9), np.uint8),
        iterations=2,
    )
    contours, _ = cv2.findContours(
        yellow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    banner_box = None
    banner_score = -1.0
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area < GOLDEN_BANNER_MIN_AREA or w < GOLDEN_BANNER_MIN_WIDTH or h < 10:
            continue
        fill = area / float(max(w * h, 1))
        score = area * fill
        if score > banner_score:
            banner_score = score
            banner_box = (x, y, w, h)

    # No yellow region means this cannot be the Golden Lane message. This is
    # the important false-positive guard missing from whole-HUD edge matching.
    if banner_box is None:
        return None, 0.0

    x, y, w, h = banner_box
    pad_x = max(2, int(w * 0.02))
    pad_y = max(1, int(h * 0.08))
    x1, x2 = max(0, x + pad_x), min(top_bar.shape[1], x + w - pad_x)
    y1, y2 = max(0, y + pad_y), min(top_bar.shape[0], y + h - pad_y)
    banner_hsv = top_hsv[y1:y2, x1:x2]
    if banner_hsv.size == 0:
        return None, 0.0

    # Black text may have antialiased grey edge pixels. Hue/saturation are not
    # useful for black, so extract it by value while remaining inside the
    # already-confirmed yellow banner rectangle.
    black_text = cv2.inRange(
        banner_hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, GOLDEN_BLACK_UPPER_VALUE]),
    )
    if cv2.countNonZero(black_text) < 18:
        return None, 0.0
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(black_text, 8)
    characters = []
    for component in range(1, component_count):
        cx, cy, cw, ch, area = stats[component]
        if area >= 3 and ch >= 6 and cw >= 1:
            characters.append((cx, cy, cw, ch, area))
    characters.sort(key=lambda item: item[0])

    # The validated message starts with "LANE X". Font antialiasing can join
    # neighbouring letters, so find the first word-space rather than assuming
    # a fixed component number; the component after that gap is X.
    if len(characters) < 3:
        return None, 0.0
    widths = [item[2] for item in characters[:6]]
    gap_threshold = max(4.0, float(np.median(widths)) * 0.45)
    digit_component = None
    for index in range(1, min(len(characters), 7)):
        previous_right = characters[index - 1][0] + characters[index - 1][2]
        gap = characters[index][0] - previous_right
        if gap >= gap_threshold:
            digit_component = characters[index]
            break
    if digit_component is None:
        return None, 0.0
    gx, gy, gw, gh, _ = digit_component
    digit = black_text[gy:gy + gh, gx:gx + gw]
    digit = cv2.resize(digit, (18, 26), interpolation=cv2.INTER_AREA)
    _, digit = cv2.threshold(digit, 80, 255, cv2.THRESH_BINARY)

    best_lane = None
    best_score = 0.0
    for template in GOLDEN_LANE_TEMPLATES:
        tmpl = template['image']
        match = cv2.matchTemplate(digit, tmpl, cv2.TM_CCOEFF_NORMED)
        score = float(match[0, 0])
        if score > best_score:
            best_score = score
            best_lane = template['shown_lane']

    if best_lane is None or best_score < GOLDEN_LANE_MATCH_THRESHOLD:
        return None, best_score

    # In-game text is 1-based ("LANE 4"); internal/HUD lanes are L0-L4.
    internal_lane_index = int(best_lane) - 1
    return int(np.clip(internal_lane_index, 0, NUM_LANES - 1)), best_score


def update_golden_lane_state(small_frame):
    """Latch EV5 for 5s without extending the timer while the banner persists."""
    with data_lock:
        if shared_data.get('golden_rearm_timer', 0) > 0:
            shared_data['golden_rearm_timer'] -= 1

        if shared_data.get('golden_lane_active') and shared_data.get('golden_lane_timer', 0) > 0:
            shared_data['golden_lane_timer'] -= 1
            if shared_data['golden_lane_timer'] <= 0:
                shared_data['golden_lane_active'] = False
                shared_data['golden_lane_target'] = None
                shared_data['golden_rearm_timer'] = GOLDEN_LANE_REARM_FRAMES
            return

        shared_data['golden_scan_counter'] = (
            shared_data.get('golden_scan_counter', 0) + 1
        ) % GOLDEN_LANE_DETECT_INTERVAL_FRAMES
        should_scan = shared_data['golden_scan_counter'] == 0

    if not should_scan:
        return

    detected_lane, detect_score = detect_golden_lane_event(small_frame)
    with data_lock:
        shared_data['golden_detect_score'] = detect_score
        if detected_lane is None or shared_data.get('golden_rearm_timer', 0) > 0:
            return

        shared_data['golden_lane_active'] = True
        shared_data['golden_lane_target'] = detected_lane
        shared_data['golden_lane_timer'] = GOLDEN_LANE_EVENT_FRAMES
        shared_data['target_lane'] = detected_lane
        shared_data['pending_lane'] = detected_lane
        shared_data['lane_arrival_confirmed'] = False
        shared_data['lane_settle_timer'] = LANE_SETTLE_FRAMES

    print(
        "Golden Lane detected: "
        f"target=L{detected_lane}, score={detect_score:.2f}, "
        f"timer={GOLDEN_LANE_EVENT_SECONDS:.1f}s"
    )


# ---------------------------------------------------------
# NEW: Darkness Improvement Helper Functions
# ---------------------------------------------------------
def set_vehicle_light(is_on, brightness):
    """
    Track the vehicle light state when Darkness Mode changes and schedule the
    Unity-recognized light command. The command is a short -1.0 pulse only,
    not continuous reverse acceleration.
    """
    if shared_data['lights_on'] == is_on:
        return

    shared_data['lights_on'] = is_on
    if is_on:
        shared_data['light_signal_timer'] = max(
            shared_data['light_signal_timer'],
            LIGHT_TOGGLE_FRAMES,
        )
        shared_data['light_retry_cooldown'] = DARKNESS_RETRY_COOLDOWN_FRAMES
    else:
        shared_data['light_retry_cooldown'] = 0

    light_state = "ON" if is_on else "OFF"
    print(f"Vehicle Light {light_state} | brightness={brightness:.1f}")


def estimate_front_darkness_metric(small_frame):
    """
    Return robust brightness values for EV1 Darkness.

    Whole-frame mean is unreliable because HUD text, sky, tokens, and lane
    lines can stay bright while the road view suddenly becomes dark. For
    Darkness, the most useful signal is the road/center ROI brightness.
    """
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    scene_mean = float(np.mean(gray))

    roi_mask = np.zeros((240, 320), dtype=np.uint8)
    brightness_top_y = DETECT_ROAD_TOP_Y
    brightness_bottom_y = min(178, DETECT_ROAD_BOTTOM_Y)
    left_top, right_top = detection_edges_at_y(brightness_top_y)
    left_bottom, right_bottom = detection_edges_at_y(brightness_bottom_y)
    brightness_polygon = np.array([
        [int(left_bottom), brightness_bottom_y],
        [int(right_bottom), brightness_bottom_y],
        [int(right_top), brightness_top_y],
        [int(left_top), brightness_top_y],
    ], dtype=np.int32)
    cv2.fillPoly(roi_mask, [brightness_polygon], 255)

    road_pixels = gray[roi_mask > 0]
    if road_pixels.size == 0:
        road_mean = scene_mean
        road_p25 = scene_mean
    else:
        road_mean = float(np.mean(road_pixels))
        road_p25 = float(np.percentile(road_pixels, 25))

    center_roi = gray[96:178, 118:202]
    center_mean = float(np.mean(center_roi)) if center_roi.size else scene_mean

    # Use the darkest stable region as the state signal, but avoid using only
    # percentile because lane markings can create very low isolated pixels.
    darkness_metric = min(scene_mean, road_mean, center_mean)

    return {
        'scene_mean': scene_mean,
        'road_mean': road_mean,
        'road_p25': road_p25,
        'center_mean': center_mean,
        'metric': darkness_metric,
    }


def update_darkness_state(brightness, brightness_info=None):
    """
    Detect EV1 Darkness reliably.

    Uses an effective road brightness metric so sudden fast darkness drops do
    not get hidden by bright HUD/text/tokens. Sends a short light/brake pulse
    on entry, then optional short retry pulses if the scene is still very dark.
    """
    with data_lock:
        previous_brightness = float(shared_data.get('front_brightness', brightness))
        brightness = float(brightness)
        brightness_drop = previous_brightness - brightness

        shared_data['last_front_brightness'] = previous_brightness
        shared_data['front_brightness'] = brightness
        shared_data['front_darkness_metric'] = brightness

        if brightness_info is not None:
            shared_data['front_road_brightness'] = float(
                brightness_info.get('road_mean', brightness)
            )
            shared_data['front_center_brightness'] = float(
                brightness_info.get('center_mean', brightness)
            )

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
                print(
                    f"Darkness Mode ON | metric={brightness:.1f} | "
                    f"prev={previous_brightness:.1f} | drop={brightness_drop:.1f} | "
                    f"road={shared_data['front_road_brightness']:.1f} | "
                    f"center={shared_data['front_center_brightness']:.1f}"
                )
        else:
            shared_data['darkness_enter_count'] = 0

            # Do not reset light_signal_timer every frame. That caused
            # acceleration_input=-1.0 forever. Instead, use one short entry
            # pulse, then occasional short retry pulses only if still critical.
            if brightness <= CRITICAL_DARKNESS_THRESHOLD:
                if shared_data['light_signal_timer'] <= 0:
                    if shared_data['light_retry_cooldown'] > 0:
                        shared_data['light_retry_cooldown'] -= 1
                    else:
                        shared_data['light_signal_timer'] = max(
                            shared_data['light_signal_timer'],
                            DARKNESS_RETRY_SIGNAL_FRAMES,
                        )
                        shared_data['light_retry_cooldown'] = DARKNESS_RETRY_COOLDOWN_FRAMES
                        print(
                            f"Darkness light retry pulse | metric={brightness:.1f}"
                        )
            else:
                shared_data['light_retry_cooldown'] = DARKNESS_RETRY_COOLDOWN_FRAMES

            if brightness > DARKNESS_EXIT_THRESHOLD:
                shared_data['darkness_exit_count'] += 1
            else:
                shared_data['darkness_exit_count'] = 0

            if shared_data['darkness_exit_count'] >= DARKNESS_EXIT_FRAMES:
                shared_data['low_brightness'] = False
                shared_data['darkness_exit_count'] = 0
                shared_data['light_signal_timer'] = 0
                set_vehicle_light(False, brightness)
                print(f"Darkness Mode OFF | metric={brightness:.1f}")

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
                'target_lane': NUM_LANES // 2,
                'current_lane': NUM_LANES // 2,
                'settled_lane': NUM_LANES // 2,
                'lane_arrival_confirmed': True,
                'lane_settle_timer': 0,
                'pending_lane': None,
                'golden_lane_active': False,
                'golden_lane_target': None,
                'golden_lane_timer': 0,
                'golden_rearm_timer': 0,
                'golden_detect_score': 0.0,
                'golden_scan_counter': 0,
                'low_brightness': False,
                'front_brightness': 255.0,
                'last_front_brightness': 255.0,
                'front_road_brightness': 255.0,
                'front_center_brightness': 255.0,
                'front_darkness_metric': 255.0,
                'light_retry_cooldown': 0,
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
                'chasing_car_seen': False,
                'chasing_car_score': 0.0,
                'chasing_car_box': None,
                'chasing_car_reason': 'NONE',
                'chasing_prev_area': 0.0,
                'chasing_prev_bottom_y': 0,
                'chasing_prev_center_x': None,
                'chasing_lost_frames': 0,
                'debug_front_frame': None,
                'debug_back_frame': None,
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
            'target_lane': NUM_LANES // 2,
            'current_lane': NUM_LANES // 2,
            'settled_lane': NUM_LANES // 2,
            'lane_arrival_confirmed': True,
            'lane_settle_timer': 0,
            'pending_lane': None,
            'golden_lane_active': False,
            'golden_lane_target': None,
            'golden_lane_timer': 0,
            'golden_rearm_timer': 0,
            'golden_detect_score': 0.0,
            'golden_scan_counter': 0,

            # Darkness / light state
            'low_brightness': False,
            'front_brightness': 255.0,
            'last_front_brightness': 255.0,
            'front_road_brightness': 255.0,
            'front_center_brightness': 255.0,
            'front_darkness_metric': 255.0,
            'light_retry_cooldown': 0,
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
            'chasing_car_seen': False,
            'chasing_car_score': 0.0,
            'chasing_car_box': None,
            'chasing_car_reason': 'NONE',
            'chasing_prev_area': 0.0,
            'chasing_prev_bottom_y': 0,
            'chasing_prev_center_x': None,
            'chasing_lost_frames': 0,
            'debug_front_frame': None,
            'debug_back_frame': None,

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


def choose_trailing_escape_direction(current_lane, chase_center_x=None):
    """
    Choose escape direction when chasing car appears behind.

    If the chasing car is more on the right side of the back camera,
    steer left. If it is more on the left side, steer right.
    """
    if chase_center_x is not None:
        if chase_center_x > 172:
            return -1.0
        if chase_center_x < 148:
            return 1.0

    # Fallback if chasing car is directly centered.
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

def detect_back_chasing_car(back_frame):
    """
    Detect EV3 / EV4 chasing car from the back camera.

    Important:
    This detector should NOT fire for every cyan/blue object.
    It requires:
    1. cyan/turquoise car body,
    2. blue headlights or dark windshield/grille support,
    3. wide car-like bounding box,
    4. central back-road ROI,
    5. temporal approaching behaviour.
    """
    small_back = cv2.resize(back_frame, (320, 240))
    hsv = cv2.cvtColor(small_back, cv2.COLOR_BGR2HSV)
    debug_frame = small_back.copy()

    roi_mask = np.zeros((240, 320), dtype=np.uint8)
    back_road_polygon = np.array([
        [BACK_ROAD_LEFT_BOTTOM, BACK_ROAD_BOTTOM_Y],
        [BACK_ROAD_RIGHT_BOTTOM, BACK_ROAD_BOTTOM_Y],
        [BACK_ROAD_RIGHT_TOP, BACK_ROAD_TOP_Y],
        [BACK_ROAD_LEFT_TOP, BACK_ROAD_TOP_Y],
    ], dtype=np.int32)
    cv2.fillPoly(roi_mask, [back_road_polygon], 255)

    # Cyan / turquoise body of the chasing car.
    # Kept narrower than before so green tokens and background objects are rejected.
    cyan_mask = cv2.inRange(
        hsv,
        np.array([76, 55, 55]),
        np.array([104, 255, 255])
    )

    # Blue headlights.
    blue_mask = cv2.inRange(
        hsv,
        np.array([104, 85, 70]),
        np.array([136, 255, 255])
    )

    # Dark windshield / grille support.
    dark_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, 82])
    )

    cyan_mask = cv2.bitwise_and(cyan_mask, roi_mask)
    blue_mask = cv2.bitwise_and(blue_mask, roi_mask)
    dark_mask = cv2.bitwise_and(dark_mask, roi_mask)

    candidate_mask = cv2.bitwise_or(cyan_mask, blue_mask)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, CHASE_KERNEL)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, CHASE_KERNEL)
    candidate_mask = cv2.dilate(candidate_mask, CHASE_KERNEL, iterations=1)

    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    rejected_candidates = []

    for contour in contours:
        metrics = token_contour_metrics(contour)
        contour_area = cv2.contourArea(contour)
        if contour_area < 80:
            if len(rejected_candidates) < 6:
                rejected_candidates.append({
                    'box': cv2.boundingRect(contour),
                    'reason': 'size',
                })
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bottom_y = y + h
        center_x = x + (w // 2)
        box_area = w * h

        gate_reason = object_gate_reason(
            metrics,
            road_mask=roi_mask,
            edge_func=back_road_edges_at_y,
            edge_margin=8,
            shape_kind='wide-car',
            min_area=80,
            max_area=12000,
        )
        if gate_reason is not None:
            if len(rejected_candidates) < 6:
                rejected_candidates.append({
                    'box': (x, y, w, h),
                    'reason': gate_reason,
                })
            continue

        if w < CHASE_MIN_WIDTH or h < CHASE_MIN_HEIGHT:
            if len(rejected_candidates) < 6:
                rejected_candidates.append({
                    'box': (x, y, w, h),
                    'reason': 'size',
                })
            continue

        # Far tiny blobs are usually tokens/signs/background.
        if bottom_y < 88:
            if len(rejected_candidates) < 6:
                rejected_candidates.append({
                    'box': (x, y, w, h),
                    'reason': 'far',
                })
            continue

        aspect_ratio = w / float(max(h, 1))
        fill_ratio = contour_area / float(max(box_area, 1))

        # Reject round token-like objects and thin road/sign fragments.
        if aspect_ratio < CHASE_MIN_ASPECT or aspect_ratio > CHASE_MAX_ASPECT:
            if len(rejected_candidates) < 6:
                rejected_candidates.append({
                    'box': (x, y, w, h),
                    'reason': 'bad-shape',
                })
            continue
        if fill_ratio < CHASE_MIN_FILL_RATIO:
            if len(rejected_candidates) < 6:
                rejected_candidates.append({
                    'box': (x, y, w, h),
                    'reason': 'too-thin',
                })
            continue

        # Chasing car should appear around the centre road corridor.
        if abs(center_x - 160) > CHASE_CENTER_TOLERANCE:
            if len(rejected_candidates) < 6:
                rejected_candidates.append({
                    'box': (x, y, w, h),
                    'reason': 'edge',
                })
            continue

        x1 = max(0, x - 10)
        y1 = max(0, y - 10)
        x2 = min(320, x + w + 10)
        y2 = min(240, y + h + 10)

        cyan_pixels = cv2.countNonZero(cyan_mask[y1:y2, x1:x2])
        blue_pixels = cv2.countNonZero(blue_mask[y1:y2, x1:x2])
        dark_pixels = cv2.countNonZero(dark_mask[y1:y2, x1:x2])

        if cyan_pixels < CHASE_MIN_CYAN_PIXELS:
            if len(rejected_candidates) < 6:
                rejected_candidates.append({
                    'box': (x, y, w, h),
                    'reason': 'color',
                })
            continue

        has_car_support = (
            blue_pixels >= CHASE_MIN_BLUE_PIXELS
            or dark_pixels >= CHASE_MIN_DARK_PIXELS
            or (cyan_pixels >= CHASE_MIN_CYAN_PIXELS * 2 and w >= CHASE_MIN_WIDTH + 18)
        )
        if not has_car_support:
            if len(rejected_candidates) < 6:
                rejected_candidates.append({
                    'box': (x, y, w, h),
                    'reason': 'no-support',
                })
            continue

        center_error = abs(center_x - 160)

        score = (
            cyan_pixels * 2.05
            + blue_pixels * 2.20
            + dark_pixels * 0.18
            + bottom_y * 3.80
            + w * 4.20
            + h * 2.10
            + aspect_ratio * 45.0
            + fill_ratio * 120.0
            - center_error * 1.15
        )

        if blue_pixels >= CHASE_MIN_BLUE_PIXELS:
            score += 130.0
        if dark_pixels >= CHASE_MIN_DARK_PIXELS:
            score += 70.0
        if bottom_y >= CHASE_CLOSE_BOTTOM_Y:
            score += 120.0

        item = {
            'box': (x1, y1, x2 - x1, y2 - y1),
            'raw_box': (x, y, w, h),
            'center_x': center_x,
            'bottom_y': bottom_y,
            'area': float(box_area),
            'contour_area': float(contour_area),
            'aspect_ratio': aspect_ratio,
            'fill_ratio': fill_ratio,
            'cyan_pixels': cyan_pixels,
            'blue_pixels': blue_pixels,
            'dark_pixels': dark_pixels,
            'score': score,
        }

        if best is None or item['score'] > best['score']:
            best = item

    cv2.polylines(debug_frame, [back_road_polygon], True, (255, 255, 255), 1)

    seen = False
    approaching = False
    already_close = False
    reason = 'NONE'
    previous_area = 0.0
    previous_bottom_y = 0
    previous_center_x = None
    area_gain = 0.0
    bottom_gain = 0

    if best is not None:
        with data_lock:
            previous_area = float(shared_data.get('chasing_prev_area', 0.0))
            previous_bottom_y = int(shared_data.get('chasing_prev_bottom_y', 0))
            previous_center_x = shared_data.get('chasing_prev_center_x')

        if previous_area > 0.0:
            area_gain = best['area'] / max(previous_area, 1.0)
        bottom_gain = int(best['bottom_y'] - previous_bottom_y)

        center_stable = (
            previous_center_x is None
            or abs(best['center_x'] - int(previous_center_x)) <= CHASE_MAX_CENTER_JUMP
        )

        # Real chasing car should become bigger or move downward in the back view.
        approaching = (
            center_stable
            and previous_area > 0.0
            and (
                area_gain >= CHASE_APPROACH_AREA_GAIN
                or bottom_gain >= CHASE_APPROACH_BOTTOM_GAIN
            )
        )

        # Backup: if it is already large/close, do not wait too long.
        already_close = (
            best['bottom_y'] >= CHASE_CLOSE_BOTTOM_Y
            and best['area'] >= CHASE_CLOSE_AREA
            and best['score'] >= CHASE_CLOSE_SCORE
        )

        strong_candidate = best['score'] >= CHASE_MIN_SCORE

        if strong_candidate and approaching:
            seen = True
            reason = 'CLOSE_APPROACH' if already_close else 'APPROACHING'

        with data_lock:
            shared_data['chasing_prev_area'] = best['area']
            shared_data['chasing_prev_bottom_y'] = best['bottom_y']
            shared_data['chasing_prev_center_x'] = best['center_x']
            shared_data['chasing_lost_frames'] = 0

        x, y, w, h = best['box']
        color = (0, 255, 255) if seen else (0, 165, 255)
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            debug_frame,
            (
                f"chase={reason} score={best['score']:.0f} "
                f"area={best['area']:.0f} bottom={best['bottom_y']}"
            ),
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_frame,
            (
                f"approach={approaching} close={already_close} "
                f"gain={area_gain:.2f} dy={bottom_gain} "
                f"cyan={best['cyan_pixels']} blue={best['blue_pixels']}"
            ),
            (5, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    for rejected in rejected_candidates:
        x, y, w, h = rejected['box']
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (80, 80, 255), 1)
        cv2.putText(
            debug_frame,
            rejected['reason'],
            (x, max(10, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (80, 80, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        with data_lock:
            shared_data['chasing_lost_frames'] += 1
            if shared_data['chasing_lost_frames'] >= CHASE_MEMORY_LOST_FRAMES:
                shared_data['chasing_prev_area'] = 0.0
                shared_data['chasing_prev_bottom_y'] = 0
                shared_data['chasing_prev_center_x'] = None

        cv2.putText(
            debug_frame,
            "chase=NONE",
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    debug_info = {
        'seen': seen,
        'reason': reason,
        'box': best['box'] if best is not None else None,
        'center_x': best['center_x'] if best is not None else None,
        'bottom_y': best['bottom_y'] if best is not None else 0,
        'area': best['area'] if best is not None else 0.0,
        'score': best['score'] if best is not None else 0.0,
        'approaching': approaching,
        'already_close': already_close,
        'area_gain': area_gain,
        'bottom_gain': bottom_gain,
        'rejected_candidates': rejected_candidates,
        'debug_frame': debug_frame,
    }

    return seen, debug_info


def choose_frame_action(
    action_kind,
    steering_target,
    low_brightness,
    critical_darkness,
    back_steering_escape=0.0,
):
    """
    Single per-frame control arbiter.

    Steering taps live here so emergency priorities can override cooldowns
    instead of being blocked by stale lower-priority lane-following state.
    """
    with data_lock:
        light_pulse_active = shared_data['light_signal_timer'] > 0
        if light_pulse_active:
            shared_data['light_signal_timer'] -= 1

        if shared_data.get('lane_settle_timer', 0) > 0:
            shared_data['lane_settle_timer'] -= 1
            if shared_data['lane_settle_timer'] == 0 and shared_data.get('pending_lane') is not None:
                # Separate from current_lane because current_lane is still an
                # optimistic/intended lane used by existing cooldown logic.
                shared_data['settled_lane'] = int(shared_data['pending_lane'])
                shared_data['lane_arrival_confirmed'] = True
                shared_data['pending_lane'] = None

        emergency_action = action_kind in {
            'police',
            'police_collision',
            'police_red',
            'red_yellow',
            'chasing',
            'golden',
        }

        if emergency_action:
            shared_data['tap_timer'] = 0
            shared_data['cooldown_timer'] = 0
            shared_data['tap_steering'] = steering_target

            if action_kind == 'chasing':
                steering = back_steering_escape
                acceleration = TRAILING_ESCAPE_ACCELERATION
                shared_data['trailing_detect_count'] = 0
                shared_data['trailing_escape_cooldown'] = TRAILING_ESCAPE_COOLDOWN_FRAMES
                lane_delta = -1 if back_steering_escape < 0 else 1
                next_lane = shared_data.get('current_lane', NUM_LANES // 2) + lane_delta
                shared_data['target_lane'] = int(np.clip(next_lane, 0, NUM_LANES - 1))
                shared_data['pending_lane'] = shared_data['target_lane']
                shared_data['lane_arrival_confirmed'] = False
                shared_data['lane_settle_timer'] = LANE_SETTLE_FRAMES
                shared_data['current_lane'] = shared_data['target_lane']
            else:
                steering = steering_target
                if action_kind == 'police_collision':
                    acceleration = POLICE_COLLISION_ACCELERATION
                elif action_kind == 'police':
                    acceleration = POLICE_DODGE_ACCELERATION
                elif action_kind == 'police_red':
                    acceleration = POLICE_RED_COLLECTION_ACCELERATION
                elif action_kind == 'golden':
                    acceleration = NORMAL_ACCELERATION
                    if shared_data.get('golden_lane_target') is not None:
                        golden_target = int(shared_data['golden_lane_target'])
                        shared_data['target_lane'] = golden_target
                        if shared_data.get('pending_lane') != golden_target:
                            shared_data['pending_lane'] = golden_target
                            shared_data['lane_arrival_confirmed'] = False
                            shared_data['lane_settle_timer'] = LANE_SETTLE_FRAMES
                else:
                    acceleration = RED_YELLOW_DODGE_ACCELERATION

            # Emergency acceleration floors outrank Darkness and light pulse.
            reason = action_kind

        else:
            if shared_data['tap_timer'] > 0:
                shared_data['tap_timer'] -= 1
                steering = shared_data['tap_steering']
                if shared_data['tap_timer'] == 0:
                    shared_data['cooldown_timer'] = (
                        DARKNESS_COOLDOWN_FRAMES
                        if shared_data['low_brightness']
                        else NORMAL_COOLDOWN_FRAMES
                    )
                reason = 'normal-tap'
            elif shared_data['cooldown_timer'] > 0:
                shared_data['cooldown_timer'] -= 1
                steering = 0.0
                reason = 'normal-cooldown'
            elif steering_target != 0.0:
                shared_data['tap_steering'] = steering_target
                shared_data['tap_timer'] = (
                    DARKNESS_TAP_FRAMES
                    if shared_data['low_brightness']
                    else NORMAL_TAP_FRAMES
                )
                steering = steering_target
                reason = 'green-or-lane'
            else:
                steering = 0.0
                reason = 'normal'

            if light_pulse_active:
                # Light pulse is acceleration-only; it must not erase steering.
                acceleration = LIGHT_TOGGLE_ACCELERATION
                reason = f"{reason}+light"
            elif low_brightness:
                acceleration = (
                    CRITICAL_DARKNESS_ACCELERATION
                    if critical_darkness
                    else DARKNESS_ACCELERATION
                )
                reason = f"{reason}+dark"
            else:
                acceleration = NORMAL_ACCELERATION

        shared_data['steering_input'] = float(steering)
        shared_data['acceleration_input'] = float(acceleration)

    return {
        'steering': float(steering),
        'acceleration': float(acceleration),
        'reason': reason,
    }

def processing_task():
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data['latest_back_frame']
    police_mode = shared_data['police_active']
    
    steering_target = 0.0
    action_kind = 'normal'
    frame_center = 160
    
    # ---------------------------------------------------------
    # BACK CAMERA ENVIRONMENT ANALYSIS (Chasing Car Detection)
    # ---------------------------------------------------------
    evade_back_car = False
    back_steering_escape = 0.0

    if back_frame is not None:
        chase_seen, chase_debug = detect_back_chasing_car(back_frame)

        with data_lock:
            if shared_data['trailing_escape_cooldown'] > 0:
                shared_data['trailing_escape_cooldown'] -= 1

            if chase_seen:
                shared_data['trailing_detect_count'] += 1
            else:
                shared_data['trailing_detect_count'] = 0

            shared_data['chasing_car_seen'] = chase_seen
            shared_data['chasing_car_score'] = chase_debug.get('score', 0.0)
            shared_data['chasing_car_box'] = chase_debug.get('box')
            shared_data['chasing_car_reason'] = chase_debug.get('reason', 'NONE')
            shared_data['debug_back_frame'] = chase_debug.get('debug_frame')

            shared_data['trailing_debug_tick'] = (
                shared_data['trailing_debug_tick'] + 1
            ) % 30

            should_print_watch = shared_data['trailing_debug_tick'] == 0

            confirmed_rear_car = (
                shared_data['trailing_detect_count'] >= TRAILING_CONFIRM_FRAMES
                and shared_data['trailing_escape_cooldown'] == 0
            )

        if should_print_watch:
            print(
                "Chasing Watch: "
                f"{TRAILING_DEBUG_VERSION} "
                f"seen={chase_seen}, "
                f"score={chase_debug.get('score', 0.0):.0f}, "
                f"bottom={chase_debug.get('bottom_y', 0)}, "
                f"center_x={chase_debug.get('center_x', None)}, "
                f"reason={chase_debug.get('reason', 'NONE')}"
            )

        if confirmed_rear_car:
            evade_back_car = True

            with data_lock:
                current_lane_for_escape = (
                    shared_data.get('settled_lane', NUM_LANES // 2)
                    if shared_data.get('lane_arrival_confirmed', False)
                    else shared_data.get('current_lane', NUM_LANES // 2)
                )

            back_steering_escape = choose_trailing_escape_direction(
                current_lane_for_escape,
                chase_debug.get('center_x')
            )

            print(
                "Chasing Car Alert! "
                f"{TRAILING_DEBUG_VERSION} "
                f"score={chase_debug.get('score', 0.0):.0f}, "
                f"bottom={chase_debug.get('bottom_y', 0)}, "
                f"center_x={chase_debug.get('center_x', None)}, "
                f"steering={back_steering_escape}"
            )

    # ---------------------------------------------------------
    # FRONT CAMERA ENVIRONMENT ANALYSIS (Token Processing)
    # ---------------------------------------------------------
    if front_frame is not None:
        small_frame = cv2.resize(front_frame, (320, 240))
        game_state, race_elapsed, race_phase = update_race_clock_from_frame(small_frame)
        update_golden_lane_state(small_frame)
        with data_lock:
            golden_lane_active = shared_data.get('golden_lane_active', False)
            golden_lane_target = shared_data.get('golden_lane_target')
        
        # ---------------------------------------------------------
        # NEW: Darkness Improvement
        # ---------------------------------------------------------
        brightness_info = estimate_front_darkness_metric(small_frame)
        brightness = brightness_info['metric']
        low_brightness = update_darkness_state(brightness, brightness_info)
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

        # Grey/white boulder hazards can be circular too. Keep them in their
        # own mask and never feed them into token contours; tokens must pass
        # both circularity and green/red/yellow HSV color membership.
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
        
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_green_contours, rejected_green_tokens = classify_token_contours(
            contours_green,
            'G',
            road_mask,
        )
        filtered_red_contours, rejected_red_tokens = classify_token_contours(
            contours_red,
            'R',
            road_mask,
        )
        filtered_yellow_contours, rejected_yellow_tokens = classify_token_contours(
            contours_yellow,
            'Y',
            road_mask,
        )
        rejected_token_observations = (
            rejected_green_tokens + rejected_red_tokens + rejected_yellow_tokens
        )
        token_observations = (
            collect_token_observations(filtered_green_contours, 'G', road_mask)
            + collect_token_observations(filtered_red_contours, 'R', road_mask)
            + collect_token_observations(filtered_yellow_contours, 'Y', road_mask)
        )
        nearest_token_observation = None
        if token_observations:
            nearest_token_observation = max(
                token_observations,
                key=lambda item: (item['cy'], item['y'] + item['h']),
            )

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
                    # Legitimate tokens in the scan slice are compact circles,
                    # not rectangular grass/road fragments.
                    if is_circular_token_shape(sc, min_area=6, max_area=450):
                        valid_green = True
                        break

            valid_danger = False
            if cv2.countNonZero(red_roi) > PROXIMITY_PIXEL_THRESHOLD or cv2.countNonZero(yellow_roi) > PROXIMITY_PIXEL_THRESHOLD:
                # Validate Danger tokens similarly to ensure it's not the curb track border line splitting
                combined_danger_mask = cv2.bitwise_or(red_roi, yellow_roi)
                sub_contours_d, _ = cv2.findContours(combined_danger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for sc in sub_contours_d:
                    if is_circular_token_shape(sc, min_area=6, max_area=450):
                        valid_danger = True
                        break
            
            # Assign states based on filtered structural results
            if valid_danger:
                lane_states[i] = 'Danger'
            elif not critical_darkness and valid_green:
                lane_states[i] = 'Target'

        # Wider look-ahead danger pass. Red/yellow always overrides green.
        # This reduces red-token collection when red tokens are slightly before
        # or after the thin proximity scan line.
        danger_lanes_from_red = mark_danger_lanes_from_token_contours(
            lane_states,
            filtered_red_contours,
            'R',
            road_mask,
        )
        danger_lanes_from_yellow = mark_danger_lanes_from_token_contours(
            lane_states,
            filtered_yellow_contours,
            'Y',
            road_mask,
        )
        danger_lanes_from_tokens = danger_lanes_from_red | danger_lanes_from_yellow

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
        red_emergency_avoidance = False

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
            action_kind = 'chasing'
            print(f"Trailing Car Alert! Escaping to steering: {steering_target}")
        elif police_hard_immediate_dodge:
            # Absolute priority: police is already close/low in the front camera.
            # Bypass red-token chasing and normal cooldown immediately.
            steering_target = police_escape_steering
            police_urgent_avoidance = True
            action_kind = 'police'
            print(f"HARD Police Dodge NOW! Steering: {steering_target}")
        elif police_collision_imminent:
            # Priority 1: Do not hit the police car head-on
            steering_target = police_escape_steering
            police_urgent_avoidance = True
            action_kind = 'police_collision'
            print(f"Collision Imminent! Full police dodge: {steering_target}")
        elif police_detected_now:
            # Priority 2: The moment a police candidate is detected in the
            # driving corridor, evade immediately without extra delay.
            steering_target = police_escape_steering
            police_urgent_avoidance = True
            action_kind = 'police'
            print(f"Instant Police Avoidance! Steering: {steering_target}")
        elif police_sudden_risk:
            # Priority 3: Crest/visibility-change protection. If a strong
            # police-like obstacle suddenly appears ahead, evade immediately.
            steering_target = police_escape_steering
            police_urgent_avoidance = True
            action_kind = 'police'
            print(f"Sudden Police Risk! Steering: {steering_target}")
        elif police_emergency_close:
            # Priority 4: Even before the police event fully confirms, treat a
            # strong close police candidate as an obstacle to avoid game over.
            steering_target = police_escape_steering
            police_urgent_avoidance = True
            action_kind = 'police'
            print(f"Emergency Police Avoidance! Steering: {steering_target}")
            
        elif police_mode:
            # Priority 5: During a confirmed police event, avoid the police car
            # first. Only when it is safe, chase one real red token within the
            # 5s window. Do not use largest_red because police lights/curb can
            # be larger than the actual EV2 red token.
            if police_obstacle_close:
                steering_target = police_escape_steering
                police_urgent_avoidance = True
                action_kind = 'police'
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
                        action_kind = 'police_red'
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

        elif golden_lane_active and golden_lane_target is not None:
            action_kind = 'golden'
            ego_lane = NUM_LANES // 2
            if golden_lane_target < ego_lane:
                steering_target = -1.0
            elif golden_lane_target > ego_lane:
                steering_target = 1.0
            else:
                steering_target = 0.0

        else:
            # Priority 3: Standard Navigation using the 5-Bucket Logic Engine
            ego_lane = 2  # The camera is fixed to the car, so the car is always in lane 2
            current_state = lane_states[ego_lane]
            best_lane = ego_lane
            
            if current_state == 'Danger':
                red_emergency_avoidance = True
                action_kind = 'red_yellow'
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
                            
            # If the centre lane is dangerous but all lanes look bad, still move away.
            if red_emergency_avoidance and best_lane == ego_lane:
                left_score = 0
                right_score = 0
                if ego_lane - 1 >= 0 and lane_states[ego_lane - 1] != 'Danger':
                    left_score += 1
                if ego_lane + 1 < NUM_LANES and lane_states[ego_lane + 1] != 'Danger':
                    right_score += 1
                best_lane = ego_lane - 1 if left_score >= right_score and ego_lane > 0 else min(NUM_LANES - 1, ego_lane + 1)

            # Fire Steering Commands
            if best_lane < ego_lane:
                steering_target = -1.0
                print(f"Logic Engine: Steering LEFT to Lane {best_lane}")
            elif best_lane > ego_lane:
                steering_target = 1.0
                print(f"Logic Engine: Steering RIGHT to Lane {best_lane}")

        arbiter_action = choose_frame_action(
            action_kind,
            steering_target,
            low_brightness,
            critical_darkness,
            back_steering_escape=back_steering_escape,
        )

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

        car_debug_point = (frame_center, ROAD_BOTTOM_Y - 8)
        token_debug_colors = {
            'G': (0, 255, 0),
            'R': (0, 0, 255),
            'Y': (0, 255, 255),
        }
        for rejected in rejected_token_observations[:10]:
            rejected_center = (int(rejected['cx']), int(rejected['cy']))
            cv2.circle(
                debug_frame,
                rejected_center,
                int(rejected['radius']),
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_frame,
                rejected['reason'],
                (rejected_center[0] + 3, max(8, rejected_center[1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.26,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )
        if nearest_token_observation is not None:
            token_color = token_debug_colors.get(nearest_token_observation['color'], (0, 255, 0))
            token_center = (
                int(nearest_token_observation['cx']),
                int(nearest_token_observation['cy']),
            )
            token_radius = int(nearest_token_observation['radius'])
            cv2.line(
                debug_frame,
                car_debug_point,
                token_center,
                token_color,
                1,
                cv2.LINE_AA,
            )
            cv2.circle(
                debug_frame,
                token_center,
                token_radius,
                token_color,
                2,
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
            overlay_golden_active = shared_data.get('golden_lane_active', False)
            overlay_golden_target = shared_data.get('golden_lane_target')
            overlay_golden_timer = shared_data.get('golden_lane_timer', 0) * TASK_PERIOD_SECONDS
            overlay_golden_score = shared_data.get('golden_detect_score', 0.0)
            overlay_golden_positioned = (
                shared_data.get('lane_arrival_confirmed', False)
                and shared_data.get('settled_lane') == overlay_golden_target
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
                f"metric={brightness:.1f} road={brightness_info['road_mean']:.1f}"
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

        cv2.putText(
            debug_frame,
            (
                f"golden={'ON' if overlay_golden_active else 'OFF'} "
                f"target={overlay_golden_target} "
                f"left={overlay_golden_timer:.1f}s "
                f"pos={'Y' if overlay_golden_positioned else 'N'} "
                f"score={overlay_golden_score:.2f}"
            ),
            (5, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            overlay_font_scale,
            overlay_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            debug_frame,
            (
                f"action={arbiter_action['reason']} | "
                f"steer={arbiter_action['steering']:.1f} "
                f"accel={arbiter_action['acceleration']:.2f}"
            ),
            (5, 72),
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
        is_server_running = any(
            t.name == "ControlServerRecovery"
            for t in threading.enumerate()
        )
        if not is_server_running:
            print("Connection missing. Restarting control server...")
            threading.Thread(
                target=setup_control_server,
                name="ControlServerRecovery",
                daemon=True
            ).start()
        return

    with data_lock:
        steering_to_send = shared_data['steering_input']
        acceleration_to_send = shared_data['acceleration_input']

    try:
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
                debug_back = shared_data.get('debug_back_frame')

            front_to_show = debug_front if debug_front is not None else front
            if front_to_show is not None:
                cv2.imshow("Front Camera - AI Driving", cv2.resize(front_to_show, (640, 480)))
            back_to_show = debug_back if debug_back is not None else back
            if back_to_show is not None:
                cv2.imshow("Back Camera", cv2.resize(back_to_show, (640, 480)))

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


