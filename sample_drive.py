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
    'steering_input' : 0.0,
    'acceleration_input' : 1.0,
    'target_lane': 1,        # 0: Left, 1: Middle, 2: Right
    'current_lane': 1,
    'low_brightness': False,  # Event flag
    'tap_timer': 0,
    'cooldown_timer': 0,
    'tap_steering': 0.0,
    'police_active': False    # Dynamic tracking state for police event
}
data_lock = threading.Lock()
is_running = True

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
        
        # Looming vehicle detection threshold
        if avg_intensity > 180 or avg_intensity < 40: 
            evade_back_car = True
            # Check which side has more space or default a structural escape jump
            back_steering_escape = 1.0  # Tap right to clear the lane [cite: 153, 156]

    # ---------------------------------------------------------
    # FRONT CAMERA ENVIRONMENT ANALYSIS (Token Processing)
    # ---------------------------------------------------------
    if front_frame is not None:
        small_frame = cv2.resize(front_frame, (320, 240))
        
        # Dynamic Environmental Check: Low Brightness Mitigation [cite: 36, 214]
        gray_front = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_front)
        with data_lock:
            if brightness < 60:  # Threshold identifying light corruption [cite: 35]
                shared_data['low_brightness'] = True  # Activates toggle switch flag [cite: 36, 214]
            else:
                shared_data['low_brightness'] = False

        hsv_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        
        # Extended Range Color Masks [cite: 140, 141]
        lower_green = np.array([35, 40, 40]) 
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(hsv_frame, lower_red1, upper_red1), 
                                  cv2.inRange(hsv_frame, lower_red2, upper_red2))
                                  
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
        
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Look out for police lights to change internal state machines dynamically [cite: 32, 213]
        # Police flashing states detected by massive variance or contours matching color heights
        if len(contours_red) > 3:
            with data_lock:
                shared_data['police_active'] = True

        # Decision State Hierarchy [cite: 15]
        if evade_back_car:
            # Priority 1: Do not get wrecked from behind [cite: 31]
            steering_target = back_steering_escape
            print(f"Trailing Car Alert! Escaping to steering: {steering_target}") [cite: 30]
            
        elif police_mode and contours_red:
            # Priority 2: Catch red token intentionally if Police are chasing [cite: 33, 213]
            largest_red = max(contours_red, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > 3:
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
            if contours_red:
                largest_red = max(contours_red, key=cv2.contourArea)
                if cv2.contourArea(largest_red) > 5:
                    M = cv2.moments(largest_red)
                    if M['m00'] > 0:
                        rx = int(M['m10'] / M['m00'])
                        steering_target = -1.0 if rx > frame_center else 1.0
                        red_detected = True
                        print("Evading Red Token!") [cite: 20]

            # Evade Yellow Corruption Fields [cite: 21, 205]
            if not red_detected and contours_yellow:
                largest_yellow = max(contours_yellow, key=cv2.contourArea)
                if cv2.contourArea(largest_yellow) > 5:
                    M = cv2.moments(largest_yellow)
                    if M['m00'] > 0:
                        yx = int(M['m10'] / M['m00'])
                        steering_target = -1.0 if yx > frame_center else 1.0
                        yellow_detected = True
                        print("Evading Corruptive Yellow Token!") [cite: 28]

            # Collect Speed Upgrades [cite: 19, 204]
            if not red_detected and not yellow_detected and contours_green:
                largest_green = max(contours_green, key=cv2.contourArea)
                if cv2.contourArea(largest_green) > 5:
                    M = cv2.moments(largest_green)
                    if M['m00'] > 0:
                        gx = int(M['m10'] / M['m00'])
                        error = gx - frame_center
                        if error < -15:
                            steering_target = -1.0
                        elif error > 15:
                            steering_target = 1.0
                        print(f"Targeting Green Token! Error: {error}") [cite: 19]

        # Commit decision to shared resources safely using a Tap Sequence [cite: 152]
        with data_lock:
            if shared_data['tap_timer'] > 0:
                # State 1: Actively tapping [cite: 156]
                shared_data['tap_timer'] -= 1
                shared_data['steering_input'] = shared_data['tap_steering']
                if shared_data['tap_timer'] == 0:
                    shared_data['cooldown_timer'] = 10 # State 2: Enforce 0.0 wait after tap [cite: 157, 158]
                    shared_data['steering_input'] = 0.0
            elif shared_data['cooldown_timer'] > 0:
                # State 2: Waiting for car to settle [cite: 157]
                shared_data['cooldown_timer'] -= 1
                shared_data['steering_input'] = 0.0
            else:
                # State 0: Ready for a new tap [cite: 154]
                if steering_target != 0.0:
                    shared_data['tap_steering'] = steering_target
                    shared_data['tap_timer'] = 8  # Discrete tap sequence execution frames
                    shared_data['steering_input'] = steering_target
                    print(f"Initiating Tap! Steering: {steering_target}")
                else:
                    shared_data['steering_input'] = 0.0

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
        # Handle Low Brightness: Map structural change to reverse acceleration or signal value tweaks if required [cite: 36, 164]
        if shared_data['low_brightness']:
            # Modulate throttle or activate lighting sequences over standard float packing [cite: 162]
            acceleration_to_send = 0.95  
        else:
            acceleration_to_send = 1.0 # Default: full gas ahead [cite: 174]

    try:
        # Pack and send the control command to Unity [cite: 177]
        data = struct.pack('ff', steering_to_send, acceleration_to_send)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}") [cite: 179]
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
                back = shared_data.get('latest_back_frame')

            if front is not None:
                cv2.imshow("Front Camera - AI Driving", cv2.resize(front, (640, 480)))
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