import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort 
from collections import deque
#tối ưu hóa bằng sử dụng bất đồng bộ
import threading
import queue

# Hàng đợi để truyền frame giữa các luồng
frame_queue = queue.Queue(maxsize=1000)  # Hàng đợi cho frame gốc
processed_frame_queue = queue.Queue(maxsize=1000)  # Hàng đợi cho frame đã xử lý

# Biến cờ để báo hiệu dừng chương trình
stop_event = threading.Event()

# Dictionary để lưu lịch sử vị trí của các track
track_history = {}

# Biến để lưu màu đèn giao thông
color_light_traffic = ""

# Tải hai mô hình YOLOv8
model_helmet = YOLO("helmet_model.pt")  # Mô hình phát hiện nón bảo hiểm
model_vehicle = YOLO("vehicle_model.pt")  # Mô hình phát hiện phương tiện
print(model_helmet.names)
print(model_vehicle.names)

# Khởi tạo DeepSort để theo dõi phương tiện
track_vehicle = DeepSort(max_age=100, n_init=1, nn_budget=50)

def detect_traffic_light_color(light_roi):
    # Kiểm tra light_roi
    if light_roi is None or light_roi.size == 0:
        print("light_roi is empty!")
        return "Unknown"
    
    # Tính diện tích vùng
    h, w = light_roi.shape[:2]
    area = w * h
    if area < 100:  # Diện tích quá nhỏ
        print("Region too small to analyze!")
        return "Unknown"
    
    # Làm mịn hình ảnh để giảm nhiễu
    light_roi = cv2.GaussianBlur(light_roi, (5, 5), 0)
    
    # Chuyển đổi sang không gian màu HSV
    hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa các ngưỡng màu
    low_red1 = np.array([0, 15, 15])
    high_red1 = np.array([10, 255, 255])
    low_red2 = np.array([160, 30, 30])
    high_red2 = np.array([179, 255, 255])
    low_green = np.array([40, 30, 30])
    high_green = np.array([90, 255, 255])
    low_yellow = np.array([20, 30, 30])
    high_yellow = np.array([40, 255, 255])
    
    # Tạo mask cho từng màu
    mask_red1 = cv2.inRange(hsv, low_red1, high_red1)
    mask_red2 = cv2.inRange(hsv, low_red2, high_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, low_green, high_green)
    mask_yellow = cv2.inRange(hsv, low_yellow, high_yellow)
    
    # Tính tỷ lệ màu
    red_ratio = cv2.countNonZero(mask_red) / area
    green_ratio = cv2.countNonZero(mask_green) / area
    yellow_ratio = cv2.countNonZero(mask_yellow) / area
    
    # Chọn màu có tỷ lệ cao nhất
    ratios = {
        "Red": red_ratio,
        "Green": green_ratio,
        "Yellow": yellow_ratio
    }
    
    max_color = max(ratios, key=ratios.get)
    max_ratio = ratios[max_color]
    
    # Đặt ngưỡng tối thiểu để tránh nhiễu
    if max_ratio > 0.2:  # Ngưỡng tối thiểu 20%
        return max_color
    else:
        return color_light_traffic  # Trả về màu trước đó nếu không đủ rõ ràng

# Luồng đọc frame
def read_frames():
    video_path = "video\\Hải Phòng.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video. Kiểm tra đường dẫn!")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Hết video hoặc lỗi khi đọc frame.")
            break
        
        # Đẩy frame vào hàng đợi
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Bỏ qua nếu hàng đợi đầy
    
    cap.release()
    frame_queue.put(None)  # Thông báo dừng

# Luồng xử lý frame
def process_frames():
    global color_light_traffic
    frame_count = 0
    last_tracks = []  # Lưu trữ tracks từ lần cập nhật DeepSort trước đó

    while True:
        if stop_event.is_set():
            break
        try:
            frame = frame_queue.get(timeout=1.0)  # Lấy frame từ hàng đợi
        except queue.Empty:
            break
        if frame is None:  # Kết thúc nếu nhận được None
            break

        # Định nghĩa vùng quan tâm (ROI) - Toàn bộ frame
        roi = frame.copy()  # Sử dụng toàn bộ frame làm ROI

        #vị trí của vạch dừng
        x1_crosswalk, x2_crosswalk, y1_crosswalk, y2_crosswalk = 200, roi.shape[1]-200, 340, 320
    
        # Phát hiện đèn giao thông 
        x1_trafficligth, y1_trafficligth, x2_trafficligth, y2_trafficligth = roi.shape[1]-25, 130, roi.shape[1]-5, 180
        roi_light_traffic = frame[y1_trafficligth:y2_trafficligth, x1_trafficligth:x2_trafficligth]
        if frame_count % 5 == 0:  # Chỉ kiểm tra mỗi 5 frame
            color_light_traffic = detect_traffic_light_color(roi_light_traffic)
        cv2.rectangle(roi, (x1_trafficligth, y1_trafficligth), (x2_trafficligth,y2_trafficligth), (0, 255, 0), 1)
        cv2.putText(roi, color_light_traffic, (x1_trafficligth, y1_trafficligth - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Phát hiện phương tiện bằng model_vehicle
        detect_vehicle = []
        results_vehicle = model_vehicle.predict(source=frame, conf=0.5, iou=0.65)[0]
    
        for box in results_vehicle.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf.item()
            cls = int(box.cls.item())
            # Định dạng cho DeepSort: [bbox, confidence, class]
            detect_vehicle.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])
    
        # Theo dõi phương tiện bằng DeepSort (chỉ cập nhật mỗi 2 frame)
        if frame_count % 2 == 0:
            current_track_vehicle = track_vehicle.update_tracks(detect_vehicle, frame=frame)
            last_tracks = current_track_vehicle  # Lưu tracks để sử dụng cho frame tiếp theo
        else:
            current_track_vehicle = last_tracks  # Sử dụng tracks cũ
    
        # Vẽ bounding box cho phương tiện và phát hiện nón bảo hiểm
        for track in current_track_vehicle:
            if not (track.is_confirmed() and track.det_conf):
                continue
            # Lấy tọa độ bounding box của phương tiện
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = list(map(int, ltrb))
            track_id = track.track_id
            label = track.det_class
            confidence = track.det_conf
            
            # Vẽ bounding box cho phương tiện
            cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(roi, model_vehicle.names[label], (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            #xác định vi phạm vượt đèn đỏ
            # Tính tọa độ trung tâm của bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Lưu lịch sử vị trí
            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=5)  # Lưu tối đa 5 vị trí gần nhất
            track_history[track_id].append((center_x, center_y))
            
            # Tính vector di chuyển (nếu có ít nhất 2 vị trí)
            is_straight = False
            if len(track_history[track_id]) >= 2:
                prev_center_x, prev_center_y = track_history[track_id][-2]  # Vị trí trước đó
                delta_x = center_x - prev_center_x
                delta_y = center_y - prev_center_y
                
                # Xác định hướng di chuyển
                # Nếu delta_y < 0 (y giảm) và |deltqqa_x| nhỏ, xe di chuyển từ dưới lên (đường thẳng)
                if delta_y < 0 and abs(delta_x) < abs(delta_y) * 2 and x1 > x1_crosswalk and x2 < x2_crosswalk:  # |delta_x| nhỏ hơn 2 lần |delta_y|
                    is_straight = True
                    
            if color_light_traffic == "Red" and  (y2 < min(y1_crosswalk, y2_crosswalk) and y2 > min(y1_crosswalk, y2_crosswalk)-125 ) and is_straight:
                cv2.putText(roi, "Vuot den do", (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
               
            if label == 1: #nếu là xe máy
                 # Cắt vùng phương tiện để phát hiện nón bảo hiểm
                 crop_img = frame[y1-25:y2, x1:x2]
                 if crop_img.size != 0:  # Kiểm tra xem vùng cắt có hợp lệ không
                     # Phát hiện nón bảo hiểm bằng model_helmet
                     results_helmet = model_helmet.predict(source=crop_img, imgsz=320, iou=0.45)[0]
         
                     for helmet_box in results_helmet.boxes:
                         hx1, hy1, hx2, hy2 = map(int, helmet_box.xyxy[0].tolist())
                         hlabel = helmet_box.cls[0]
                         hconfidence = helmet_box.conf[0]
                         htext = f"{model_helmet.names[int(hlabel)]}"
         
                         # Vẽ bounding box cho nón bảo hiểm
                         if(hconfidence > 0.6): # Chỉ vẽ nếu nón bảo hiểm được phát hiện
                             color = (0, 0, 255) if model_helmet.names[int(hlabel)] == "Without_Helmet" else (0, 255, 0)
                             cv2.rectangle(roi, (x1 + hx1, y1-25 + hy1), (x1 + hx2, y1-25 + hy2), color, 1)
                             cv2.putText(roi, htext, (x1 + hx1, y1 + hy1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1, cv2.LINE_AA)
                           
        #xác định vi trí của vạch dừng
        cv2.line(roi, (x1_crosswalk,y1_crosswalk), (x2_crosswalk,y2_crosswalk), (255, 0, 0), 2) # Vẽ đường kẻ ngang  

        # Đẩy frame đã xử lý vào hàng đợi
        try:
            processed_frame_queue.put_nowait(roi)
        except queue.Full:
            pass

        frame_count+=1
    # Đánh dấu kết thúc bằng cách đẩy None vào hàng đợi
    processed_frame_queue.put(None)

# Luồng hiển thị frame
def display_frames():
    while True:
        frame = processed_frame_queue.get()  # Lấy frame từ hàng đợi
        if frame is None:  # Kết thúc nếu nhận được None
            break
        
        cv2.imshow("Phat hien vi pham giao", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    
    cv2.destroyAllWindows()

# Chạy các luồng
read_thread = threading.Thread(target=read_frames)
process_thread = threading.Thread(target=process_frames)
display_thread = threading.Thread(target=display_frames)

read_thread.start()
process_thread.start()
display_thread.start()

read_thread.join()
process_thread.join()
display_thread.join()