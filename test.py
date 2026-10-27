from ultralytics import YOLO
import cv2
import datetime
import firebase_admin
from firebase_admin import credentials, db

# 1. Firebase Initialize karein
# (Apni Firebase project ki downloaded JSON file ka path yahan dein)
cred = credentials.Certificate('C:/Users/NICAT/Desktop/drone_new/serviceAccountKey.json') 
firebase_admin.initialize_app(cred, {
    'databaseURL': 'your firebase project link'
})

# Firebase reference set karein jo aapke database structure ('DroneMonitoring') ke mutabiq hai
ref = db.reference('DroneMonitoring')

# 2. Apne trained model ka path dein
model = YOLO('C:/Users/NICAT/Desktop/drone_new/runs/detect/train/weights/best.pt')

# 3. Mobile IP Webcam ka URL yahan dalein 
ip_webcam_url = "http://10.10.60.234:8080/video"  # Apne mobile ka IP yahan change karein

cap = cv2.VideoCapture(ip_webcam_url)

if not cap.isOpened():
    print("Error: IP Webcam connect nahi ho saka. URL ya Wi-Fi connection check karein!")
    exit()

print("Mobile IP Webcam live detection aur Firebase sync shuru ho gayi hai. Band karne ke liye 'q' press karein.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame read nahi ho saka ya connection disconnect ho gaya.")
        break

    # 4. Model se frame par prediction karwayein
    results = model(frame, conf=0.5)

    # 5. Drones count calculate karein
    drone_count = 0
    boxes = results[0].boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        # Note: Agar aapke model mein Drone ki class ID mukhtalif hai (maslan 1 ya 0) toh apne data.yaml ke mutabiq check kar lein
        if cls_id == 1: 
            drone_count += 1

    # Status determine karein
    if drone_count > 0:
        status_text = "Danger Detected!"
    else:
        status_text = "Safe"

    # Current timestamp generate karein
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 6. Firebase Realtime Database par values update karein
    ref.set({
        'active_drones': drone_count,
        'status': status_text,
        'timestamp': current_time
    })

    # 7. Results ko frame par plot karein
    annotated_frame = results[0].plot()

    # 8. Live video window dikhayein
    cv2.imshow("YOLOv8 Mobile IP Webcam Detection", annotated_frame)

    # 'q' dabane par window band ho jayegi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
