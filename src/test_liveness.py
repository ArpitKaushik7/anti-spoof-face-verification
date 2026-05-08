import cv2
from antispoof.liveness import is_live_face

cap = cv2.VideoCapture(0)

print("Checking liveness...")
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        break

    live, conf = is_live_face(frame)   # ✅ correct

    if live:
        print(f"REAL ✅ ({conf:.2f})")
    else:
        print(f"SPOOF ❌ ({conf:.2f})")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()