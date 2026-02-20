import cv2
import numpy as np
import torch
from model import CNN

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("best_mnist_cnn.pth", map_location=device))
model.eval()

MEAN = 0.1307
STD = 0.3081

# ================= LOAD HSV =================
hsv_value = np.load("hsv_value.npy")
lower_range = hsv_value[0]
upper_range = hsv_value[1]

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = None
kernel = np.ones((5,5), np.uint8)

x1, y1 = 0, 0
smooth_x, smooth_y = 0, 0
alpha = 0.4

prediction = None
confidence = 0

# ================= LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, 1)
    mask = cv2.dilate(mask, kernel, 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 800:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        cx = x + w // 2
        cy = y + h // 2

        smooth_x = int(alpha * cx + (1 - alpha) * smooth_x)
        smooth_y = int(alpha * cy + (1 - alpha) * smooth_y)

        if x1 == 0 and y1 == 0:
            x1, y1 = smooth_x, smooth_y
        else:
            cv2.line(canvas, (x1, y1), (smooth_x, smooth_y), (255,255,255), 12)

        x1, y1 = smooth_x, smooth_y
        cv2.circle(frame, (smooth_x, smooth_y), 8, (0,255,0), -1)

    else:
        x1, y1 = 0, 0

    frame = cv2.add(frame, canvas)

    # ================= SHOW PREDICTION BOX =================
    if prediction is not None:
        if confidence > 80:
            color = (0,255,0)
        elif confidence > 50:
            color = (0,255,255)
        else:
            color = (0,0,255)

        cv2.rectangle(frame, (20,20), (300,140), color, 3)
        cv2.putText(frame, f"Digit: {prediction}", (40,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, f"Conf: {confidence:.1f}%", (40,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, "P: Predict | C: Clear | ESC: Exit",
                (20, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow("Air Digit Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    # ================= PREDICT =================
    if key == ord('p'):
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours_digit, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_digit:
            c = max(contours_digit, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            digit = thresh[y:y+h, x:x+w]

            size = max(w,h)
            square = np.zeros((size,size), dtype=np.uint8)
            square[(size-h)//2:(size-h)//2+h,
                   (size-w)//2:(size-w)//2+w] = digit

            square = cv2.resize(square, (20,20))
            digit28 = np.zeros((28,28), dtype=np.uint8)
            digit28[4:24,4:24] = square

            coords = np.column_stack(np.where(digit28 > 0))
            if len(coords) > 0:
                y_center, x_center = coords.mean(axis=0)
                shiftx = int(14 - x_center)
                shifty = int(14 - y_center)
                M = np.float32([[1,0,shiftx],[0,1,shifty]])
                digit28 = cv2.warpAffine(digit28, M, (28,28))

            digit28 = digit28.astype("float32") / 255.0
            digit28 = (digit28 - MEAN) / STD

            tensor = torch.tensor(digit28).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1).max().item()*100

    if key == ord('c'):
        canvas = None
        prediction = None

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()