Experiment 1
// C++ code
const int trigPin = 9;
const int echoPin = 10;
const int ledPin = 7;
const int buzzerPin = 6;
long duration;
int distance;
void setup() {
pinMode(trigPin, OUTPUT);
pinMode(echoPin, INPUT);
pinMode(ledPin, OUTPUT);
pinMode(buzzerPin, OUTPUT);
Serial.begin(9600);
}
void loop() {
digitalWrite(trigPin, LOW);
delayMicroseconds(2);
digitalWrite(trigPin, HIGH);
delayMicroseconds(10);
digitalWrite(trigPin, LOW);
duration = pulseIn(echoPin, HIGH);
distance = duration * 0.034 / 2;
Serial.print("Distance: ");
Serial.println(distance);
if (distance >40) {
digitalWrite(ledPin, HIGH);
} else {
digitalWrite(ledPin, LOW);
}
if (distance < 10) {
digitalWrite(buzzerPin, HIGH);
} else {
digitalWrite(buzzerPin, LOW);
}
delay(500);

Experiment 4


from google.colab import files
uploaded = files.upload()
import cv2
import numpy as np
import matplotlib.pyplot as plt
# -----Load image
img = cv2.imread('image1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# ----Convert to Lab
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
# -----Reshape for K-means
Z = lab.reshape((-1,3))
Z = np.float32(Z)
# -----K-means
k = 4
_, labels, centers = cv2.kmeans(Z, k, None,
(cv2.TERM_CRITERIA_EPS +
cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
10, cv2.KMEANS_RANDOM_CENTERS)
# Recreate segmented image
centers = np.uint8(centers)
res = centers[labels.flatten()]
segmented = res.reshape(lab.shape)
segmented = cv2.cvtColor(segmented, cv2.COLOR_LAB2RGB)
# Show original & segmented
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(segmented)
plt.title("Segmented")
plt.axis('off')
plt.show()
# Show individual segments
labels2D = labels.reshape(img.shape[:2])
for i in range(k):
mask = (labels2D == i)
seg = np.zeros_like(img)
seg[mask] = img[mask]
plt.imshow(seg)
plt.title(f"Segment {i}")
plt.axis('off')
plt.show()
EXPERIMENT 5 
!pip install opencv-python numpy matplotlib import 
cv2 
from google.colab.patches import cv2_imshow 
from google.colab import files uploaded = 
files.upload() filename = 
list(uploaded.keys())[0] 
# Step 3: Read image img = 
cv2.imread(filename) 
# Step 4: Check if loaded if img is None: print("❌
Error: Image not loaded. Check file format.") else: 
 print("✅ Image loaded successfully") 
cv2_imshow(img) print("Original 
Image:") cv2_imshow(img) 
 # -----------------------------
 # Step 3: Convert to Grayscale 
 # -----------------------------
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
print("Grayscale Image:") cv2_imshow(gray) 
 # -----------------------------
 # Step 4: Gaussian Blur # ------------
----------------- blur = 
Sarthaki Asane | 3
cv2.GaussianBlur(gray, (5,5), 0) 
print("Blurred Image:") 
cv2_imshow(blur) 
 # -----------------------------
 # Step 5: Edge Detection # -----
------------------------ edges = 
cv2.Canny(blur, 100, 200) 
print("Edge Detection:") 
cv2_imshow(edges) 
 # -----------------------------
 # Step 6: Thresholding (Segmentation) 
 # -----------------------------
 _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
print("Threshold Image:") cv2_imshow(thresh) 
 # ----------------------------- 
# Step 7: Contour Detection 
 # -----------------------------
 contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, 
cv2.CHAIN_APPROX_SIMPLE) 
 contour_img = img.copy() 
cv2.drawContours(contour_img, contours, -1, (0,255,0), 2) 


Experiment 5 
!pip install opencv-python numpy matplotlib import 
cv2 
from google.colab.patches import cv2_imshow 
from google.colab import files uploaded = 
files.upload() filename = 
list(uploaded.keys())[0] 
# Step 3: Read image img = 
cv2.imread(filename) 
# Step 4: Check if loaded if img is None: print("❌
Error: Image not loaded. Check file format.") else: 
 print("✅ Image loaded successfully") 
cv2_imshow(img) print("Original 
Image:") cv2_imshow(img) 
 # -----------------------------
 # Step 3: Convert to Grayscale 
 # -----------------------------
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
print("Grayscale Image:") cv2_imshow(gray) 
 # -----------------------------
 # Step 4: Gaussian Blur # ------------
----------------- blur = 

cv2.GaussianBlur(gray, (5,5), 0) 
print("Blurred Image:") 
cv2_imshow(blur) 
 # -----------------------------
 # Step 5: Edge Detection # -----
------------------------ edges = 
cv2.Canny(blur, 100, 200) 
print("Edge Detection:") 
cv2_imshow(edges) 
 # -----------------------------
 # Step 6: Thresholding (Segmentation) 
 # -----------------------------
 _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
print("Threshold Image:") cv2_imshow(thresh) 
 # ----------------------------- 
# Step 7: Contour Detection 
 # -----------------------------
 contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, 
cv2.CHAIN_APPROX_SIMPLE) 
 contour_img = img.copy() 
cv2.drawContours(contour_img, contours, -1, (0,255,0), 2) 
print("Contours Detected:") cv2_imshow(contour_img) 

EXPERIMENT 6
!pip install opencv-python numpy matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files

uploaded = files.upload()
filename = list(uploaded.keys())[0]
img = cv2.imread(filename)
img = cv2.resize(img, (500, 500))
cv2_imshow(img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 500:  # remove noise
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Square"
        elif len(approx) > 6:
            shape = "Circle"
        else:
            shape = "Unknown"

        cv2.drawContours(img, [approx], 0, (0,255,0), 3)
        cv2.putText(img, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Detected Shapes")
plt.axis('off')
plt.show()
