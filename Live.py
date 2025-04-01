import cv2
import dlib
import numpy as np
import time

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def crop_face(image, landmarks):
    # Ambil titik landmark yang mengelilingi wajah dan crop wajah
    jawline_points = landmarks[0:17]
    jawline = cv2.convexHull(jawline_points)
    mouth_points = landmarks[48:68]
    mouth = cv2.convexHull(mouth_points)

    # Menggambar poligon yang mengelilingi bagian wajah
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [jawline, mouth], -1, (255, 255, 255), -1)

    # Membuat gambar hasil dengan hanya bagian wajah yang terpilih
    cropped_face = cv2.bitwise_and(image, image, mask=mask)

    # Resize gambar menjadi seukuran wajah
    x, y, w, h = cv2.boundingRect(mask)
    cropped_face = cropped_face[y:y+h, x:x+w]  # Potong gambar sesuai bounding box

    return cropped_face

cap = cv2.VideoCapture(0)

start_time = None
capture_done = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    if not capture_done:
        try:
            # Deteksi landmark wajah
            landmarks = get_landmarks(frame)
            
            if start_time is None:
                start_time = time.time()

            elapsed_time = time.time() - start_time

            if elapsed_time >= 3 and not capture_done:
                cv2.imwrite("captured_face.jpg", frame)
                print("Image captured")
                capture_done = True
                start_time = None  # Reset start time
        except NoFaces:
            start_time = None  # Reset start time if no faces detected
            cv2.putText(frame, "No Faces Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Tampilkan frame dengan deteksi wajah
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if capture_done:
    # Baca gambar yang tercapture
    captured_image = cv2.imread("captured_face.jpg")

    # Ubah gambar menjadi mode warna HSV
    hsv_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2HSV)

    # Tentukan batas-batas warna untuk mencari warna yang dominan
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([179, 255, 255])

    # Masking gambar untuk mendapatkan warna yang sesuai dengan batas
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Hitung jumlah piksel yang sesuai dengan mask
    total_pixels = cv2.countNonZero(mask)

    # Hitung proporsi warna dominan dalam gambar
    dominant_percentage = (total_pixels / (captured_image.shape[0] * captured_image.shape[1])) * 100

    # Tampilkan nilai warna HSV jika warna dominan ditemukan
    if dominant_percentage > 3:  # Ambil threshold 5% untuk warna yang dianggap dominan
        # Cari nilai rata-rata warna HSV di dalam mask
        avg_hsv_color = cv2.mean(hsv_image, mask=mask)[:3]
        hue, saturation, value = avg_hsv_color  # Pisahkan nilai HSV

        # Ubah nilai Saturation dan Value menjadi persen
        saturation_percent = (saturation / 255) * 100
        value_percent = (value / 255) * 100

        print("Detected Hue:", int(hue))
        print("Detected Saturation:", int(saturation_percent), "%")
        print("Detected Value:", int(value_percent), "%")
    else:
        print("Warna dominan tidak ditemukan")

    cv2.imshow("Captured Image", captured_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
