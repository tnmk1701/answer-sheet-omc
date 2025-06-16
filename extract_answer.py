import cv2
import pytesseract
from ultralytics import YOLO
from imutils import contours
import numpy as np
import imutils
import re
import argparse

# Khởi tạo bộ phân tích đối số
parser = argparse.ArgumentParser(description="Xử lý một ảnh bằng các mô hình YOLO.")
parser.add_argument("image_path", type=str, help="Đường dẫn đến ảnh đầu vào.")
args = parser.parse_args()

# Tải các mô hình YOLO
model_aa = YOLO("./weights/best1.pt")
model_bb = YOLO("./weights/best2.pt")

# Đường dẫn đến ảnh đầu vào từ đối số
image_path = args.image_path
image_origin = cv2.imread(image_path)

# Dự đoán trên ảnh sử dụng mô hình YOLO đầu tiên
results_aa = model_aa.predict(image_path)[0]

# Định nghĩa ánh xạ từ chỉ số sang chữ cái
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

# Khởi tạo từ điển để giữ kết quả cuối cùng
final_results = {}

# Lặp qua mỗi hộp được phát hiện từ model_aa
for i, box in enumerate(results_aa.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    cropped_img = image_origin[y1:y2, x1:x2]
    h, w = cropped_img.shape[:2]
    new_w = 350
    new_h = int(h * (new_w / w))
    cropped_img = cv2.resize(cropped_img, (new_w, new_h))

    # Dự đoán trên ảnh cắt sử dụng mô hình YOLO thứ hai
    results_bb = model_bb.predict(cropped_img)[0]

    # Xử lý mỗi hộp được phát hiện từ model_bb
    for j, box_bb in enumerate(results_bb.boxes):
        x1_bb, y1_bb, x2_bb, y2_bb = map(int, box_bb.xyxy[0].cpu().numpy())
        cropped_img_bb = cropped_img[y1_bb:y2_bb, x1_bb:x2_bb]

        # Sử dụng pytesseract để đọc số từ ảnh cắt
        result_text = pytesseract.image_to_string(cropped_img_bb, config='--psm 6 digits')
        result_text = re.sub(r'\D', '', result_text)  # Loại bỏ ký tự không phải số
        if result_text.isdigit():
            result_text = int(result_text)  # Chuyển đổi sang số nguyên

            # Chuyển đổi ảnh cắt sang ảnh xám, làm mờ nhẹ, sau đó tìm các cạnh
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

            # Áp dụng phương pháp ngưỡng Otsu để phân ngưỡng ảnh
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # Tìm các đường viền trong ảnh ngưỡng, sau đó khởi tạo danh sách các đường viền tương ứng với các câu hỏi
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            questionCnts = []

            # Lặp qua các đường viền
            for c in cnts:
                # Tính toán hộp bao của đường viền, sau đó sử dụng hộp bao để suy ra tỷ lệ khung hình
                (x, y, w, h) = cv2.boundingRect(c)
                ar = w / float(h)
                # Để đánh dấu đường viền là một câu hỏi, khu vực phải đủ rộng, đủ cao và có tỷ lệ khung hình xấp xỉ bằng 1
                if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
                    questionCnts.append(c)

            # Sắp xếp các đường viền câu hỏi từ trái sang phải cho một hàng câu trả lời
            if questionCnts:
                questionCnts = contours.sort_contours(questionCnts, method="left-to-right")[0]

                # Sắp xếp các đường viền cho câu hỏi hiện tại từ trái sang phải, sau đó khởi tạo chỉ số của câu trả lời đã được tô
                cnts = contours.sort_contours(questionCnts)[0]
                bubbled = None

                # Lặp qua các đường viền đã sắp xếp
                for (k, c) in enumerate(cnts):
                    # Tạo một mặt nạ chỉ hiện vùng "bong bóng" hiện tại cho câu hỏi
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)

                    # Áp dụng mặt nạ vào ảnh ngưỡng, sau đó đếm số pixel không phải số 0 trong khu vực bong bóng
                    mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                    total = cv2.countNonZero(mask)

                    # Nếu tổng hiện tại có số pixel không phải số 0 lớn hơn, thì chúng ta đang xem xét câu trả lời hiện đang được tô
                    if bubbled is None or total > bubbled[0]:
                        bubbled = (total, k)

                # Ghi lại câu trả lời đã tô trong từ điển kết quả cuối cùng
                final_results[result_text] = INDEX_TO_LETTER[bubbled[1]]

# Sắp xếp final_results theo khóa tăng dần
sorted_results = dict(sorted(final_results.items()))

# Hiển thị kết quả cuối cùng đã sắp xếp
print(sorted_results)
