import cv2
import easyocr
import pytesseract
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image


def custom_config(oem, psm):
    return f'--oem {oem} --psm {psm} -l vie'

img = cv2.imread('crop_9.png')  # ('crop_temp.png')  #
img_rgb_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blur = cv2.convertScaleAbs(img_rgb_, alpha=1.5, beta=0)
# blur = cv2.GaussianBlur(img_rgb_, (3, 3), 0)
resized = cv2.resize(blur, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
# # resized = img_rgb_
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# resized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# print('pytesseract')
# for oem in range(4):
#     for psm in range(14):
#         config = custom_config(oem, psm)
#         try:
#             text = pytesseract.image_to_string(resized, config=config)
#             print(config)
#             print(text)
#             print('#############################################')
#         except:
#             pass


# reader = easyocr.Reader(['vi', 'en'], gpu=False)
# results = reader.readtext(resized)
#
# for box, text, confidence in results:
#     print(f"easyocr🔸 {text} (Độ tin cậy: {confidence:.2f})")


config = Cfg.load_config_from_name('vgg_transformer')  # ('vgg_seq2seq')  #
config['device'] = 'cpu'
config['cnn']['pretrained'] = False  # tránh tải lại mô hình nếu đã có sẵn

detector = Predictor(config)
resized = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img2readtext = Image.fromarray(thresh)
text = detector.predict(img2readtext)

print("vietocr📄 Kết quả OCR:", text)

# convertScaleAbs, resize2: crop_9.png, crop_30.png
# without prepro: crop_4.png
# threshold: crop_4.png