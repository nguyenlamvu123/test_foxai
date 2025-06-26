import os, json, cv2, datetime  # , tabula
import pytesseract
import numpy as np

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA


debug = False
imgout = 'crop_'
extout = '.png'
current_dir = os.path.dirname(__file__)
tempimg = os.path.join(current_dir, f'{imgout}temp{extout}')
os.makedirs(os.path.join(current_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "models", "mistral"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "vector_store"), exist_ok=True)
os.makedirs(os.path.join(current_dir, "vector_store", "faiss_index"), exist_ok=True)


def downloadmodel():
    import requests

    # Link trực tiếp đến file .gguf trên Hugging Face
    url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

    output_path = os.path.join(current_dir, "models", "mistral", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    if os.path.exists(output_path):
        return

    response = requests.get(url, stream=True)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
downloadmodel()

def group_h_lines(h_lines, thin_thresh):
    new_h_lines = []
    while len(h_lines) > 0:
        thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]
        lines = [line for line in h_lines if thresh[1] -
                 thin_thresh <= line[0][1] <= thresh[1] + thin_thresh]
        h_lines = [line for line in h_lines if thresh[1] - thin_thresh >
                   line[0][1] or line[0][1] > thresh[1] + thin_thresh]
        x = []
        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
        x_min, x_max = min(x) - int(5 * thin_thresh), max(x) + int(5 * thin_thresh)
        new_h_lines.append([x_min, thresh[1], x_max, thresh[1]])
    return new_h_lines

def group_v_lines(v_lines, thin_thresh):
    new_v_lines = []
    while len(v_lines) > 0:
        thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]
        lines = [line for line in v_lines if thresh[0] -
                 thin_thresh <= line[0][0] <= thresh[0] + thin_thresh]
        v_lines = [line for line in v_lines if thresh[0] - thin_thresh >
                   line[0][0] or line[0][0] > thresh[0] + thin_thresh]
        y = []
        for line in lines:
            y.append(line[0][1])
            y.append(line[0][3])
        y_min, y_max = min(y) - int(4 * thin_thresh), max(y) + int(4 * thin_thresh)
        new_v_lines.append([thresh[0], y_min, thresh[0], y_max])
    return new_v_lines

def find_h(img_bin, kernel_len):
    hor_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_len, 1),  # (1, 1),  #
    )
    image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=9)
    horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3)

    h_lines = cv2.HoughLinesP(
        horizontal_lines,
        1,
        np.pi / 180,
        30,
        # minLineLength=40,
        maxLineGap=250
    )

    return group_h_lines(h_lines, kernel_len), image_horizontal

def find_v(img_bin, kernel_len):
    ver_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, kernel_len),  # (3, 1),  #
    )
    image_vertical = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_vertical, ver_kernel, iterations=3)

    v_lines = cv2.HoughLinesP(
        vertical_lines,
        1,
        np.pi / 180,
        30,
        # minLineLength=40,
        maxLineGap=250
    )

    return group_v_lines(v_lines, kernel_len), image_vertical

def findcoord_v0(img):  # dùng cv2_inRange() rồi căn cứ vào contour có 4 góc
    # height, width = img.shape

    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    edges = cv2.Canny(
        image=img_blur,
        threshold1=50,
        threshold2=90
    )
    cnts = cv2.findContours(image=edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    # cv2.imshow('Contours', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # sort_list_area = [(cv2.contourArea(cnt), cnt) for cnt in contours]
    # sort_list_area.sort(key=lambda x: x and x[0])
    # try:
    #     c = sort_list_area[-1][1]
    # except Exception as e:
    #     print('exception', e, 'in findcoord() function')
    #     return None, None

    four_point_contours = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            four_point_contours.append(approx)
            # # s1, s0, _, _ = cv2.boundingRect(contour)
            # cv2.circle(img_, (good_s1, good_s0), radius=0, color=(255, 0, 0), thickness=5)
    cv2.drawContours(
        img,
        contours, #  four_point_contours,  #
        -1,
        (0, 255, 0),
        2
    )
    cv2.imwrite(
        'nhap.jpg',
        img
    )

def findcoord(img, th1, th2, api=False):
    height, width = img.shape
    img2readtext = img.copy()
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    img_bin = cv2.Canny(
        image=img_blur,
        threshold1=th1,
        threshold2=th2
    )
    bi_i: np.ndarray = np.zeros_like(img_bin[:, :])

    kernel_len = img.shape[1] // 120
    new_horizontal_lines, image_horizontal = find_h(img_bin, kernel_len)
    # nhl = sorted(new_horizontal_lines, key=lambda x: x[2] - x[0])[: 10]  # TODO
    # for h in nhl:
    for h in new_horizontal_lines:
        cv2.line(bi_i, (h[0], h[1]), (h[2], h[3]), 255)  # type: ignore
    new_vertical_lines, image_vertical = find_v(img_bin, kernel_len)
    nvl = sorted(new_vertical_lines, key=lambda x: x[3] - x[1])[3: ]  # TODO
    for v in nvl:
    # for v in new_vertical_lines:
        cv2.line(bi_i, (v[0], v[1]), (v[2], v[3]), 255)  # type: ignore

    cnts = cv2.findContours(image=bi_i, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    filenames = list()
    lim = height * width / 3
    for i, cnt in enumerate(contours):
        curr = cv2.contourArea(cnt)
        if curr > lim:
            continue
        x, y, w, h = cv2.boundingRect(cnt)  # Lấy tọa độ khung chữ nhật quanh contour
        cropped = img[y:y + h, x:x + w]  # Cắt ảnh theo vùng đó
        img2readtext[y:y+h, x:x+w] = 0
        imgname = os.path.join(current_dir, f"{imgout}{i}{extout}")
        filenames.append(imgname)
        cv2.imwrite(imgname, cropped)
    cv2.imwrite(tempimg, img2readtext)
    if api:
        analyze_images(filenames)
    if debug:
        combined = cv2.add(image_horizontal, image_vertical)
        cv2.drawContours(
            img,
            contours, #  four_point_contours,  #
            -1,
            (0, 255, 0),
            1
        )
        print()

def list_image_filenames():
    return [
        os.path.join(current_dir, file) for file in os.listdir(current_dir)
        if file.endswith(extout) and file.startswith(imgout)
    ]

# Đọc ảnh từ tên file
def load_image(filename):
    img = cv2.imread(filename)
    img_rgb_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_rgb = cv2.GaussianBlur(img_rgb_, (3, 3), 0)
    img_rgb = cv2.convertScaleAbs(img_rgb_, alpha=1.5, beta=0)
    return img_rgb

# Hiển thị preview các ảnh đã chọn
def show_selected_images(filenames):  # TODO improve
    images = [load_image(f) for f in filenames]
    return images

def analyze_images(filenames):
    img2readtext = load_image(tempimg)
    freetext = pytesseract.image_to_string(img2readtext, lang='vie')
    textintable = ''
    for f in filenames:
        img = load_image(f)
        textintable = f"{pytesseract.image_to_string(img, lang='vie')}\n{textintable}"
    jso = [
        {
            'text in table': textintable,
            'free text': freetext,
        }
    ]
    text = f"Đã phân tích {len(filenames)} ảnh."

    now = datetime.datetime.now()
    jsofile = f'{now.strftime("%Y%m%d_%H%M%S")}.csv'
    with open(jsofile, "w", encoding="utf-8") as f:  # lưu dữ liệu đã trích xuất vào file trong local
        json.dump(jso, f, ensure_ascii=False, indent=4)
    # TODO xóa các file tạm
    ingest_json_to_faiss(jsofile)
    return text, jso

# def findtable():
#     # Đọc và trích xuất bảng từ PDF
#     pdf_file = "3.pdf"
#     tables = tabula.read_pdf(pdf_file, pages='all', multiple_tables=True)
#
#     # In ra tất cả các bảng trích xuất được
#     for i, table in enumerate(tables):
#         print(f"Table {i + 1}:")
#         print(table)
#
#     # Lưu bảng thành file CSV nếu muốn
#     tabula.convert_into(pdf_file, "output.csv", output_format="csv", pages='all')

def list_documents():
    return [c for c in os.listdir() if c.endswith('.csv')]

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    documents = []
    for item in data:
        if isinstance(item, dict):
            text = json.dumps(item, ensure_ascii=False)
            documents.append(text)
    return documents

def ingest_json_to_faiss(json_path, faiss_path=os.path.join(current_dir, "vector_store", "faiss_index")):
    texts = load_json(json_path)
    if len(texts) == 0:
        return
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents(texts)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(faiss_path)

def get_retriever(faiss_path=os.path.join(current_dir, "vector_store", "faiss_index")):
    """tải một vector database FAISS đã được lưu trước đó, và tạo ra một đối tượng retriever để thực hiện tìm kiếm theo độ tương đồng"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever


if __name__ == '__main__':
    img = cv2.imread(
        'test_foxai.jpg',  # '3.png',  #
        flags=0
    )
    findcoord(img, 0, 10)
    # findtable()
