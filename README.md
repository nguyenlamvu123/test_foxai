---
title: Test Foxai
emoji: 🦀
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
---
tesseract is required  
### on linux:  
- sudo apt-get update  
- sudo apt install tesseract-ocr-vie  
- sudo apt-get install libleptonica-dev tesseract-ocr tesseract-ocr-dev libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn  
https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i

Truy cập HuggingFace: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main
- Tải file mistral-7b-instruct-v0.1.Q4_K_M.gguf
- Đặt file vào thư mục: models/mistral/mistral-7b-instruct.Q4_K_M.gguf


## Luồng Hoạt Động Tổng Quan  
### Người dùng → Giao diện Gradio (`python3 grad.py`) → Truy xuất chunk liên quan:
1. trích xuất từng ô của dữ liệu dạng bảng, để người dùng quyết định ô nào lấy, ô nào không 
2. tách riêng dữ liệu dạng bảng và dữ liệu tự do
3. lưu json kết quả vào local đồng thời trả về UI
4. Xử lý JSON JSON thành vector và lưu vào cơ sở dữ liệu FAISS 
### → Sinh câu trả lời → Trả kết quả về UI  
### Người dùng → FastAPI (`python3 api_be.py`) → POST /upload → Truy xuất chunk liên quan:
1. trích xuất từng ô của dữ liệu dạng bảng, để người dùng quyết định ô nào lấy, ô nào không 
2. tách riêng dữ liệu dạng bảng và dữ liệu tự do
3. lưu json kết quả vào local 
4. Xử lý JSON JSON thành vector và lưu vào cơ sở dữ liệu FAISS
5. trả về json báo thành công 

#### quét tài liệu ảnh png, trả ra json dạng  
`{
"text in table": " Mãi mũ 3mm.  Phần mềm tươn  j tắc thông tín thế thao  " ,
"free text": "“THÒNG TÍN THÊM. DỰ ÁN Phần mềm trích xuất phim "
} `  

       


#### API (có file postman để test trong source)


phần cắt bảng đã chạy khá ổn với file `3.png` (có trong source), cần cải thiện kết quả đọc text; lưu file json được trả ra vào local
