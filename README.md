tesseract is required  
### on linux:  
- sudo apt-get update  
- sudo apt install tesseract-ocr-vie  
- sudo apt-get install libleptonica-dev tesseract-ocr tesseract-ocr-dev libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn  
https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i

#### quét tài liệu ảnh png, trả ra json dạng  
`{
"text in table": " Mãi mũ 3mm.  Phần mềm tươn  j tắc thông tín thế thao  " ,
"free text": "“THÒNG TÍN THÊM. DỰ ÁN Phần mềm trích xuất phim "
} `  

    python3 grad.py   


#### API (có file postman để test trong source)  
    python3 api_be.py 


phần cắt bảng đã chạy khá ổn với file `3.png` (có trong source), cần cải thiện kết quả đọc text; lưu file json được trả ra vào local
