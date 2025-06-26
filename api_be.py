import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import shutil
import os

from coordinate import cv2, list_documents, findcoord

app = FastAPI()

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----------- MODELS ------------
class Question(BaseModel):
    query: str


# ----------- ENDPOINTS ----------

@app.post("/ask")
def ask_question(q: Question):
    try:
        # answer = answer_question(q.query)
        return {"answer": q.query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def get_documents():
    try:
        docs = list_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), th1=50, th2=90):
    try:
        img_ = os.path.join(UPLOAD_DIR, file.filename)

        with open(img_, "wb") as f:
            f.write(await file.read())

        img = cv2.imread(img_, flags=0)
        findcoord(img, int(th1), int(th2), api=True)
        return {"message": f"Tài liệu '{file.filename}' đã được xử lý thành công."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
