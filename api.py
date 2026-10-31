import io
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_rect = YOLO("D:/mlab/dips_assignment/runs/detect/stage1_rect_r2/weights/best.pt") # โมเดลกรอบสี่เหลี่ยม
model_seg = YOLO("D:/mlab/dips_assignment/runs/segment/stage1_seg_r2/weights/best.pt") # โมเดลวาดเส้น Polygon

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), mode: str = Form("rectangle")): 
    
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # 2. เช็กโหมดและวาดรูป
    if mode == "segment":
        results = model_seg.predict(img, conf=0.5)
        result = results[0]
        # บังคับวาดเฉพาะ Mask (Polygon) ห้ามวาด Box
        im_array = result.plot(boxes=False, masks=True) 
    else:
        results = model_rect.predict(img, conf=0.5)
        result = results[0]
        # บังคับวาดเฉพาะ Box อย่างเดียว
        im_array = result.plot(boxes=True, masks=False) 
        
    # ดึงชื่อคลาสออกมาแสดงผล텍สต์
    detected_classes = [result.names[int(box.cls[0])] for box in result.boxes]
    unique_classes = list(set(detected_classes))
    prediction_text = ", ".join(unique_classes) if unique_classes else "Not Found"
    
    # แปลงรูปกลับไปให้หน้าเว็บ
    im_pil = Image.fromarray(im_array[..., ::-1])
    buffered = io.BytesIO()
    im_pil.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return JSONResponse({
        "prediction_text": prediction_text,
        "image_base64": f"data:image/jpeg;base64,{img_base64}"
    })