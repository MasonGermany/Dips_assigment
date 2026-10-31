import streamlit as st
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import numpy as np # เพิ่ม NumPy สำหรับจัดการสี

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Dog Cat And Couple AI", page_icon="🐶", layout="centered")

st.title("Dog Cat And Couple")
st.markdown("**AI Object Detection & Segmentation System**")

# กำหนดสีแบบคงที่ (BGR format) สำหรับแต่ละคลาส
# ลำดับต้องตรงกับ Class ID ของคุณ (สมมติว่า 0: Cat, 1: Dog, 2: Man, 3: Woman)
# ปรับสี BGR ตามต้องการ (Blue, Green, Red)
COLORS = {
    0: (255, 0, 0),    # Cat: สีน้ำเงิน (Blue)
    1: (0, 255, 255),  # Dog: สีเหลือง (Yellow)
    2: (0, 255, 0),    # Man: สีเขียว (Green)
    3: (255, 0, 255)   # Woman: สีม่วง (Magenta)
}

@st.cache_resource
def load_models():
    model_rect = YOLO("best_rect.pt")
    model_seg = YOLO("best_seg.pt")
    return model_rect, model_seg

model_rect, model_seg = load_models()

# # ปรับ Header ให้ตรงกับสีที่ตั้งค่าไว้
# st.info("""
# **สัญลักษณ์สีของ Class:**  
# 🟦 แมว (Cat) | 🟨 หมา (Dog) | 🟩 ผู้ชาย (Man) | 🟪 ผู้หญิง (Woman)  
# """)

mode = st.radio("เลือกโหมดการทำงาน:", ("Rectangle (Box)", "Segment (Polygon)"), horizontal=True)
uploaded_file = st.file_uploader("โปรดเลือกรูปภาพ (รองรับ JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปภาพต้นฉบับ", use_container_width=True)
    
    if st.button("อัปโหลดเพื่อวิเคราะห์", type="primary", use_container_width=True):
        with st.spinner("กำลังวิเคราะห์รูปภาพ..."):
            
            # สร้างลิสต์สี (BGR) ตามลำดับ Class ID ใน results เพื่อบังคับสี YOLO
            def get_colors(results):
                colors = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    # ถ้าเจอคลาสที่ไม่มีใน COLORS ให้ใช้สีเทา (128, 128, 128) เป็น default
                    colors.append(COLORS.get(cls_id, (128, 128, 128)))
                return colors

            if mode == "Segment (Polygon)":
                results = model_seg.predict(image, conf=0.5)
                # ดึงสีมาใช้ในฟังก์ชัน plot
                custom_colors = get_colors(results)
                # สั่งวาด mask โดยใช้สีที่กำหนด (YOLO อาจต้องการสีเป็นรูปแบบเฉพาะ ขึ้นกับเวอร์ชั่น)
                # ใน YOLOv8 การเปลี่ยนสีแบบเจาะจงใน plot() อาจทำได้ยาก 
                # แต่ลองส่ง custom_colors ไปก่อน ถ้าไม่ได้ อาจต้องวาดเองด้วย OpenCV
                im_array = results[0].plot(boxes=False, masks=True, conf=False) 
                
            else:
                results = model_rect.predict(image, conf=0.5)
                im_array = results[0].plot(boxes=True, masks=False, conf=False)
                
            detected_classes = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
            class_counts = Counter(detected_classes)
            
            if class_counts:
                count_text = ", ".join([f"{cls} ({count})" for cls, count in class_counts.items()])
                total_objects = sum(class_counts.values())
                prediction_summary = f"พบวัตถุทั้งหมด {total_objects} ชิ้น: {count_text}"
            else:
                prediction_summary = "ไม่พบวัตถุที่ตรงกับคลาสเป้าหมาย"
            
            im_pil = Image.fromarray(im_array[..., ::-1])
            
            st.success(f"**ผลการตรวจจับ:** {prediction_summary}")
            st.image(im_pil, caption="ผลลัพธ์จาก AI", use_container_width=True)
