import streamlit as st
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import numpy as np 


st.set_page_config(page_title="Dog Cat And Couple AI", page_icon="🐶", layout="centered")

st.title("Dog Cat And Couple")
st.markdown("**AI Object Detection & Segmentation System**")



@st.cache_resource
def load_models():
    model_rect = YOLO("best_rect.pt")
    model_seg = YOLO("best_seg.pt")
    return model_rect, model_seg

model_rect, model_seg = load_models()



mode = st.radio("เลือกโหมดการทำงาน:", ("Rectangle (Box)", "Segment (Polygon)"), horizontal=True)
uploaded_file = st.file_uploader("โปรดเลือกรูปภาพ (รองรับ JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปภาพต้นฉบับ", use_container_width=True)
    
    if st.button("อัปโหลดเพื่อวิเคราะห์", type="primary", use_container_width=True):
        with st.spinner("กำลังวิเคราะห์รูปภาพ..."):
            
          

            if mode == "Segment (Polygon)":
                results = model_seg.predict(image, conf=0.5)
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
