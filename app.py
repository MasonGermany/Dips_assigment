import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Dog Cat And Couple AI", page_icon="🐶", layout="centered")

st.title("Dog Cat And Couple")
st.markdown("**AI Object Detection & Segmentation System**")

# ฟังก์ชันโหลดโมเดล (ใช้ @st.cache_resource เพื่อไม่ให้โหลดใหม่ทุกครั้งที่กดปุ่ม)
@st.cache_resource
def load_models():
    # นำไฟล์ best.pt ของทั้งสองโหมดมาไว้ในโฟลเดอร์เดียวกับโค้ด
    model_rect = YOLO("best_rect.pt") # แก้ชื่อไฟล์ให้ตรง
    model_seg = YOLO("best_seg.pt")   # แก้ชื่อไฟล์ให้ตรง
    return model_rect, model_seg

model_rect, model_seg = load_models()

# ปุ่มเลือกโหมด
mode = st.radio("เลือกโหมดการทำงาน:", ("Rectangle (Box)", "Segment (Polygon)"), horizontal=True)

# อัปโหลดรูปภาพ
uploaded_file = st.file_uploader("โปรดเลือกรูปภาพ (รองรับ JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปภาพต้นฉบับ", use_container_width=True)
    
    if st.button("อัปโหลดเพื่อวิเคราะห์", type="primary", use_container_width=True):
        with st.spinner("กำลังวิเคราะห์รูปภาพ..."):
            
            # ประมวลผลตามโหมดที่เลือก
            if mode == "Segment (Polygon)":
                results = model_seg.predict(image, conf=0.5)
                result = results[0]
                im_array = result.plot(boxes=False, masks=True)
            else:
                results = model_rect.predict(image, conf=0.5)
                result = results[0]
                im_array = result.plot(boxes=True, masks=False)
                
            # ดึงชื่อคลาสออกมา
            detected_classes = [result.names[int(box.cls[0])] for box in result.boxes]
            unique_classes = list(set(detected_classes))
            prediction_text = ", ".join(unique_classes) if unique_classes else "Not Found"
            
            # แปลงภาพกลับมาแสดงผล
            im_pil = Image.fromarray(im_array[..., ::-1])
            
            st.success(f"**ผลการตรวจจับ:** {prediction_text}")
            st.image(im_pil, caption="ผลลัพธ์จาก AI", use_container_width=True)