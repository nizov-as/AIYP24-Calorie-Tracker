import streamlit as st
import requests
from PIL import Image
import io
import base64
import json
from streamlit_lottie import st_lottie


BASE_URL = "http://127.0.0.1:8000/api/v1/models"

st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #9dcf86;
            text-align: center;
            margin-top: 20px;
        }
        .subheader {
            font-size: 24px;
            color: #639149;
            font-weight: bold;
            text-align: center;
        }
        .sidebar .sidebar-content {
            background-color: #e1fcef;
        }
        .stButton>button {
            background-color: #255e48;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #5F9EA0;
        }
    </style>
""", unsafe_allow_html=True)

# Заголовки
title = "🍽️ Calorie-Tracker 🍽️"
subheader_model = "Выберите модель"
subheader_image = "Загрузите изображение для обработки"
subheader_prediction = "Результаты предсказания:"

# Заголовок страницы
st.markdown(f'<p class="title">{title}</p>', unsafe_allow_html=True)

def load_lottie_from_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_animation = load_lottie_from_file("Animation.json")
st_lottie(lottie_animation, height=200)

# Выбор модели
st.sidebar.markdown(f"<h2 class='subheader'>{subheader_model}</h2>", unsafe_allow_html=True)
model_id = st.sidebar.selectbox("Реализованные варианты:", ["detect", "classific"])

# Загрузка модели
if st.sidebar.button("🔄 Загрузить модель"):
    try:
        response = requests.post(f"{BASE_URL}/load", params={"model_id": model_id})
        if response.status_code == 200:
            st.sidebar.success("✅ Модель успешно загружена!")
        else:
            error_detail = response.json().get("detail", response.text)
            st.sidebar.error(f"❌ Ошибка: {error_detail}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Произошла ошибка при загрузке модели: {str(e)}")

# Заголовок для загрузки изображения
st.markdown(f'<p class="subheader">{subheader_image}</p>', unsafe_allow_html=True)

# Загрузка изображений
uploaded_files = st.file_uploader(
    "Выберите изображение/изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# Предсказание
if uploaded_files and st.button("🔍 Сделать предсказание"):
    files = [("images", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
    try:
        # Отправляем запрос на сервер
        response = requests.post(
            f"{BASE_URL}/predict",
            params={"model_id": model_id},
            files=files
        )
        
        if response.status_code == 200:
            # Получаем предсказания
            predictions = response.json().get("predictions", [])
            
            if not predictions:
                st.warning("⚠️ Сервис не вернул предсказаний.")
            else:
                # Отображение результатов
                st.header(subheader_prediction)
                for pred in predictions:
                    # Отображение класса и уверенности
                    if "class" in pred and "confidence" in pred:
                        st.write(f"**Класс:** {pred['class']}")
                        st.write(f"**Уверенность:** {pred['confidence'] * 100:.2f}%")
                        
                    # Отображение изображения с результатами
                    if "image" in pred:
                        image_data = pred["image"]  # Base64 строка
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        st.image(image, caption="Результат с боксами", use_column_width=True)
        else:
            error_detail = response.json().get("detail", response.text)
            st.error(f"❌ Ошибка: {error_detail}")
    except Exception as e:
        st.error(f"⚠️ Произошла ошибка: {str(e)}")

# Состояние сервиса
if st.sidebar.button("🛠️ Проверить состояние сервиса"):
    try:
        response = requests.get("http://127.0.0.1:8000/")
        if response.status_code == 200:
            st.sidebar.success(f"✅ Сервис работает: {response.json().get('status')}")
        else:
            st.sidebar.error(f"❌ Сервис недоступен. Код ответа: {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Ошибка при проверке состояния: {str(e)}")