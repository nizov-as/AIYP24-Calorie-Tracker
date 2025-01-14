import streamlit as st
import requests
from PIL import Image
import io
import base64
import json
from streamlit_lottie import st_lottie
import time  # Для имитации прогресса
import logging
from logging.handlers import RotatingFileHandler
import os

BASE_URL = "http://fastapi:8000/api/v1/models"

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Настройка логирования
LOG_FILE = os.path.join(log_dir, "app.log")
logger = logging.getLogger("CalorieTracker")
logger.setLevel(logging.INFO)

# Проверяем, есть ли уже обработчики, чтобы избежать дублирования
if not logger.handlers:
    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Стилизация
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

# Заголовок страницы
title = "🍽️ Calorie-Tracker 🍽️"
st.markdown(f'<p class="title">{title}</p>', unsafe_allow_html=True)


# Функция загрузки анимации
def load_lottie_from_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_animation = load_lottie_from_file("Animation.json")
st_lottie(lottie_animation, height=200)

# Боковая панель
st.sidebar.markdown(
    '<h2 class="subheader">Дополнительный функционал:</h2>',
    unsafe_allow_html=True
)

# Флаг завершения анализа EDA
if "eda_completed" not in st.session_state:
    st.session_state["eda_completed"] = False

# Кнопка для запуска EDA
eda_placeholder = st.sidebar.empty()
if st.sidebar.button("📊 Провести EDA датасета UECFOOD256"):
    with st.spinner("Выполняется анализ данных..."):
        for i in range(100):
            time.sleep(0.03)  # Имитируем выполнение для красоты))))
            eda_placeholder.progress(i + 1)
        try:
            # Запрос на сервер для выполнения EDA
            response = requests.post(f"{BASE_URL}/eda")
            if response.status_code == 200:
                images = response.json().get("images", [])
                if images:
                    st.session_state["eda_images"] = images
                    st.session_state["eda_completed"] = True
                    st.sidebar.success("✅ Анализ данных завершён!")
                    logger.info("EDA успешно завершен, графики получены.")
                else:
                    st.sidebar.warning("⚠️ Сервис не вернул графики.")
                    logger.warning("EDA выполнен, но графики отсутствуют.")
            else:
                error_detail = response.json().get("detail", response.text)
                st.sidebar.error(f"❌ Ошибка: {error_detail}")
                logger.error(f"Ошибка выполнения EDA: {error_detail}")
        except Exception as e:
            st.sidebar.error(f"⚠️ Произошла ошибка: {str(e)}")
            logger.exception("Ошибка при выполнении EDA.")

# Кнопка для дообучения
if st.sidebar.button("⚙️ Дообучить модель YOLO на двух картинках ))"):
    with st.spinner("Дообучение модели, пожалуйста подождите..."):
        try:
            response = requests.post(
                f"{BASE_URL}/fit", json={"model_id": 'detect'}
            )
            if response.status_code == 200:
                message = response.json().get(
                    "message",
                    "Модель успешно дообучена!"
                )
                st.sidebar.success(f"✅ {message}")
                logger.info("Дообучение модели YOLO успешно завершено.")
            else:
                error_detail = response.json().get("detail", response.text)
                st.sidebar.error(f"❌ Ошибка: {error_detail}")
                logger.error(f"Ошибка дообучения модели: {error_detail}")
        except Exception as e:
            st.sidebar.error(
                f"⚠️ Произошла ошибка при дообучении модели: {str(e)}"
            )
            logger.exception("Ошибка при дообучении модели YOLO.")

# Ручка для получения списка загруженных моделей
MODELS_URL = f"{BASE_URL}/loaded_models"
# Инициализация состояния для отображения списка
if "show_models" not in st.session_state:
    st.session_state["show_models"] = False  # Изначально скрыто
if "models_list" not in st.session_state:
    st.session_state["models_list"] = []  # Пустой список моделей
# Кнопка для отображения/скрытия списка загруженных моделей
if st.sidebar.button("📂 Показать/Скрыть загруженные модели"):
    # Переключаем состояние отображения
    st.session_state["show_models"] = not st.session_state["show_models"]
    
    # Если показываем список, отправляем запрос
    if st.session_state["show_models"]:
        try:
            response = requests.get(MODELS_URL)
            if response.status_code == 200:
                models = response.json().get("models", [])
                st.session_state["models_list"] = models  # Сохраняем список в состоянии
                logger.info("Список загруженных моделей получен.")
            else:
                st.session_state["models_list"] = []
                error_detail = response.json().get("detail", response.text)
                st.sidebar.error(f"❌ Ошибка: {error_detail}")
                logger.error(f"Ошибка при получении списка моделей: {error_detail}")
        except Exception as e:
            st.session_state["models_list"] = []
            st.sidebar.error(f"⚠️ Произошла ошибка: {str(e)}")
            logger.exception("Ошибка при запросе списка моделей.")
# Отображение списка моделей
if st.session_state["show_models"]:
    if st.session_state["models_list"]:
        st.sidebar.markdown("### Загруженные модели:")
        for model in st.session_state["models_list"]:
            st.sidebar.write(f"- ID: {model['model_id']} | Статус: {model['status']}")
    else:
        st.sidebar.warning("⚠️ Список загруженных моделей пуст.")
# Инициализация состояния для управления отображением результатов дообучения
if "show_fit_results" not in st.session_state:
    st.session_state["show_fit_results"] = False
# Кнопка для показа/скрытия результатов дообучения
if st.sidebar.button("📊 Посмотреть/Скрыть результаты дообучения"):
    st.session_state["show_fit_results"] = not st.session_state["show_fit_results"]
# Если результаты нужно показать
if st.session_state["show_fit_results"]:
    with st.spinner("Получение результатов дообучения..."):
        try:
            # Запрос к серверу
            response = requests.get(f"{BASE_URL}/fit/results")
            if response.status_code == 200:
                # Извлечение данных из ответа
                results_data = response.json().get("data", {})
                results_table = results_data.get("results_table", "")
                results_image = results_data.get("val_batch0_labels_image", "")
                # Отображение таблицы результатов
                if results_table:
                    st.write("### Результаты дообучения модели:")
                    st.write(results_table, unsafe_allow_html=True)
                else:
                    st.warning("Таблица результатов отсутствует.")
                # Отображение изображения результатов
                if results_image:
                    st.write("### Визуализация (val_batch0_labels.jpg):")
                    image = Image.open(io.BytesIO(base64.b64decode(results_image.split(",")[1])))
                    st.image(image, caption="val_batch0_labels.jpg")
                else:
                    st.warning("Изображение результатов отсутствует.")
                logger.info("Результаты дообучения успешно получены.")
            else:
                error_detail = response.json().get("detail", response.text)
                st.sidebar.error(f"❌ Ошибка: {error_detail}")
                logger.error(f"Ошибка получения результатов дообучения: {error_detail}")
        except Exception as e:
            st.sidebar.error(f"⚠️ Ошибка при получении результатов: {str(e)}")
            logger.exception("Ошибка получения результатов дообучения.")

# Кнопка для проверки состояния сервиса
if st.sidebar.button("🛠️ Проверить состояние сервиса"):
    try:
        response = requests.get("http://fastapi:8000/")
        if response.status_code == 200:
            st.sidebar.success(
                f"✅ Сервис работает: {response.json().get('status', 'OK')}"
            )
            logger.info("Сервис доступен и работает.")
        else:
            st.sidebar.error(
                f"❌ Сервис недоступен. Код ответа: {response.status_code}"
            )
            logger.error(
                f"Проблема с сервисом: код ответа {response.status_code}"
            )
    except Exception as e:
        st.sidebar.error(f"⚠️ Ошибка при проверке состояния: {str(e)}")
        logger.exception("Ошибка проверки состояния сервиса.")

# Основной экран
# Выбор и загрузка модели
st.markdown('<h2 class="subheader">Выбор модели</h2>', unsafe_allow_html=True)
models = {
    "detect": "Модель для обнаружения и классификации объектов (YOLO).",
    "classific": "Модель для классификации изображений (InceptionV3).",
    "custom": "Пользовательская дообученная модель."
}

selected_model = st.selectbox(
    "Выберите модель для загрузки:",
    options=list(models.keys()),
    format_func=lambda x: f"{x.capitalize()} - {models[x]}"
)

if st.button("Загрузить модель"):
    with st.spinner(
        f"Загрузка модели {selected_model}, пожалуйста подождите..."
    ):
        try:
            response = requests.post(
                f"{BASE_URL}/load", params={"model_id": selected_model}
            )
            if response.status_code == 200:
                st.success(f"✅ Модель '{selected_model}' успешно загружена!")
                logger.info(f"Модель {selected_model} успешно загружена.")
            else:
                temp_str = response.json().get(
                    'detail',
                    'Неизвестная ошибка'
                )
                st.error(
                    f"❌ Ошибка: {temp_str}"
                )
                logger.error(
                    f"Ошибка загрузки модели {selected_model}: {response.text}"
                )
        except Exception as e:
            st.error(f"⚠️ Произошла ошибка при загрузке модели: {str(e)}")
            logger.exception(f"Ошибка при загрузке модели {selected_model}.")

# Загрузка изображения для предсказания
st.markdown(
    '<h2 class="subheader">Загрузка изображения для предсказания</h2>',
    unsafe_allow_html=True
)
uploaded_files = st.file_uploader(
    "Выберите изображение/изображения",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files and st.button("🔍 Сделать предсказание"):
    files = [
        ("images", (file.name, file.getvalue(), file.type))
        for file in uploaded_files
    ]
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            params={"model_id": selected_model},
            files=files
        )
        if response.status_code == 200:
            predictions = response.json().get("predictions", [])
            if not predictions:
                st.warning("⚠️ Сервис не вернул предсказаний.")
                logger.warning("Предсказания отсутствуют.")
            else:
                st.markdown(
                    '<h2 class="subheader">Результаты предсказания:</h2>',
                    unsafe_allow_html=True
                )
                for pred in predictions:
                    if "class" in pred and "confidence" in pred:
                        st.write(f"**Класс:** {pred['class']}")
                        st.write(
                            f"**Уверенность:** {pred['confidence'] * 100:.2f}%"
                        )
                    if "image" in pred:
                        image_data = pred["image"]
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        st.image(
                            image,
                            caption="Результат с боксами",
                            use_column_width=True
                        )
                logger.info("Предсказания успешно выполнены и отображены.")
        else:
            temp_str = response.json().get(
                'detail',
                'Неизвестная ошибка'
            )
            st.error(
                f"❌ Ошибка: {temp_str}"
            )
            logger.error(f"Ошибка предсказания: {response.text}")
    except Exception as e:
        st.error(f"⚠️ Произошла ошибка: {str(e)}")
        logger.exception("Ошибка выполнения предсказания.")

# Показать/скрыть графики EDA, если анализ завершен
if st.session_state["eda_completed"]:
    show_graphs = st.checkbox("Показать/Скрыть графики EDA", value=True)

    if show_graphs:
        for idx, img_base64 in enumerate(
            st.session_state.get("eda_images", [])
        ):
            image_bytes = base64.b64decode(img_base64)
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption=f"График {idx + 1}", use_column_width=True)
        logger.info("Графики EDA успешно отображены.")
