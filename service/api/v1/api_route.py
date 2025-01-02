from fastapi import APIRouter, HTTPException, UploadFile, File
from http import HTTPStatus
from pydantic import BaseModel
from typing import Union, Dict, List
import numpy as np
from keras.models import load_model
import cv2
import os
from ultralytics import YOLO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from pathlib import Path
from service.logger_config import setup_logger
import pandas as pd

router = APIRouter()

logger = setup_logger()

loaded_models = {}


class ApiResponse(BaseModel):
    message: str
    data: Union[Dict, None] = None


class ModelListResponse(BaseModel):
    models: List[Dict[str, Union[str, str]]]


class PredictRequest(BaseModel):
    model_id: str
    images: List[UploadFile]


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Union[str, float]]]


class LoadResponse(BaseModel):
    message: str


class RemoveResponse(BaseModel):
    message: str


class FitRequest(BaseModel):
    model_id: str  # Добавляем model_id


class FitResponse(BaseModel):
    message: str


@router.post("/eda", response_model=Dict[str, List[str]])
async def eda():
    categories = 'category.txt'
    bbox_files = 'bb_info.txt'
    imgs_format = 'jpg'

    category_ids = []
    category_names = []

    path = Path(__file__).parent.parent / 'UECFOOD256'

    # Загружаем категории
    with open(os.path.join(path, categories), 'r') as list_:
        for i, line in enumerate(list_):
            if i > 0:  # skip header
                line = line.rstrip('\n').split('\t')
                category_ids.append(int(line[0]))
                category_names.append(line[1])

    categories_images = []
    categories_bbox_info = []

    for id_pos, id in enumerate(category_ids):
        categories_images.append([])
        categories_bbox_info.append([])

        # Читаем файлы
        imgs_file_list = os.path.join(path, str(id), bbox_files)
        with open(imgs_file_list, 'r') as list_:
            for i, line in enumerate(list_):
                if i > 0:  # skip header
                    line = line.rstrip('\n').split(' ')
                    categories_images[id_pos].append(line[0])
                    line = list(map(float, line[1:]))
                    categories_bbox_info[id_pos].append(line)

    # График 1: Топ-10 категорий с наибольшим количеством изображений
    category_count = {
        category_names[i]: len(categories_images[i])
        for i in range(len(category_names))
    }
    sorted_categories = sorted(
        category_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    x = [x[0] for x in sorted_categories]
    y = [x[1] for x in sorted_categories]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=x, y=y, palette="viridis")
    plt.xticks(rotation=45)
    plt.title('Топ-10 категорий с наибольшим количеством изображений')

    # Сохраняем график в буфер
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64_1 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # График 2: Плотность расположения объектов
    centers_x = [
        (bbox[0] + bbox[2]) / 2
        for bbox_info in categories_bbox_info
        for bbox in bbox_info
    ]
    centers_y = [
        (bbox[1] + bbox[3]) / 2
        for bbox_info in categories_bbox_info
        for bbox in bbox_info
    ]

    plt.figure(figsize=(8, 8))
    sns.kdeplot(x=centers_x, y=centers_y, cmap="Reds", fill=True)
    plt.xlabel("Центр X")
    plt.ylabel("Центр Y")
    plt.title("Плотность расположения объектов на изображениях")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64_2 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # График 3: Распределение площадей объектов
    areas = [
        (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        for bbox_info in categories_bbox_info
        for bbox in bbox_info
    ]

    plt.figure(figsize=(8, 6))
    plt.hist(areas, bins=50)
    plt.xlabel("Площадь bounding box")
    plt.ylabel("Количество объектов")
    plt.title("Распределение площадей объектов в bounding box")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64_3 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    logger.info("EDA успешно завершен, графики отрисованы.")
    return {
        "images": [
            image_base64_1,
            image_base64_2,
            image_base64_3
        ]
    }


# Дообучение модели
@router.post("/fit", response_model=FitResponse)
async def fit(request: FitRequest):
    model_id = request.model_id

    if model_id == 'detect':
        if 'detect' not in loaded_models:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="YOLO model not loaded."
            )

        yolov_model = loaded_models['detect']
        # data_path = os.path.join(os.getcwd(), 'fit/data.yaml')
        data_path = Path(__file__).parent.parent / "fit/data.yaml"

        yolov_model.train(data=data_path, epochs=1,
                          imgsz=480,
                          batch=32,
                          lr0=0.0015,
                          device="cpu",
                          name='yolov11s_client')

        # Сохранение дообученной модели
        yolov_model.save(Path(__file__).parent.parent / 'models/custom.pt')
        loaded_models['custom'] = yolov_model  # Сохраняем в loaded_models

        logger.info("Дообучение модели YOLO успешно завершено.")
        return FitResponse(
            message="YOLO model trained and saved as 'custom' successfully."
        )

    else:
        logger.error("Для дообучения была выбрана ннекорректная модель.")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Invalid model_id provided."
        )
    
    
@router.get("/fit/results", response_model=ApiResponse)
async def get_training_results():
    results_dir = Path("runs/detect/yolov11s_client")
    results_csv_path = results_dir / "results.csv"
    results_image_path = results_dir / "val_batch0_labels.jpg"

    if not results_dir.exists():
        logger.error("Директория с результатами обучения не найдена.")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Results directory not found."
        )

    # Проверяем наличие файла CSV
    if not results_csv_path.exists():
        logger.error("CSV файл с результатами обучения отсутствует.")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Results CSV file not found."
        )

    # Загружаем данные из results.csv
    results_df = pd.read_csv(results_csv_path)
    results_table_html = results_df.to_html(index=False)

    # Проверяем наличие файла изображения
    if not results_image_path.exists():
        logger.error("Файл изображение с результатами обучения отсутствует.")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Results image file not found."
        )

    # Кодируем изображение в base64
    with open(results_image_path, "rb") as image_file:
        image_bytes = image_file.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    logger.info("Результаты обучения возвращены.")
    return ApiResponse(
        message="Training results retrieved successfully.",
        data={
            "results_table": results_table_html,
            "val_batch0_labels_image": f"data:image/jpeg;base64,{encoded_image}"
        }
    )


# Загрузка моделей
@router.post("/load", response_model=List[LoadResponse])
async def load(model_id: str):
    try:
        model_directory = Path(__file__).parent.parent / "models"

        if model_id == 'detect':
            yolov_model_path = os.path.join(
                model_directory,
                'custom_yolov11s_e100.pt'
            )
            yolov_model = YOLO(yolov_model_path)
            loaded_models['detect'] = yolov_model
            logger.info(f"Модель YOLO успешно загружена.")
            return [
                LoadResponse(message="YOLO model loaded successfully")
            ]

        elif model_id == 'classific':
            classification_model_path = os.path.join(
                model_directory,
                'best_model_101class.keras'
            )
            classification_model = load_model(classification_model_path)
            loaded_models['classific'] = classification_model
            logger.info(f"Модель классификации успешно загружена.")
            return [
                LoadResponse(
                    message="Classification model loaded successfully"
                )
            ]

        elif model_id == 'custom':
            # Загрузка кастомной модели
            for key, model in loaded_models.items():
                if key == 'custom':
                    logger.info(f"Дообученная YOLO успешно загружена.")
                    return [
                        LoadResponse(
                            message="Custom model loaded successfully"
                        )
                    ]
            logger.error(f"Дообученная YOLO не найдена")
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail="Custom model not found."
            )

        else:
            logger.info(f"Некорректное значение model_id - {model_id}")
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Use 'detect', 'classific' or 'custom' model_id."
            )

    except Exception as e:
        logger.error("Ошибка при загрузке модели")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict(model_id: str, images: List[UploadFile] = File(...)):
    if model_id not in loaded_models:
        logger.error(f"Модель {model_id} не загружена")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Model not found"
        )

    model = loaded_models[model_id]
    predictions = []

    for uploaded_file in images:
        # Чтение файла
        image_bytes = await uploaded_file.read()
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        if image is None:
            logger.error(f"Ошибка при декодировании входного изображения")
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Could not decode image: {uploaded_file.filename}",
            )

        if model_id == "detect":
            results = model(image)  # Получаем предсказания

            for result in results:
                for *box, conf, cls in result.boxes.data.tolist():
                    # Распаковываем координаты бокса
                    x_min, y_min, x_max, y_max = map(int, box)
                    class_name = model.names[int(cls)]

                    # Добавляем предсказание в список
                    predictions.append({
                        "class": class_name,
                        "confidence": conf,
                    })

                    # Отрисовка bounding box
                    color = (0, 255, 0)  # Зеленый цвет
                    cv2.rectangle(
                        image,
                        (x_min, y_min),
                        (x_max, y_max),
                        color,
                        2
                    )

                    # Добавление текста
                    label = f"{class_name} ({conf:.2f})"
                    cv2.putText(
                        image,
                        label,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

            # Конвертируем изображение в Base64
            _, buffer = cv2.imencode(".jpg", image)
            image_base64 = base64.b64encode(buffer).decode("utf-8")

            logger.info("Модель {model_id} сделала предсказание")
            predictions.append({
                "image": image_base64,  # Возвращаем изображение в Base64
            })

        elif model_id == 'classific':
            # Подготовка изображения для классификации
            img = cv2.resize(image, (200, 200))  # Изменяем размер изображения
            img = np.expand_dims(img, axis=0)
            img = img / 255.0  # Нормализация изображения

            # Предсказание класса
            pred = model.predict(img)
            index = np.argmax(pred)
            class_name = str(index)
            confidence = pred[0][index]  # Уверенность предсказания
            logger.info("Модель {model_id} сделала предсказание")
            predictions.append({
                'class': class_name,  # Получаем предсказанный класс
                'confidence': confidence  # Уверенность предсказания
            })

        elif model_id == "custom":
            results = model(image)  # Получаем предсказания

            for result in results:
                for *box, conf, cls in result.boxes.data.tolist():
                    # Распаковываем координаты бокса
                    x_min, y_min, x_max, y_max = map(int, box)
                    class_name = model.names[int(cls)]

                    # Добавляем предсказание в список
                    predictions.append({
                        "class": class_name,
                        "confidence": conf,
                    })

                    # Отрисовка bounding box
                    color = (0, 255, 0)  # Зеленый цвет
                    cv2.rectangle(
                        image,
                        (x_min, y_min),
                        (x_max, y_max),
                        color,
                        2
                    )

                    # Добавление текста
                    label = f"{class_name} ({conf:.2f})"
                    cv2.putText(
                        image,
                        label,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

            # Конвертируем изображение в Base64
            _, buffer = cv2.imencode(".jpg", image)
            image_base64 = base64.b64encode(buffer).decode("utf-8")
            logger.info("Модель {model_id} сделала предсказание")
            predictions.append({
                "image": image_base64,  # Возвращаем изображение в Base64
            })
    return PredictionResponse(predictions=predictions)

# Ручка для получения списка загруженных моделей
@router.get("/loaded_models", response_model=ModelListResponse)
async def get_loaded_models():
    if not loaded_models:
        logger.error("Загруженные модели отсутствуют")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="The list of loaded models is empty."
        )
    logger.info("Возвращен список загруженных моделей.")
    models = [
        {"model_id": model_id, "status": "loaded"}
        for model_id in loaded_models.keys()
    ]
    return ModelListResponse(models=models)

@router.delete("/remove/{model_id}", response_model=RemoveResponse)
async def remove(model_id: str):
    if model_id not in loaded_models:
        logger.error(f"Модель {model_id} не загружена")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Model not found"
        )

    del loaded_models[model_id]  # Удаляем модель из loaded_models
    logger.info(f"Модель {model_id} удалена из загруженных.")
    return RemoveResponse(message=f"Model '{model_id}' removed successfully.")


@router.delete("/remove_all", response_model=RemoveResponse)
async def remove_all():
    if not loaded_models:
        logger.info(f"Нет загруженных моделей.")
        return RemoveResponse(message="No models to remove")

    removed_models = list(loaded_models.keys())
    loaded_models.clear()  # Очищаем словарь загруженных моделей
    logger.info(f"Загруженные модели удалены.")
    return RemoveResponse(message=", ".join(
        f"Model '{model_id}' removed"
        for model_id in removed_models
    ))
