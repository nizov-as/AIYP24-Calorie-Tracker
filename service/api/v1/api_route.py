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
from pathlib import Path

router = APIRouter()

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


# Дообучение модели
@router.post("/fit", response_model=FitResponse)
async def fit(request: FitRequest):
    model_id = request.model_id

    if model_id == 'detect':
        if 'detect' not in loaded_models:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="YOLO model not loaded.")

        yolov_model = loaded_models['detect']
        data_path = os.path.join(os.getcwd(), 'fit/data.yaml')

        yolov_model.train(data=data_path, epochs=1,
                          imgsz=480,
                          batch=32,
                          lr0=0.0015,
                          device="cpu",
                          name='yolov11s_client')

        # Сохранение дообученной модели
        yolov_model.save(os.path.join(os.getcwd(), 'models/custom.pt'))
        loaded_models['custom'] = yolov_model  # Сохраняем в loaded_models

        return FitResponse(message="YOLO model trained and saved as 'custom' successfully.")

    else:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid model_id provided.")


# Загрузка моделей
@router.post("/load", response_model=List[LoadResponse])
async def load(model_id: str):
    try:
        model_directory = os.path.join(os.getcwd(), 'models')

        if model_id == 'detect':
            yolov_model_path = os.path.join(model_directory, 'custom_yolov11s_e100.pt')
            yolov_model = YOLO(yolov_model_path)
            loaded_models['detect'] = yolov_model
            return [LoadResponse(message="YOLO model loaded successfully")]

        elif model_id == 'classific':
            classification_model_path = os.path.join(model_directory, 'custom_model.h5')
            classification_model = load_model(classification_model_path)
            loaded_models['classific'] = classification_model
            return [LoadResponse(message="Classification model loaded successfully")]

        elif model_id == 'custom':
            # Загрузка кастомной модели
            for key, model in loaded_models.items():
                if key == 'custom':
                    return [LoadResponse(message="Custom model loaded successfully")]

            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Custom model not found.")

        else:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST,
                                detail="Invalid model_id provided. Use 'detect', 'classific' or 'custom'.")

    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/predict", response_model=PredictionResponse)
async def predict(model_id: str, images: List[UploadFile] = File(...)):
    if model_id not in loaded_models:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Model not found")

    model = loaded_models[model_id]
    predictions = []

    for uploaded_file in images:
        # Чтение файла
        image_bytes = await uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
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
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

                    # Добавление текста
                    label = f"{class_name} ({conf:.2f})"
                    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Конвертируем изображение в Base64
            _, buffer = cv2.imencode(".jpg", image)
            image_base64 = base64.b64encode(buffer).decode("utf-8")

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
            predictions.append({
                'class': class_name,  # Получаем предсказанный класс
                'confidence': confidence  # Уверенность предсказания
            })

    return PredictionResponse(predictions=predictions)


@router.delete("/remove/{model_id}", response_model=RemoveResponse)
async def remove(model_id: str):
    if model_id not in loaded_models:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Model not found")

    del loaded_models[model_id]  # Удаляем модель из loaded_models
    return RemoveResponse(message=f"Model '{model_id}' removed successfully.")


@router.delete("/remove_all", response_model=RemoveResponse)
async def remove_all():
    if not loaded_models:
        return RemoveResponse(message="No models to remove")

    removed_models = list(loaded_models.keys())
    loaded_models.clear()  # Очищаем словарь загруженных моделей
    return RemoveResponse(message=", ".join(f"Model '{model_id}' removed" for model_id in removed_models))


