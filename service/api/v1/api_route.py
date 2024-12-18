from fastapi import APIRouter, HTTPException, UploadFile, File
from http import HTTPStatus
from pydantic import BaseModel
from typing import Union, Dict, List
import numpy as np
import torch
import cv2
import os
from torchvision import models, transforms
from ultralytics import YOLO

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


# Загрузка моделей
@router.post("/load", response_model=List[LoadResponse])
async def load(model_id: str):
    try:
        # Определяем путь к папке models
        model_directory = os.path.join(os.getcwd(), 'models')

        if model_id == 'detect':
            # Загрузка YOLO модели
            yolov_model_path = os.path.join(model_directory, 'custom_yolov11s_e100.pt')
            yolov_model = YOLO(yolov_model_path)  # Загружаем модель с весами
            loaded_models['detect'] = yolov_model
            return [LoadResponse(message="YOLO model loaded successfully")]

        elif model_id == 'classifi':
            # Загрузка модели классификации

            return [LoadResponse(message="Classification model loaded successfully")]

        else:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid model_id provided. Use 'detect' or 'classifi'.")

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
        image = await uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Could not decode image: {uploaded_file.filename}")

        if model_id == 'detect':
            results = model(image)  # Получаем предсказания
            for result in results:
                for *box, conf, cls in result.boxes.data.tolist():  # Получаем предсказания
                    predictions.append({
                        'class': model.names[int(cls)],  # Получаем имя класса
                        'confidence': conf  # Уверенность предсказания
                    })
        # elif model_id == 'classifi':

            # predictions.append({
            #     'class': str(predicted.item()),  # Получаем предсказанный класс
            #     'confidence': confidence[predicted].item()  # Уверенность предсказания
            # })

    return PredictionResponse(predictions=predictions)


# @router.delete("/remove/{model_id}", response_model=RemoveResponse)
# async def remove(model_id: str):
#
#
# @router.delete("/remove_all", response_model=RemoveResponse)
# async def remove_all():


