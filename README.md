## Сборка Docker-образа и запуск контейнера

### 1. **Подготовка**

* Склонировать репозиторий, переключиться на ветку "Docker_build"
* По [ссылке](https://drive.google.com/file/d/1N4Qy6LwOzENtHG8QdCQbBKlpSDVbCjZL/view?usp=sharing) загрузить веса классификатора и положить их в папку ./service/api/models/ данной ветки
* Для запуска функции EDA необходимо скачать [датасет](http://foodcam.mobi/dataset256.zip) и поместить папку UECFOOD256 в папку ./service/data/ данной ветки

### 2. **Сборка Docker-образа и запуск контейнера**

* Проверить, что на вашем устройстве установлен Docker Desktop. Открыть Docker Desktop.
* Открыть консоль (терминал) в корне проекта и последовательно выполнить следующие команды:
```bash
docker login -u docker-username
docker compose build (для первого запуска)
docker compose up
```
* Перейти по адресу - http://localhost:8501
