# Запуск

1. Склонировать репозиторий, переключиться на ветку "Obedkov_Igor_docker_v1"

2. По [ссылке](https://drive.google.com/file/d/1N4Qy6LwOzENtHG8QdCQbBKlpSDVbCjZL/view?usp=sharing) загрузить веса классификатора и положить их в папку ./service/api/models/ (данная папка имеется в ветке Obedkov_Igor_docker_v1)

3. Для запуска функции EDA необходимо скачать [датасет](http://foodcam.mobi/dataset256.zip) и поместить папку UECFOOD256 в папку ./service/data/ (данная папка имеется в ветке Obedkov_Igor_docker_v1)

4. Открыть консоль в корне проекта и последовательно выполнить следующие команды:

* docker compose build (для первого запуска)
* docker compose up

5. Перейти по адресу - http://localhost:8501
