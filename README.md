# Запуск

По [ссылке](https://drive.google.com/file/d/1N4Qy6LwOzENtHG8QdCQbBKlpSDVbCjZL/view?usp=sharing) загрузить веса классификатора и положить их в папку ./service/api/models/

Для запуска функции EDA необходимо скачать [датасет](http://foodcam.mobi/dataset256.zip) и поместить папку UECFOOD256 в папку ./service/data/ 

Открыть консоль в корне проекта и последовательно выполнить следующие команды:

1. docker compose build (для первого запуска)

2. docker compose up

Перейти по адресу - http://localhost:8501