## Руководство к использованию
Данный проект включает Streamlit-приложение, которое находится в отдельной ветке репозитория. Это приложение можно запустить локально или в контейнере Docker. Ниже описаны инструкции по сборке Docker-образа и запуску контейнера, а также использование приложения.

Видео-инструкция по работе streamlit-сервиса расположена по [ссылке](https://drive.google.com/drive/folders/1s5in2uoodIR3TUFKN_xn9afbP1ZJV_8V?usp=share_link). 

---

## Инструкция по использованию системы

### Способ 1. Веб-сервер с использованием Uvicorn (данный способ относится к ветке YOLO_train_Streamlit_service)

1. **Клонируйте репозиторий и перейдите в ветку с приложением:**
   ```bash
   git clone <URL_репозитория>
   cd <папка_репозитория>
   git checkout YOLO_train_Streamlit_service
   ```

2. **Установите зависимости:**
   Убедитесь, что у вас установлен Python 3.9+. Установите версии библиотек
   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux/MacOS
   source venv/scripts/activate   # Windows
   pip install -r requirements.txt
   ```

3. Скачайте датасет по [ссылке](http://foodcam.mobi/dataset256.html) и перенесите папку датасета (с названием UECFOOD256) в папку service/api ветки YOLO_train_Streamlit_service. 

4. **Откройте 2 разные консоли (терминала)**

5. **Запустите веб-сервер с помощью Uvicorn** (в первой консоли, из корневой папки проекта)
   ```bash
   uvicorn service.main:app --reload
   ```

6. **Запустите приложение локально** (во второй консоли, из папки streamlit_app)
   ```bash
   streamlit run app.py
   ```

7. **Откройте приложение в браузере:**
   Приложение доступно по умолчанию по адресу [http://localhost:8501](http://localhost:8501).

---

### Способ 2. Сборка Docker-образа и запуск контейнера

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


