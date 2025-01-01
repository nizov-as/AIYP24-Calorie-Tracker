## Проект "Calorie Tracker" 
**Команда 50, тема №32 "Дневник калорий"**

**Куратор команды:** Артём Беляев - @karaoke_tutu, artbelyaev0

**Команда проекта:**
- Участник 1, Игорь Объедков - @Jgjffkbvcfjjffjjv, igorobed
- Участник 2, Денис Круглов - @Den991, KrugD
- Участник 3, Александра Ломакина - @sal0oom, sal0m
- Участник 4, Александр Низов - @nizov_as, nizov-as

**Описание проекта:**

Телеграмм бот с детекцией и классификацией еды на фото и ее калораж, используя внутреннюю базу калорий. Предусмотрена возможность с помощью сервиса считать статистику по внесенной пользователем информации, используя обученную LLM модель.

## Руководство к использованию
Данный проект включает Streamlit-приложение, которое находится в отдельной ветке репозитория. Это приложение можно запустить локально или в контейнере Docker. Ниже описаны инструкции по сборке Docker-образа и запуску контейнера, а также использование приложения.

---

## Инструкция по использованию системы

1. **Клонируйте репозиторий и перейдите в ветку с приложением:**
   ```bash
   git clone <URL_репозитория>
   cd <папка_репозитория>
   git checkout <название_ветки>
   ```

2. **Установите зависимости:**
   Убедитесь, что у вас установлен Python 3.9+ и виртуальная среда.
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **Запустите приложение локально:**
   ```bash
   streamlit run app.py
   ```

4. **Откройте приложение в браузере:**
   Приложение доступно по умолчанию по адресу [http://localhost:8501](http://localhost:8501).

---

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


