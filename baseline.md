Цель построения бейзлайна — обучить модель, которая при подаче на вход модели изображения блюда с едой будет выдавать ответ в виде классификации блюда на фото, а также, при хорошем сценарии, делать детекцию/сегментацию блюда.

Для выбора методов были дополнительно сделаны анализы статей "Simultaneous Food Localization and Recognition" и "A Large-Scale Benchmark for Food Image Segmentation" (находятся в ветках проекта) для проработки предметной области и изучения основных методов.

Для выбора архитектуры бейзлайна остановились на обучении двух моделей — InceptionV3 и YOLO:

* InceptionV3 - глубокая свёрточная нейронная сеть, разработанная Google, которая используется для задач классификации изображений;
* YOLO - семейство глубоких нейронных сетей, разработанных для задач обнаружения объектов в реальном времени (создана Джозефом Редмоном и далее развивалась сообществом).

Модель YOLO была дообучена на датасете, описанном в файле EDA.md, модель InceptionV3 - на другом датасете с соотношением фото и количества блюд 1к1. 

В качестве метрики для оценки моделей использовались accuracy (доля верно предсказанных классов) и mAP50-95 (отражает, насколько хорошо модель определяет объекты различных размеров, принимая во внимание точность предсказанных ограничивающих рамок (bounding boxes) и их соответствие реальным данным).

В дальнейшем команда будет принимать решение о том, какую из двух моделей развивать и использовать для сервиса.

На текущий момент InceptionV3 - более простая модель, которая выполняет только классификацию, YOLO - более продвинутая, выполняющая помимо классификации еще и детекцию блюд на фото.
