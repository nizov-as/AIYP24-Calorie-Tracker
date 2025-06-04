import matplotlib.pyplot as plt
from io import BytesIO
import io
import random
import logging
from aiogram.types import BufferedInputFile
from aiogram import Router, F
from aiogram.types import Message, Update, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, ContentType, ReplyKeyboardRemove
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from states import UserProfile, FoodLogState, WaterLogState, ActivityLogState
from utils import calculate_water_goal, calculate_calorie_goal, get_weather, get_random_tasty_recipe, get_food_info_nutritionix
from googletrans import Translator
from yolo_detection import FoodDetector

logger = logging.getLogger(__name__) 

router = Router()

translator = Translator()

users = {}

# Словарь калорий для разных типов активности (за 30 минут)
WORKOUT_CALORIES = {
    "Бег": 300,
    "Плавание": 250,
    "Ходьба": 150,
    "Силовая тренировка": 200,
    "Кардио тренировка": 250,
    "Сноуборд": 250,
    "Лыжи": 300,
    "Коньки": 200,
    "Ролики": 200,
}

# Словарь с уровнями активности и их коэффициентами
ACTIVITY_LEVELS = {
    "сидячий образ жизни": 1.2,
    "1–3 тренировки в неделю": 1.375,
    "3–5 тренировок в неделю": 1.55,
    "6–7 тренировок в неделю": 1.725,
    "физическая работа": 1.9
}

# Загрузка весов модели
food_detector = FoodDetector("models/food_yolov11s.onnx") 

def create_profile_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Создать профиль")],
        ],
        resize_keyboard=True,
    )


def create_main_menu_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🍴 Добавить прием пищи"), KeyboardButton(text="💧 Добавить воду")],
            [KeyboardButton(text="🏋️ Добавить тренировку"), KeyboardButton(text="🍽️ Полезный рецепт")],
            [KeyboardButton(text="📋 Персональные рекомендации"), KeyboardButton(text="📊 Текущий прогресс")],
            [KeyboardButton(text="📈 Графики прогресса"), KeyboardButton(text="👤 Профиль")]
        ],
        resize_keyboard=True,
        input_field_placeholder="Выберите действие..."
    )


def get_user_profile(user_id: int):
    """Возвращает профиль пользователя или None, если профиль не заполнен."""
    return users.get(user_id)


async def ensure_profile(message: Message):
    """Проверяет, настроен ли профиль пользователя. Возвращает True, если да."""
    if not get_user_profile(message.from_user.id):
        await message.reply(
            "Пожалуйста, сначала настройте профиль с помощью команды: /set_profile.",
            reply_markup=create_profile_keyboard()
        )
        return False
    return True


# Функция перевода с русского на английский
async def translate_to_eng(text: str):
    translated = await translator.translate(text, src='ru', dest='en')
    return translated.text


# Функция перевода с английского на русский
async def translate_to_ru(text: str) -> str:
    translated = await translator.translate(text, src='en', dest='ru')
    return translated.text


@router.message(Command("start"))
async def cmd_start(message: Message):
    user = get_user_profile(message.from_user.id)
    name = message.from_user.first_name
    
    if user:
        try:
            # Попробуем отправить сообщение с прогрессом
            water_progress = f"{user['logged_water']}/{user['water_goal']} мл"
            calories_progress = f"{user['logged_calories']}/{user['calorie_goal']} ккал"
            
            await message.answer(
                f"🌟 <b>С возвращением, {name}!</b>\n\n"
                f"💧 Сегодня выпито: {water_progress}\n"
                f"🍏 Потреблено калорий: {calories_progress}\n\n"
                "Выберите действие:",
                reply_markup=create_main_menu_keyboard(),
                parse_mode="HTML"
            )
        except Exception as e:
            # Если что-то пошло не так, отправим простой текст
            await message.answer(
                f"🌟 <b>С возвращением, {name}!</b>\n\n"
                "Выберите действие:",
                reply_markup=create_main_menu_keyboard(),
                parse_mode="HTML"
            )
    else:
        await message.answer(
            f"👋 <b>Привет, {name}!</b>\nЯ ваш персональный помощник по здоровому образу жизни!\n\n"
            "Давайте настроим ваш профиль для персонализированных рекомендаций",
            parse_mode="HTML",
            reply_markup=create_profile_keyboard()
        )


@router.message(F.text == "Создать профиль")
async def handle_create_profile(message: Message, state: FSMContext):
    await set_profile(message, state)


@router.message(F.text == "👤 Профиль")
async def view_profile(message: Message):
    if not await ensure_profile(message):
        return
        
    user = get_user_profile(message.from_user.id)
    
    profile_text = (
        "👤 <b>Ваш профиль</b>\n\n"
        "▫️ <b>Вес:</b> {} кг\n"
        "▫️ <b>Рост:</b> {} см\n"
        "▫️ <b>Возраст:</b> {}\n"
        "▫️ <b>Город:</b> {}\n"
        "▫️ <b>Активность:</b> {}\n\n"
        "💧 <b>Норма воды:</b> {} мл/день\n"
        "🍏 <b>Норма калорий:</b> {} ккал/день"
    ).format(
        user['weight'], user['height'], user['age'],
        user['city'], user['activity'],
        user['water_goal'], user['calorie_goal']
    )
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✏️ Изменить профиль", callback_data="edit_profile")],
        [InlineKeyboardButton(text="📊 Показать графики", callback_data="show_graphs")]
    ])
    
    await message.answer(profile_text, reply_markup=keyboard, parse_mode="HTML")

@router.callback_query(F.data == "edit_profile")
async def edit_profile(callback: CallbackQuery, state: FSMContext):
    await callback.message.delete()
    pass  # Игнорируем ошибку, если сообщение уже удалено
    
    # Отправляем новое сообщение вместо ответа на старое
    await callback.message.answer("Давайте обновим ваш профиль! Введите ваш вес (в кг):")
    await state.set_state(UserProfile.weight)

    
@router.message(Command("set_profile"))
async def set_profile(message: Message, state: FSMContext):
    # Всегда отправляем новое сообщение вместо ответа
    await message.answer("Введите ваш вес (в кг):")
    await state.set_state(UserProfile.weight)


@router.message(UserProfile.weight)
async def process_weight(message: Message, state: FSMContext):
    try:
        weight = int(message.text)
        if weight <= 0:
            raise ValueError("Вес должен быть положительным числом.")
        await state.update_data(weight=weight)
        await message.reply("Введите ваш рост (в см):")
        await state.set_state(UserProfile.height)
    except ValueError:
        await message.reply("Пожалуйста, введите корректное значение веса (в кг).")


@router.message(UserProfile.height)
async def process_height(message: Message, state: FSMContext):
    try:
        height = int(message.text)
        if height <= 0:
            raise ValueError("Рост должен быть положительным числом.")
        await state.update_data(height=height)
        await message.reply("Введите ваш возраст:")
        await state.set_state(UserProfile.age)
    except ValueError:
        await message.reply("Пожалуйста, введите корректное значение роста (в см).")


@router.message(UserProfile.age)
async def process_age(message: Message, state: FSMContext):
    try:
        age = int(message.text)
        if age <= 0:
            raise ValueError("Возраст должен быть положительным числом.")
        await state.update_data(age=age)

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text=activity, callback_data=activity)] for activity in ACTIVITY_LEVELS.keys()
            ]
        )
        await message.reply(
            "Выберите уровень вашей активности:",
            reply_markup=keyboard
        )
        await state.set_state(UserProfile.activity)
    except ValueError:
        await message.reply("Пожалуйста, введите корректное значение возраста.")


@router.callback_query(UserProfile.activity)
async def process_activity(callback: CallbackQuery, state: FSMContext):
    selected_activity = callback.data
    if selected_activity not in ACTIVITY_LEVELS:
        await callback.answer("Некорректный выбор. Попробуйте снова.")
        return

    activity_coefficient = ACTIVITY_LEVELS[selected_activity]
    await state.update_data(activity=activity_coefficient)

    # Отправляем новое сообщение вместо редактирования
    await callback.message.answer(
        f"✅ Вы выбрали: <b>{selected_activity}</b>\nКоэффициент активности: {activity_coefficient}\n\n"
        "В каком городе вы находитесь?",
        parse_mode="HTML"
    )
    await state.set_state(UserProfile.city)


@router.message(UserProfile.city)
async def process_city(message: Message, state: FSMContext):
    data = await state.get_data()
    weight, height, age, activity = data["weight"], data["height"], data["age"], data["activity"]
    city = message.text

    weather = await get_weather(city)
    water_goal = calculate_water_goal(weight, activity, weather)
    calorie_goal = calculate_calorie_goal(weight, height, age, activity)

    users[message.from_user.id] = {
        "weight": weight,
        "height": height,
        "age": age,
        "activity": activity,
        "city": city,
        "water_goal": water_goal,
        "calorie_goal": calorie_goal,
        "logged_water": 0,
        "logged_calories": 0,
        "burned_calories": 0
    }

    await message.reply(
        f"Профиль настроен! Удачи на пути к здоровому образу жизни! 🌟\n"
        f"Ваша дневная норма воды: {water_goal} мл.\n"
        f"Ваша дневная норма калорий: {calorie_goal} ккал.\n\n"
        "Выберите действие:",
        reply_markup=create_main_menu_keyboard()
    )
    await state.clear()


@router.message(F.text == "📈 Графики прогресса")
async def handle_graph_request(message: Message):
    await send_progress_graphs(message)


async def send_progress_graphs(message: Message):
    if not await ensure_profile(message):
        return  
    
    user = get_user_profile(message.from_user.id)

    # Настройка стиля графиков
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titlepad'] = 15
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Ваш прогресс на сегодня', fontsize=16, y=1.05)
    
    # График воды
    water_left = max(user['water_goal'] - user['logged_water'], 0)
    ax1.pie(
        [user['logged_water'], water_left],
        labels=['Выпито', 'Осталось'],
        colors=['#1f77b4', '#d3d3d3'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax1.set_title('💧 Потребление воды', fontsize=14)
    
    # График калорий
    calories_left = max(user['calorie_goal'] - user['logged_calories'], 0)
    ax2.pie(
        [user['logged_calories'], calories_left],
        labels=['Съедено', 'Осталось'], 
        colors=['#2ca02c', '#d3d3d3'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax2.set_title('🍎 Баланс калорий', fontsize=14)
    
    plt.tight_layout()
    
    # Сохраняем и отправляем
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    caption = (
        "📈 <b>Ваши графики прогресса</b>\n\n"
        f"💧 Выпито воды: {user['logged_water']}/{user['water_goal']} мл\n"
        f"🍏 Потреблено калорий: {user['logged_calories']:.0f}/{user['calorie_goal']} ккал\n"
        f"🔥 Сожжено: {user['burned_calories']:.0f} ккал"
    )
    
    await message.answer_photo(
        BufferedInputFile(buf.read(), filename='progress.png'),
        caption=caption,
        parse_mode='HTML'
    )

@router.message(F.text == "💧 Добавить воду")
async def add_water(message: Message, state: FSMContext):
    if not await ensure_profile(message):
        return
    await message.reply("Введите количество воды (в мл):")
    await state.set_state(WaterLogState.amount)


@router.message(WaterLogState.amount)
async def process_water(message: Message, state: FSMContext):
    try:
        amount = int(message.text)
        if amount <= 0:
            raise ValueError("Количество воды должно быть положительным числом.")
        
        user_id = message.from_user.id
        users[user_id]["logged_water"] += amount
        
        # Создаем прогресс-бар
        def create_water_bar(current, total):
            progress = int(current / total * 10)
            return f"{'💧' * progress}{'⬜' * (10 - progress)}"
        
        water_bar = create_water_bar(
            users[user_id]['logged_water'], 
            users[user_id]['water_goal']
        )
        
        await message.reply(
            f"✅ Добавлено: <b>{amount} мл</b> воды!\n\n"
            f"{water_bar}\n"
            f"{users[user_id]['logged_water']}/{users[user_id]['water_goal']} мл",
            reply_markup=create_main_menu_keyboard(),
            parse_mode="HTML"
        )
        await state.clear()
    except ValueError:
        await message.reply("⚠️ Пожалуйста, введите корректное значение (в мл).")


@router.message(F.text == "🍴 Добавить прием пищи")
async def add_food(message: Message, state: FSMContext):
    if not await ensure_profile(message):
        return
    
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Сфотографировать еду 📷")],
            [KeyboardButton(text="Ввести название вручную ✏️")]
        ],
        resize_keyboard=True,
        one_time_keyboard=True
    )
    
    await message.reply(
        "Как вы хотите добавить прием пищи?",
        reply_markup=keyboard
    )
    await state.set_state(FoodLogState.waiting_for_food_input)


@router.message(FoodLogState.waiting_for_food_input, F.text == "Сфотографировать еду 📷")
async def request_photo(message: Message, state: FSMContext):
    await message.reply("Пожалуйста, сделайте фото вашей еды и отправьте мне:", reply_markup=ReplyKeyboardRemove())
    await state.set_state(FoodLogState.waiting_for_food_photo)


@router.message(FoodLogState.waiting_for_food_input, F.text == "Ввести название вручную ✏️")
async def request_name_manually(message: Message, state: FSMContext):
    await message.reply("Введите название продукта или блюда:", reply_markup=ReplyKeyboardRemove())
    await state.set_state(FoodLogState.waiting_for_food_name)

@router.message(FoodLogState.waiting_for_food_name)
async def process_food_name(message: Message, state: FSMContext):
    food_name = message.text
    try:
        food_name_translated = await translate_to_eng(food_name)
        food_info = await get_food_info_nutritionix(food_name_translated)

        if not food_info:
            await message.answer(
                "❌ <b>Не удалось найти информацию о продукте</b>\n"
                "Попробуйте уточнить название или ввести вручную",
                parse_mode="HTML"
            )
            return

        await state.update_data(
            food_name=food_info["name"], 
            calories_per_100g=food_info["calories"],
            selected_food=food_name_translated
        )
        
        await message.answer(
            "🍎 <b>Информация о продукте:</b>\n\n"
            f"▫️ <b>Название:</b> {food_name}\n"
            f"▫️ <b>Калории:</b> {food_info['calories']} ккал/100г\n\n"
            "<i>Сколько грамм вы съели? Укажите число:</i>",
            parse_mode="HTML"
        )
        await state.set_state(FoodLogState.waiting_for_food_weight)
        
    except Exception as e:
        logger.error(f"Ошибка обработки продукта: {e}")
        await message.answer(
            "⚠️ <b>Произошла ошибка</b>\n"
            "Попробуйте еще раз или выберите другой продукт",
            parse_mode="HTML"
        )



@router.message(FoodLogState.waiting_for_food_weight)
async def process_food_weight(message: Message, state: FSMContext):
    try:
        weight = int(message.text)
        if weight <= 0:
            raise ValueError("Вес должен быть положительным числом.")
            
        data = await state.get_data()
        selected_food = data['selected_food']
        food_name = data.get('food_name', selected_food)
        
        food_info = await get_food_info_nutritionix(selected_food)
        
        if not food_info:
            await message.answer(
                "❌ <b>Ошибка данных</b>\n"
                "Не удалось получить информацию о продукте",
                parse_mode="HTML"
            )
            return
            
        calories = (weight * food_info['calories']) / 100
        
        user_id = message.from_user.id
        users[user_id]["logged_calories"] += calories
        
        # Создаем визуализацию
        calorie_percent = min(100, (calories / users[user_id]['calorie_goal']) * 100)
        progress_bar = "🟩" * int(calorie_percent / 10) + "⬜" * (10 - int(calorie_percent / 10))
        
        await message.answer(
            "✅ <b>Прием пищи добавлен!</b>\n\n"
            f"▫️ <b>Продукт:</b> {food_name}\n"
            f"▫️ <b>Количество:</b> {weight}g\n"
            f"▫️ <b>Калории:</b> {calories:.1f} ккал\n\n"
            f"{progress_bar} {calorie_percent:.1f}% от дневной нормы",
            reply_markup=create_main_menu_keyboard(),
            parse_mode="HTML"
        )
        await state.clear()
        
    except ValueError:
        await message.answer(
            "⚠️ <b>Некорректный ввод</b>\n"
            "Пожалуйста, введите число грамм (например: 150)",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Ошибка обработки веса: {e}")
        await message.answer(
            "⚠️ <b>Произошла ошибка</b>\n"
            "Попробуйте еще раз",
            parse_mode="HTML"
        )


@router.message(F.text == "🏋️ Добавить тренировку")
async def add_activity(message: Message, state: FSMContext):
    if not await ensure_profile(message):
        return

    # Клавиатура с типами активности
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=activity, callback_data=activity)] for activity in WORKOUT_CALORIES.keys()
        ]
    )
    await message.reply("Выберите тип тренировки:", reply_markup=keyboard)
    await state.set_state(ActivityLogState.activity_type)


@router.callback_query(ActivityLogState.activity_type)
async def process_activity_selection(callback: CallbackQuery, state: FSMContext):
    activity = callback.data
    if activity not in WORKOUT_CALORIES:
        await callback.answer("Выберите тип тренировки из списка.")
        return

    await state.update_data(activity_type=activity)
    
    activity_emoji = {
        "Бег": "🏃",
        "Плавание": "🏊",
        "Ходьба": "🚶",
        "Силовая тренировка": "💪",
        "Кардио тренировка": "❤️",
        "Сноуборд": "🏂",
        "Лыжи": "⛷️",
        "Коньки": "⛸️",
        "Ролики": "🛼"
    }.get(activity, "🏋️")
    
    await callback.message.answer(
        f"{activity_emoji} <b>Вы выбрали:</b> {activity}\n\n"
        "<i>Введите длительность тренировки в минутах:</i>",
        parse_mode="HTML"
    )
    await state.set_state(ActivityLogState.duration)


@router.message(ActivityLogState.duration)
async def process_activity_duration(message: Message, state: FSMContext):
    try:
        duration = int(message.text)
        if duration <= 0:
            raise ValueError("Длительность должна быть положительной.")
            
        data = await state.get_data()
        activity_type = data["activity_type"]
        calories_burned = WORKOUT_CALORIES[activity_type] * (duration / 30)

        user_id = message.from_user.id
        users[user_id]["burned_calories"] += calories_burned
        extra_water = round((duration / 30) * 200)
        users[user_id]["water_goal"] += extra_water

        # Визуализация результатов
        progress_text = (
            f"✅ <b>Тренировка добавлена!</b>\n\n"
            f"▫️ <b>Тип:</b> {activity_type}\n"
            f"▫️ <b>Длительность:</b> {duration} мин\n"
            f"▫️ <b>Сожжено:</b> {calories_burned:.0f} ккал\n\n"
            f"💦 <b>Дополнительно выпейте:</b> {extra_water} мл воды\n\n"
            f"🔥 <b>Всего сожжено сегодня:</b> {users[user_id]['burned_calories']:.0f} ккал"
        )

        await message.answer(
            progress_text,
            reply_markup=create_main_menu_keyboard(),
            parse_mode="HTML"
        )
        await state.clear()
        
    except ValueError:
        await message.answer(
            "⚠️ <b>Некорректный ввод</b>\n"
            "Пожалуйста, введите число минут (например: 45)",
            parse_mode="HTML"
        )

@router.message(F.text == "🍽️ Полезный рецепт")
async def send_random_recipe(message: Message):
    await message.answer("🔍 <i>Ищу для вас вкусный и полезный рецепт...</i>", parse_mode="HTML")
    
    try:
        recipe_text = await get_random_tasty_recipe()
        
        # Форматируем рецепт
        formatted_recipe = (
            "🍴 <b>Вкусный и полезный рецепт</b>\n\n"
            f"{recipe_text}\n\n"
            "Приятного аппетита! 😊"
        )
        
        await message.answer(formatted_recipe, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Ошибка получения рецепта: {e}")
        await message.answer(
            "❌ <b>Не удалось загрузить рецепт</b>\n"
            "Попробуйте позже или запросите другой рецепт",
            parse_mode="HTML"
        )


@router.message(F.text == "📋 Персональные рекомендации")
async def send_recommendations(message: Message):
    if not await ensure_profile(message):
        return

    user = get_user_profile(message.from_user.id)
    calories_logged = user.get("logged_calories", 0)
    calorie_goal = user.get("calorie_goal", 0)
    calories_burned = user.get("burned_calories", 0)

    # Стилизованные рекомендации
    recommendations = ["🌟 <b>Ваши персональные рекомендации</b> 🌟\n"]
    
    # Рекомендации по воде
    water_percent = (user['logged_water'] / user['water_goal']) * 100
    if water_percent < 70:
        recommendations.append(
            "\n💦 <b>Вода:</b> Вы выпили только {:.1f}% от нормы. "
            "Попробуйте выпить стакан воды сейчас!".format(water_percent)
        )
    else:
        recommendations.append(
            "\n💧 <b>Вода:</b> Отличный прогресс! Продолжайте в том же духе."
        )

    # Рекомендации по тренировкам
    if calories_burned == 0:
        workout_rec = random.choice([
            "🏃 Попробуйте легкую пробежку сегодня!",
            "🚶 Совершите прогулку не менее 30 минут",
            "💪 Сделайте короткую силовую тренировку"
        ])
        recommendations.append(f"\n🏋️ <b>Активность:</b> {workout_rec}")
    else:
        recommendations.append(
            f"\n🔥 <b>Активность:</b> Вы уже сожгли {calories_burned:.0f} ккал! "
            "Отличный результат!"
        )

    # Рекомендации по питанию
    calorie_diff = calorie_goal - calories_logged
    if calorie_diff > 300:
        food_rec = random.choice([
            "🍎 Перекусите фруктами или орехами",
            "🥗 Приготовьте легкий овощной салат",
            "🥛 Выпейте протеиновый коктейль"
        ])
        recommendations.append(
            f"\n🍏 <b>Питание:</b> Вам нужно еще ~{calorie_diff:.0f} ккал. {food_rec}"
        )
    elif calorie_diff < -300:
        recommendations.append(
            f"\n⚠️ <b>Питание:</b> Вы превысили норму на {abs(calorie_diff):.0f} ккал. "
            "Рассмотрите легкую кардио-тренировку."
        )
    else:
        recommendations.append(
            "\n✅ <b>Питание:</b> Вы в пределах своей нормы калорий. Так держать!"
        )

    await message.answer("\n".join(recommendations), parse_mode="HTML")


@router.message(F.text == "📊 Текущий прогресс")
async def view_progress(message: Message):
    if not await ensure_profile(message):
        return

    user = users[message.from_user.id]
    
    # Создаем визуальные индикаторы
    def create_progress_bar(current, total, filled="🟩", empty="⬜"):
        ratio = min(current / total, 1)
        filled_count = round(10 * ratio)
        return filled * filled_count + empty * (10 - filled_count)
    
    water_bar = create_progress_bar(user['logged_water'], user['water_goal'], "💧", "⬜")
    calorie_bar = create_progress_bar(user['logged_calories'], user['calorie_goal'], "🍎", "⬜")
    
    # Форматируем сообщение
    progress_message = (
        "📊 <b>Ваш прогресс за сегодня</b>\n\n"
        f"{water_bar} <b>Вода:</b> {user['logged_water']}/{user['water_goal']} мл\n"
        f"{calorie_bar} <b>Калории:</b> {user['logged_calories']:.0f}/{user['calorie_goal']} ккал\n\n"
        f"🔥 <b>Сожжено калорий:</b> {user['burned_calories']:.0f}\n"
        f"⚖️ <b>Баланс:</b> {user['logged_calories'] - user['burned_calories']:.0f} ккал"
    )
    
    await message.answer(progress_message, parse_mode="HTML")


@router.message(FoodLogState.waiting_for_food_photo, F.content_type == ContentType.PHOTO)
async def handle_food_photo(message: Message, state: FSMContext):
    try:
        # Загрузка фото
        photo = message.photo[-1]
        image_buffer = BytesIO()
        await message.bot.download(photo, destination=image_buffer)
        image_bytes = image_buffer.getvalue()
        image_buffer.close()

        # Детекция еды
        detected_foods, visualized_img = food_detector.detect_and_visualize(image_bytes)

        # Проверка: ничего не найдено
        if not detected_foods:
            await message.answer_photo(
                BufferedInputFile(visualized_img, filename="detection.jpg"),
                caption="Не удалось распознать еду. Введите название вручную."
            )
            await state.set_state(FoodLogState.waiting_for_food_name)
            return

        # Переводим названия и создаём кнопки
        keyboard_buttons = []
        for food_en in detected_foods.keys():
            food_ru = await translate_to_ru(food_en)
            # Формат: Название на русском (английское)
            display_text = f"{food_ru} ({food_en})"
            keyboard_buttons.append([KeyboardButton(text=display_text)])

        keyboard = ReplyKeyboardMarkup(
            keyboard=keyboard_buttons,
            resize_keyboard=True
        )

        # Сохраняем только оригинальные данные
        await state.update_data(
            detected_foods=detected_foods,
            selected_foods={}
        )

        # Отправляем изображение с подписями
        await message.answer_photo(
            BufferedInputFile(visualized_img, filename="detection.jpg"),
            caption="Вот что я распознал:"
        )

        # Отправляем клавиатуру с переводами
        await message.reply("Выберите продукт:", reply_markup=keyboard)
        await state.set_state(FoodLogState.waiting_for_food_selection)

    except Exception as e:
        print(f"Ошибка обработки фото: {str(e)}")
        await message.reply("Ошибка при анализе фото. Введите название вручную.")
        await state.set_state(FoodLogState.waiting_for_food_name)


@router.message(FoodLogState.waiting_for_food_selection)
async def process_food_selection(message: Message, state: FSMContext):
    try:
        data = await state.get_data()
        detected_foods = data.get('detected_foods', {})
        
        # Извлекаем английское название из текста (формат "русское (английское)")
        selected_text = message.text
        if "(" in selected_text and ")" in selected_text:
            food_en = selected_text.split("(")[1].split(")")[0].strip()
        else:
            food_en = selected_text
            
        if food_en not in detected_foods:
            await message.reply("Пожалуйста, выберите продукт из списка.", reply_markup=ReplyKeyboardRemove())
            return
            
        # Сохраняем выбранный продукт (английское название)
        await state.update_data(selected_food=food_en)
        
        await message.reply(
            f"Выбрано: {selected_text}\nВведите количество в граммах:",
            reply_markup=ReplyKeyboardRemove()
        )
        await state.set_state(FoodLogState.waiting_for_food_weight)
        
    except Exception as e:
        print(f"Ошибка выбора продукта: {str(e)}")
        await message.reply("Произошла ошибка. Попробуйте снова.", reply_markup=ReplyKeyboardRemove())


@router.errors()
async def global_error_handler(event: Update, exception: Exception):
    """Глобальный обработчик ошибок"""
    logging.error(f"Ошибка при обработке запроса: {exception}", exc_info=True)
    
    # Попробуем отправить сообщение пользователю
    try:
        chat_id = event.message.chat.id if event.message else event.callback_query.message.chat.id
        await event.bot.send_message(
            chat_id,
            "⚠️ Произошла непредвиденная ошибка. Пожалуйста, попробуйте снова или начните заново с /start"
        )
    except Exception as e:
        logging.error(f"Ошибка при отправке сообщения об ошибке: {e}")
    
    return True


def setup_handlers(dp):
    dp.include_router(router)