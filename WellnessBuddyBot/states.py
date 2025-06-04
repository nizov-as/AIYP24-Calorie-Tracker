from aiogram.fsm.state import State, StatesGroup


class UserProfile(StatesGroup):
    weight = State()
    height = State()
    age = State()
    activity = State()
    city = State()
    calorie_goal = State()


class FoodLogState(StatesGroup):
    waiting_for_food_input = State()
    waiting_for_food_photo = State()
    waiting_for_food_selection = State()
    waiting_for_food_name = State()
    waiting_for_food_weight = State()


class ActivityLogState(StatesGroup):
    activity_type = State()
    duration = State()


class WaterLogState(StatesGroup):
    amount = State()
