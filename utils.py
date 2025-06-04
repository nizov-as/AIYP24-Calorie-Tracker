import aiohttp
import httpx
import requests
import random
from config import API_KEY, API_KEY_TASTY, API_HOST, API_KEY_NUTR, APP_ID


async def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data["current"]["temp_c"]
    return 20  # если вдруг не удалось вернуть температуру, вернем 20 градусов


async def get_food_info(product_name):
    url = f"https://world.openfoodfacts.org/cgi/search.pl?action=process&search_terms={product_name}&json=true"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                products = data.get('products', [])
                if products:
                    first_product = products[0]
                    return {
                        'name': first_product.get('product_name', 'Неизвестно'),
                        'calories': first_product.get('nutriments', {}).get('energy-kcal_100g', 0)
                    }
    return None


async def get_food_info_nutritionix(product_name):
    url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'

    headers = {
        'x-app-id': APP_ID,
        'x-app-key': API_KEY_NUTR,
        'Content-Type': 'application/json',
    }

    data = {
        "query": product_name,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                if 'foods' in data and len(data['foods']) > 0:
                    first_product = data['foods'][0]
                    return {
                        'name': first_product.get('food_name', 'Неизвестно'),
                        'calories': first_product.get('nf_calories', 0)
                    }
                else:
                    return None
            else:
                print(f"Error {response.status}: {await response.text()}")
                return None


def calculate_water_goal(weight, activity, temperature):
    base = weight * 30
    extra_activity = (activity // 30) * 500
    extra_temp = 500 if temperature > 25 else 0
    return base + extra_activity + extra_temp


def calculate_calorie_goal(weight, height, age, activity_level):
    return (10 * weight + 6.25 * height - 5 * age + 5) * activity_level


async def get_random_tasty_recipe():
    """Получает случайный рецепт из Tasty API."""
    url = f"https://{API_HOST}/recipes/list"
    headers = {
        "x-rapidapi-host": API_HOST,
        "x-rapidapi-key": API_KEY_TASTY
    }
    params = {
        "from": 0,
        "size": 20,
        "tags": "under_30_minutes"  # Рецепты, которые можно приготовить за 30 минут
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data and data["results"]:
            # Выбираем случайный рецепт
            recipe = random.choice(data["results"])
            title = recipe.get("name", "Без названия")
            description = recipe.get("description", "Описание отсутствует")
            recipe_url = recipe.get("original_video_url", "Нет видео")

            text = f"🍴 {title}\n\n{description}\n\n🔗 Видео-рецепт: {recipe_url}"
            return text
        else:
            return "Не удалось найти рецепты. Попробуйте позже."
    else:
        return f"Ошибка API: {response.status_code}"