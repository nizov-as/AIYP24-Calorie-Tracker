# Отрываем входной файл для чтения
with open(r"C:\Users\krugd\OneDrive\Рабочий стол\HSE_progect\dataset\AIYP24-Calorie-Tracker\UECFOOD256\category.txt", 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# Открываем выходной файл для записи
with open('categ_need.txt', 'w', encoding='utf-8') as categ_need:
    for line in lines:
        # Разделяем строку на части по пробелам
        parts = line.split(maxsplit=1)
        # Если у нас есть хотя бы одно число и описание
        if len(parts) == 2:
            number, description = parts
            # Записываем в выходной файл с разделителем ":"
            categ_need.write(f"{number.strip()} : {description.strip()}\n")
        else:
            categ_need.write(line)

print("Файл успешно обработан.")
