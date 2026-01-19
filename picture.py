from PIL import Image
import requests
from io import BytesIO


class Pictures:
    @staticmethod
    def generate(prompt: str, filename: str):
        """
        Генерация изображения через Pollinations.ai и сохранение в PNG.

        prompt: описание изображения
        filename: имя файла без расширения
        """
        url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '+')}"
        response = requests.get(url)
        if response.status_code != 200:
            print("Ошибка при генерации изображения:", response.status_code)
            return

        # Загружаем изображение в PIL
        img = Image.open(BytesIO(response.content))

        # Масштабируем до 32x32 пикселей (пиксельная графика)
        #img = img.resize((32, 32), resample=Image.NEAREST)

        # Сохраняем как PNG с расширением
        output_file = f"{filename}.png"
        img.save(output_file)
        print(f"Изображение сохранено как {output_file} (32x32)")


# Пример использования
name = "iron slavic helmet"
prompt = (
    f"32x32 pixel {name}, simple colors, cartoon style, "
    "black background"
    "minimalistic, clear outline"
)

Pictures.generate(prompt, name)
