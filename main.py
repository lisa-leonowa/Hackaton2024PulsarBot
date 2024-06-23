import os
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

API_TOKEN = '6669661722:AAFkxCrSipnuciGyd2UqUbwNiCj3eFO8sH0'
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

np.set_printoptions(suppress=True)

model = load_model("C:/Users/eliza/PycharmProjects/pythonhakaton/keras_model.h5", compile=False)

class_names = open("C:/Users/eliza/PycharmProjects/pythonhakaton/labels.txt", "r").readlines()


def model_img(username_tg):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(f"C:/Users/eliza/PycharmProjects/pythonhakaton/download_img/{username_tg}.jpg").convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    answer = [str(class_name[2:]), str(confidence_score)]
    return answer


@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    photo_id = message.photo[-1].file_id
    file_info = await bot.get_file(photo_id)
    file_path = file_info.file_path
    file_url = f'https://api.telegram.org/file/bot{API_TOKEN}/{file_path}'

    async with bot.session.get(file_url) as response:
        if response.status == 200:
            photo_data = await response.read()
            path = f'C:/Users/eliza/PycharmProjects/pythonhakaton/download_img/{message.chat.username}.jpg'
            with open(path, 'wb') as file:
                file.write(photo_data)
            answer = model_img(message.chat.username)
            os.remove(f'C:/Users/eliza/PycharmProjects/pythonhakaton/download_img/{message.chat.username}.jpg')
            if str(answer[0]) == 'LOGO\n':
                await message.reply('Является одним из Национальных проектов:\n'
                                    '1. Экология\n'
                                    '2. Производительность труда\n'
                                    '3. Туризм и индустрия гостеприимства\n'
                                    '4. Цифровая экономика\n'
                                    '5. Образование\n'
                                    '6. Магистральный план энергетическая часть\n'
                                    '7. Малое и среднее предпринимательство и поддержка индивидуальной '
                                    'предпринимательской инициативы\n'
                                    '8. Международная кооперация и экспорт\n'
                                    '9. Наука и университеты\n'
                                    '10. Культура\n'
                                    '11. Здравоохранение\n'
                                    '12. Комплексный план модернизации и расширения магистральной инфраструктуры\n'
                                    '13. Жилье и городская среда\n'
                                    '14. Демография\n'
                                    '15. Безопасные качественные дороги\n')
            elif str(answer[0]) == 'BAD\n':
                await message.reply("Не является Национальным проектом!")
            else:
                await message.reply("Не удалось распознать фото!")
        else:
            await message.reply("Не удалось загрузить фото!")


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.answer("Привет! Я бот, который может распознавать фотографии и определять, "
                         "является ли объект на фото Национальным проектом."
                         " Просто отправь мне фотографию, и я попробую ее распознать.")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
