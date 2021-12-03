import sqlite3
import os
import shutil


class Database:
    def __init__(self, path):
        self.path = path
        self.connection = sqlite3.connect(self.path)

    def get_images(self, hash):
        try:
            cur = self.connection.cursor()
            result = cur.execute(f"""SELECT DISTINCT image FROM images
                               WHERE hash = {hash}""")
            self.connection.commit()

            return result
        except sqlite3.Error as err:
            print('Что-то не так: ', err)
        finally:
            print('Подключение к базе данных завершено.')

    def put_image(self, hash, bin_image, name='None'):
        info_tuple = (hash, bin_image, name)
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"""INSERT INTO images(hash, image, name)
             VALUES(?, ?, ?)""", info_tuple)

            self.connection.commit()

            print('Изображение загружено')

            cursor.close()
        except sqlite3.Error as err:
            print('Что-то не так: ', err)

    def close(self):
        self.connection.close()


class Buffer:
    def __init__(self, name_of_directory):
        self.name_of_directory = name_of_directory
        self.path = os.getcwd() + f'\\{name_of_directory}'
        self.count = 0
        self.last_image_path = ''

        os.mkdir(name_of_directory)
        os.system(f'attrib {self.path} +H')

    def __repr__(self):
        return f'Путь к директории: {self.path}'

    def put_image(self, bin_image, name='BufferingImage.jpg'):
        if os.path.exists(self.path + f'\\{name}'):
            self.count += 1
            self.last_image_path = self.path + f'\\{name}({self.count})'
            with open(self.last_image_path, 'wb') as image:
                image.write(bin_image)
        else:
            self.last_image_path = self.path + f'\\{name}'
            with open(self.last_image_path, 'wb') as image:
                image.write(bin_image)
                self.count = 0
        print('Process is passed')

    def get_image(self, name):
        try:
            with open(self.path + f'\\{name}') as image_file:
                return image_file.read()
        except FileNotFoundError:
            return 'Файл не найден'

    def get_path(self):
        return self.path

    def get_last_image(self):
        return self.last_image_path

    def close(self):
        shutil.rmtree(self.path)