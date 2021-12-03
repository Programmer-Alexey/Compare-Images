from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore
from compare2images import Ui_MainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog

import cv2
import sys
import time

from db import Buffer, Database


class MyWidget(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.key = 1
        self.index = 0
        self.similar_images = None
        self.path1 = None
        self.path2 = None

        self.setupUi(self)
        self.button_next.setVisible(False)
        self.score_label.setText('')
        self.text_label.setText('Сравните две фотографии.')
        self.text_label.setAlignment(QtCore.Qt.AlignCenter)

        width = self.left_label.size().width()
        height = self.left_label.size().height()

        self.pixmap = QPixmap('images/gallery_icon.png')
        self.pixmap = self.pixmap.scaled(width * 10, height, QtCore.Qt.KeepAspectRatio)

        self.left_label.setPixmap(self.pixmap)
        self.left_label.setAlignment(QtCore.Qt.AlignCenter)

        self.right_label.setPixmap(self.pixmap)
        self.right_label.setAlignment(QtCore.Qt.AlignCenter)

        self.find_similar.clicked.connect(self.fsimilar)
        self.button_next.clicked.connect(self.next_image)
        self.compare2images.clicked.connect(self.start_condition)
        self.who_am_i.clicked.connect(self.videoflow)
        self.btn_send.clicked.connect(self.send_to_comparing)

    def videoflow(self):  # Достаем кадр из видеопотока. Только если есть камера.
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        recognize_model = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        start = time.time()
        while True:
            ret, img = cap.read()
            frame = cv2.flip(img, 1)
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = recognize_model.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (110, 110, 110), 2)

            if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start >= 5:
                cv2.imwrite('YourPhoto.jpg', img)
                break
            cv2.imshow('LOOK ON CAMERA', frame)

        cv2.destroyAllWindows()

        self.key = 2
        width = self.right_label.size().width()
        height = self.right_label.size().height()

        self.path1 = 'YourPhoto.jpg'
        pixmap = QPixmap('YourPhoto.jpg').scaled(width, height, QtCore.Qt.KeepAspectRatio)
        self.left_label.setPixmap(pixmap)
        self.send_to_comparing()

    def next_image(self):
        """Функция, которая достает изображения из базы данных и показывает их по очереди."""
        try:
            image = self.similar_images[self.index % len(self.similar_images)][0]
            self.index += 1

            buf = Buffer('image')
            buf.put_image(image, name='image.jpg')

            pixmap = QPixmap('image/image.jpg')
            pixmap = pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio)
            # pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio)
            self.right_label.setPixmap(pixmap)
            self.text_label.setText('Нашлись изображения')
            buf.close()
        except Exception:
            self.button_next.setVisible(False)
            width = self.right_label.size().width()
            height = self.right_label.size().height()
            who_pixmap = QPixmap('who_icon.png')
            who_pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)
            self.right_label.setPixmap(who_pixmap)
            self.right_label.setAlignment(QtCore.Qt.AlignCenter)
            self.text_label.setText("Похожих не нашлось")

    def fsimilar(self):  # Состояние 2: "Найти похожие". Нужно доработать отображение
        self.key = 2
        width = self.right_label.size().width()
        height = self.right_label.size().height()

        who_pixmap = QPixmap('images/who_icon.png')
        who_pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)
        self.right_label.setPixmap(who_pixmap)
        self.right_label.setAlignment(QtCore.Qt.AlignCenter)

        self.text_label.setText('Поиск по базе данных. Положите одно изображение в левый блок')

        self.pixmap = QPixmap('images/gallery_icon.png')
        self.pixmap = self.pixmap.scaled(width * 10, height, QtCore.Qt.KeepAspectRatio)

        self.left_label.setPixmap(self.pixmap)
        self.left_label.setAlignment(QtCore.Qt.AlignCenter)

    def start_condition(self):  # Начальное положение и настройки
        self.key = 1
        self.button_next.setVisible(False)
        self.score_label.setText('')
        self.text_label.setText('Сравните две фотографии.')
        self.text_label.setAlignment(QtCore.Qt.AlignCenter)

        width = self.left_label.size().width()
        height = self.left_label.size().height()

        self.pixmap = QPixmap('images/gallery_icon.png')
        self.pixmap = self.pixmap.scaled(width * 10, height, QtCore.Qt.KeepAspectRatio)

        self.left_label.setPixmap(self.pixmap)
        self.left_label.setAlignment(QtCore.Qt.AlignCenter)

        self.right_label.setPixmap(self.pixmap)
        self.right_label.setAlignment(QtCore.Qt.AlignCenter)

    def mousePressEvent(self, event):  # Реализация диалогового окна без использования кнопок.
        if self.key == 1:
            width = self.left_label.size().width()
            height = self.left_label.size().height()
            if 30 <= event.x() <= 30 + width and 10 <= event.y() <= 10 + height:
                self.path1 = QFileDialog.getOpenFileName(self, 'Выбрать Картинку', '')[0]
                self.im_to_label(self.left_label, self.path1)
            if 35 + width <= event.x() <= 35 + 2 * width and 10 <= event.y() <= 10 + height:
                self.path2 = QFileDialog.getOpenFileName(self, 'Выбрать Картинку', '')[0]
                self.im_to_label(self.right_label, self.path2)
        elif self.key == 2:
            width = self.left_label.size().width()
            height = self.left_label.size().height()
            if 30 <= event.x() <= 30 + width and 10 <= event.y() <= 10 + height:
                self.path1 = QFileDialog.getOpenFileName(self, 'Выбрать Картинку', '')[0]
                self.im_to_label(self.left_label, self.path1)

    def send_to_comparing(self):  # Отправляем изображения на проверку
        if self.key == 1:
            if self.path1 is None and self.path2 is None or len(self.path1) == 0 or len(self.path2) == 0:
                self.text_label.setText('Вы не выбрали фотографии!')
            elif self.path1 is None:
                self.text_label.setText('Вы не выбрали левую фотографию!')
            elif self.path2 is None:
                self.text_label.setText('Вы не выбрали правую фотографию!')
            else:
                self.text_label.setText('')

                database = Database('my_db.sqlite')

                # Кладем изображения в базу данных
                """hash1 = image_hash(self.path1)
                with open(self.path1, 'rb') as file1:
                    bin_image1 = file1.read()
                database.put_image(hash1, bin_image1)

                hash2 = image_hash(self.path2)
                with open(self.path1, 'rb') as file2:
                    bin_image2 = file2.read()
                database.put_image(hash2, bin_image2)
                database.close()"""

                # Сравнивание фотографий
                compare_image = CompareImage(self.path1, self.path2)
                image_difference = round(compare_image.compare_image(), 3) * 1000
                if image_difference >= 500:
                    text = f'Совпадение: {int(image_difference // 10)}.{int(image_difference % 10)}%,' \
                           f' вы однозначно похожи!'
                else:
                    text = f'Совпадение: {int(image_difference // 10)}.{int(image_difference % 10)}%,' \
                           f' кажется, вы не похожи!'
                self.score_label.setText(f'{int(image_difference // 10)}.{int(image_difference % 10)}%')
                self.text_label.setText(text)
                self.score_label.setAlignment(QtCore.Qt.AlignCenter)
        elif self.key == 2:
            if self.path1 is None:
                self.text_label.setText('Вы не выбрали фотографию!')
            else:
                hash1 = image_hash(self.path1)
                database = Database('my_db.sqlite')
                self.similar_images = list(database.get_images(hash1))
                self.button_next.setVisible(True)
                self.next_image()

    # Изменение размера виджетов при увеличении размера окна(На будущее)
    """def resizeEvent(self, event):
        pass"""

    def im_to_label(self, label, path):  # Функция, которая вставляет изображение в блок
        if path is None:
            pass
        else:
            pixmap = QPixmap(path)
            pixmap = pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio)
            label.setPixmap(pixmap)


def cropper(image_path):  # Функция обрезки части фотографии, где есть лицо
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detect_model = cv2.CascadeClassifier(r'cascades/haarcascade_frontalface_default.xml')

    faces = detect_model.detectMultiScale(gray, minSize=(100, 100))

    try:
        for x, y, w, h in faces:
            crop = image[y: y + h, x: x + w]
        return crop
    except NameError:
        print('Лицо не нашлось')


"""Далее идет блок преобразования изображения 
в хеш для удобного поиска в базе данных.
Вдобавок идут функции для Сравнения двух хешей."""


def image_hash(image_path):
    image = cropper(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hashing_image = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    average_pixel = hashing_image.mean()

    bin_image = cv2.threshold(hashing_image, average_pixel, 255, 0)[1]
    img_hash = ''
    for i in range(8):
        for j in range(8):
            if bin_image[i][j] == 255:
                img_hash += '1'
            else:
                img_hash += '0'
    return '1' + img_hash


def get_hash_difference(hash1, hash2):
    diff = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            diff += 1
    return diff


def compare_hash(image1_path, image2_path):
    image1 = cropper(image1_path)
    image2 = cropper(image2_path)

    hash1 = image_hash(image1)
    hash2 = image_hash(image2)

    return 1 - get_hash_difference(hash1, hash2) / len(hash1)


class CompareImage(object):  # Класс сравнения изображений.

    def __init__(self, image_1_path, image_2_path):
        self.minimum_commutative_image_diff = 1
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path

    def compare_image(self):
        image_1 = cropper(self.image_1_path)
        image_2 = cropper(self.image_2_path)
        commutative_image_diff = self.get_image_difference(image_1, image_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            print("Matched")
            return commutative_image_diff
        return commutative_image_diff - 1

    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)

        #
        img_template_probability_match = \
            cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # Далее сравниерние по гистограммам и через ии-модель
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWidget()
    ex.show()
    sys.exit(app.exec_())

# 36893488147419103230 max hash
