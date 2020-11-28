# PropantCheckChallenge - Mobile
## Содержание
1. [Описание решения](#overview)
2. [Запуск обучения](#training)
```
Ne_Beite
├── data
│   ├── labels
│   │   └── train.csv
│   └── train
├── models
│   ├── regr_tree.model
├── README
├── train_distributions.py
├── preprocess.py
├── hough.py
├── requirements.txt
└── run.py
```
## Описание решения <a name="overview"></a> 
 С помощью методов из OpenCV мы вырезаем из фотографии внутренний прямоугольник с гранулами проппанта и запускаем cv2.HoughCircles() для поиска окружностей на картинке. Далее высчитываем средний радиус и делим количество пикселей на переднем плане на фото на площадь круга с средним радиусом. Это и будет количеством гранул в ответе. 
 Для подсчёта распределения гранул по ситам строим распределения радиусов найденных окружностей и по ним обучаем DecisionTreeRegressor, который затем будет сконвертирован в нейронку Torch для предсказания распределения бинов.

## Запуск обучения <a name="training"></a>
1. Для установки виртуального окружения и зависимостей:\
```$ python -m venv env```  
```$ source env/bin/activate```  on Linux, ```$ env\Scripts\activate``` on Windows  
```$ pip install -r requirements.txt```  
2. Загрузите фотографии в папку data/train
3. Для запуска процесса обучения и создания модели:\
```(env) $ python run.py```

Модель regr_tree.model сохранится в папке models.
