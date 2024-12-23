import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ========================================================
# 1) データの読み込み (ImageDataGeneratorを使用)
# ========================================================
# ディレクトリ構成: sample/<クラス名>/
# classes の指定でフォルダをクラスとして扱う
DATA_DIR = 'sample'
IMG_SIZE = (128, 128)  # リサイズ後のサイズ(任意)
BATCH_SIZE = 32
SAVE_DIR = 'model'

# 以下のクラスリストを用意 (フォルダ名と一致させる)
classes = [
    "mandibular_left_1st_premolar",
    "mandibular_left_2st_premolar",
    "mandibular_right_1st_premolar",
    "mandibular_right_2st_premolar",
]

# ImageDataGeneratorで学習用とテスト用に分けるため、
# まずはtrain用生成器を用意し、validation_splitで分割する
datagen = ImageDataGenerator(
    rescale=1.0/255,         # 画素値を0～1に正規化
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    fill_mode="nearest",
    validation_split=0.2     # 20% を検証データに
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    classes=classes,
    class_mode='categorical',
    subset='training',      # trainingデータ
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    classes=classes,
    class_mode='categorical',
    subset='validation',    # validationデータ
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ========================================================
# 2) CNNモデルの構築
# ========================================================
model = Sequential([
    # 畳み込み層1
    Conv2D(64, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2,2)),

    # 畳み込み層2
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    # 全結合へ向けてFlatten
    Flatten(),

    # 全結合層
    Dense(256, activation='relu'),
    Dropout(0.5),

    # 出力層(クラス数=4)
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========================================================
# 3) モデルの学習
# ========================================================
EPOCHS = 15

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# ========================================================
# 5) テストデータでの評価
# ========================================================
# この例では train_gen/val_gen のみ用意しましたが、
# 実際には train/val とは別に test 用フォルダを作って
# 評価することを推奨します。
# ここでは val_gen を「テスト的に」評価してみます。
val_gen.reset()  # 評価前にバッチの位置をリセットしておく
pred_probs = model.predict(val_gen)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = val_gen.classes  # flow_from_directory で自動生成される

acc = accuracy_score(true_labels, pred_labels)
print(f"Validation Accuracy: {acc*100:.2f}%")

print(classification_report(true_labels, pred_labels, target_names=classes))

model.save(os.path.join(SAVE_DIR,"teeth_cnn_saved_model.h5"))