import os
import tempfile

from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# ---------------------------
# 事前学習済みモデルをロード
# ---------------------------
MODEL_PATH = "model/teeth_cnn_saved_model.h5"
model = load_model(MODEL_PATH)

# 学習時のクラス順序に合わせてクラス名を定義
classes = [
    "mandibular_left_1st_premolar",
    "mandibular_left_2st_premolar",
    "mandibular_right_1st_premolar",
    "mandibular_right_2st_premolar",
]

class_name = {
    "mandibular_left_1st_premolar" : "下顎左第1小臼歯",
    "mandibular_left_2st_premolar" : "下顎左第2小臼歯",
    "mandibular_right_1st_premolar" : "下顎右第1小臼歯",
    "mandibular_right_2st_premolar" : "下顎右第2小臼歯"
}

# ---------------------------
# ルート: 画像アップロードフォーム
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']  # <input type="file" name="file"> に対応
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                temp_path = tmp.name
                file.save(temp_path)

            # 画像をモデル入力用に前処理
            img = image.load_img(temp_path, target_size=(128, 128))
            x = image.img_to_array(img)
            x = x / 255.0  # 学習時にrescale=1/255していたので、同じ処理を行う
            x = np.expand_dims(x, axis=0)  # (1, 128, 128, 3) へ

            # 推論
            preds = model.predict(x)
            pred_idx = np.argmax(preds, axis=1)[0]
            predicted_class = classes[pred_idx]

            os.remove(temp_path)

            # 結果表示ページ(result.html)へ
            return render_template('result.html', 
                                   predicted_class=class_name[predicted_class])
    else:
        # GETメソッドならアップロードフォーム(index.html)を表示
        return render_template('index.html')


# ---------------------------
# Flaskの起動
# ---------------------------
if __name__ == '__main__':
    # ポート番号やデバッグモードは適宜変更してください
    app.run(debug=True)
