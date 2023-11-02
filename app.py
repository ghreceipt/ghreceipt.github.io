from flask import Flask, redirect, render_template, request, url_for

#画像のアップロード先のディレクトリ
UPLOAD_DIR = 'images'
#アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = {'png', 'jpg', 'gif'}

app = Flask(__name__)
app.debug = True
app.config['upload_dir'] = UPLOAD_DIR

from PIL import Image
import numpy as np
import os
import shutil
import h5py
from tensorflow.keras.models import load_model

@app.route('/cifar10', methods=['POST'])
def upload():
    #アップロードファイルがあるかどうかのチェック
    f = request.files.get('upfile')
    if f.filename == '':
        print(f'ファイルがありません。{request.url}へリダイレクトします。')
        return redirect(request.url)

    #許可された拡張子かどうかのチェック
    ext = request.files.get('upfile').filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        print(f'拡張子{ext}は不正です。')
        return redirect(request.url)

    #アップロードされたファイルをサーバーに保存
    f.save(os.path.join(UPLOAD_DIR, f.filename))

    #画像ファイルの読み込み
    img = Image.open(os.path.join(UPLOAD_DIR, f.filename))
    img = img.convert('RGB') #アルファを除去
    img = img.resize((32, 32))
    img = np.asarray(img)

    print(img.shape)
    img = img.reshape((1, 32, 32, 3))

    #モデルで結果を予測
    model = load_model('model/cnn_model.h5', compile=True)
    print(img.shape)
    pred = model.predict(img)
    print('model predict:', pred)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    pred_class_name = class_names[np.argmax(pred)]
    print(pred_class_name)

    #pic_with_path = str(f.filename)
    print(url_for('static', filename=os.path.join('images', f.filename)))
    return render_template('index.html', result=pred_class_name, img_src=url_for('static', filename='image/'+f.filename))

@app.route('/cifar10', methods=['GET'])
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()