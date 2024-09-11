from flask import Flask, request, render_template, send_file
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', show_plot=False)

@app.route('/plot', methods=['POST'])
def plot():
    from_year = int(request.form['from_year'])
    ref_days = int(request.form['ref_days'])
    code = request.form['code']
    code_dl = code + ".t"

    end_date = datetime.now()
    start_date = datetime(end_date.year - from_year, 1, 1)
    df = yf.download(code_dl, start=start_date, end=end_date, interval="1d")

    # データの前処理
    data = df.filter(["Close"])
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # 訓練データと検証データの分割
    training_data_len = int(np.ceil(len(dataset) * 0.7))
    train_data = scaled_data[0:int(training_data_len), :]

    # 訓練データの作成
    x_train, y_train = [], []
    for i in range(ref_days, len(train_data)):
        x_train.append(train_data[i-ref_days:i, 0])
        y_train.append(train_data[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # LSTMモデル構築
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # モデルの訓練
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # モデルの保存
    savefile = os.path.join(os.path.dirname(__file__), "kabuka_o.h5")
    model.save(savefile)

    # 検証用データの作成
    test_data = scaled_data[training_data_len - ref_days:, :]
    x_test, y_test = [], dataset[training_data_len:, :]
    for i in range(ref_days, len(test_data)):
        x_test.append(test_data[i-ref_days:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # 予測値の算出
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # 予測のプロット
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16, 6))
    plt.title('LSTM Model Predictions')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Real', 'Prediction'], loc='lower right')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # プロットを閉じる
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
