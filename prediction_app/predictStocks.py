import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from numpy import newaxis
from sklearn.preprocessing import MinMaxScaler
import logging
import sys

logger = logging.getLogger(__name__)

def normalize_data(df):
    logger.info("Normalizing data")
    scaler = MinMaxScaler()
    window_size = df.shape[0] // 2
    for di in range(0, df.shape[0], window_size):
        scaler.fit(df.iloc[di:di + window_size, :])
        df.iloc[di:di + window_size, :] = scaler.transform(df.iloc[di:di + window_size, :])
    return df, scaler


def split_data(data):
    test_perc = 5
    test_set_size = int(np.round(test_perc / 100 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size
    logger.info("Training Set Size: %s Test Set Size: %s", train_set_size, test_set_size)

    train_data = data[:train_set_size, :, :]
    test_data = data[train_set_size:, :, :]
    train_data = smoothing(train_data)
    return [train_data, test_data]


def timesteps(data, time_step):
    data_raw = data.values
    data = [data_raw[index: index + time_step] for index in range(len(data_raw) - time_step)]

    logger.info("Putting data into timesteps of size: %s", time_step)
    logger.info("Number of timesteps: %s", len(data_raw)/time_step)
    return np.array(data)

def smoothing(train):
    # smoothing function
    logger.info("Performing EMA smoothing function")
    alpha = 0.5
    for j in range(5):
        ema = 0.0
        for i in range(train.shape[0]):
            ema = alpha * train[i, :, j] + (1 - alpha) * ema
            train[i, :, j] = ema
    return train


def build_model(layers):
    model = Sequential()

    model.add(GRU(
        input_shape=(None, layers[0]),
        units=layers[1],
        return_sequences=True,
    ))

    model.add(LSTM(
        layers[2],
        return_sequences=False,
    ))
    model.add(Dropout(0.2))

    model.add(Dense(
        units=layers[3]
    ))
    model.add(Activation("linear"))

    model.compile(loss="mean_squared_error", optimizer="Adadelta")
    return model


def predict_point_by_point(model, data):
    # predict each timestep given the last sequence of true data
    # in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    logger.info("Predicting each data point one by one")
    # predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequences_full(model, data, window_size):
    # shift the window by 1 new prediction each time
    # re-run predictions on new window

    # declare a temp variable to hold data and one to hold predicted values
    logger.info("Predicting full sequence")
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        # run the model on data in curr_frame and spit out the first predicted value
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0:5])

        # slice first value off of curr_frame
        curr_frame = curr_frame[1:]

        # add predicted value on the back end of curr_frame and use for predictions
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return np.array(predicted)


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    logger.info("Predicting 50 data points based on each timestep")
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0:5])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return np.array(prediction_seqs)


def predict_future(model, data, window_size, prediction_len, future_steps):
    logger.info("Predicting data points past the end of the test set")
    curr_frame = data[-prediction_len + future_steps]
    predicted = []
    for i in range(prediction_len):
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0:5])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return np.array(predicted)


def plot_results(predicted_data, true_data):
    logger.info("Plotting Results")
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data)
    plt.plot(predicted_data)
    plt.show()


def plot_results_future(predicted_data, true_data, predicted_len, future_steps):
    logger.info("Plotting Future Predictions")
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data)
    plt.plot(range(predicted_len + future_steps, predicted_len + future_steps + 25), predicted_data)
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    logger.info("Plotting sequences of predictions")
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data)
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        plt.plot(range(i * prediction_len, i * prediction_len + prediction_len), data)
    plt.show()


def processStock(df):
    # create a copy of the stock data and normalize
    # Normalize between 0&1 created a better fit for
    time_steps = 25
    n_steps = time_steps - 1
    batch_size = 50
    n_epochs = 80
    future_steps = 5
    curr_price = df.iloc[-1, 3]

    try:
        df, scaler = normalize_data(df)
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to Normalize data: %s", str(error))

    # Adds column for time_steps based on time_step
    try:
        df = timesteps(df, time_steps)
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to split data into timesteps: %s", str(error))

    # split data into train and test-sets
    try:
        train, test = split_data(df)
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to split data into test/train sets: %s", str(error))

    # target datasets are last timesteps, used for verification of model
    train_data = train[:, :-1, :]
    train_target = train[:, -1, :]

    test_data = test[:, :-1, :]
    test_target = test[:, -1, :]

    try:
        model = build_model([5, 50, 50, 5])
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to Build Model: %s", str(error))

    logger.info("Fitting Model")
    try:
        model.fit(
            train_data,
            train_target,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=0
        )
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to unable to fit model: %s", str(error))

    logger.info("Running Predictions")
    #predictedp = predict_point_by_point(model, test_data)
    #predictedm = predict_sequences_multiple(model, test_data, n_steps, time_steps)
    #predictedf = predict_sequences_full(model, test_data, n_steps)
    try:
        predictedfuture = predict_future(model, test_data, n_steps, time_steps, future_steps)
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to predict values %s", str(error))

    logger.info("Reversing Scaler")
    try:
        predictedfuture = scaler.inverse_transform(predictedfuture)
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to inverse transformation: %s", str(error))

    try:
        result = list(predictedfuture[-5:,3])
        extra_data = [
            predictedfuture[-5:-4,1][0],
            curr_price,
            model.evaluate(test_data, test_target, verbose=0),
            len(df)
        ]
        result.extend(extra_data)
    except:
        error = sys.exc_info()[0]
        logger.debug("Unable to format results array: %s", str(error))

    logger.info("Finished Predicting Returning Results.")
    return result
