import time
import pandas as pd
from sklearn.metrics import mean_squared_error
from predictStocks import normalize_data, split_data, timesteps, build_model, predict_point_by_point


# run a repeated experiment
def experiment(repeats, series, n_epochs, time_step, batch_size):
    # transform data to be supervised learning
    series_norm, scaler = normalize_data(series)

    #Adds time_steps dimension
    series_withsteps = timesteps(series_norm, time_step)

    # split data into train and test-sets
    train, test = split_data(series_withsteps)

    train_data = train[:, :-1, :]
    train_target = train[:, -1, :]

    test_data = test[:, :-1, :]
    test_target = test[:, -1, :]

    model = build_model([5, 50, 50, 5])

    start_time = time.time()

    # run experiment
    error_scores = []
    run_time = []
    for r in range(repeats):
        # fit the model
        model.fit(
            train_data,
            train_target,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=2
        )

        # forecast test dataset
        predictedptest = predict_point_by_point(model, test_data)
        # report performance
        rmse = (mean_squared_error(test_target[:, 3], predictedptest[:, 3]))**0.5
        print('%d) Test RMSE: %.3f' % (r + 1, rmse))
        error_scores.append(rmse)
        run_time.append(time.time() - start_time)
    return error_scores, run_time


# load dataset and drop na values
df = pd.read_csv("C:/Users/jmper/OneDrive/Documents/Data Sets/Stocks/LGCY.csv", usecols=[1, 2, 3, 4, 5])
df = df.dropna()

#define experiment parameters
time_step = 25
batch_size = 50

# experiment
repeats = 30
results = pd.DataFrame()
run_time = pd.DataFrame()
# vary training epochs
epochs = 80
#for l in layer:
results['return states'],run_time['return states'] = experiment(repeats, df, epochs, time_step, batch_size)
# summarize results
print(results.describe())
print(run_time.describe())
