from sys import path
from os.path import dirname, realpath


MY_DIR = dirname(realpath(__file__))
path.append(MY_DIR)
PARENT_DIR = dirname(path[0])
path.append(PARENT_DIR)



import numpy as np
import tensorflow as tf

from tensorflow import keras



def get_and_append_data(relative_filepath):

    data = np.genfromtxt(MY_DIR + "/" + relative_filepath, delimiter=',')

    print(data.shape)
    # fill = np.ones( (data.shape[0],1) )
    # print(fill.shape)
    # data = np.concatenate( [data, fill] , axis=1)

    data = np.insert(data, -1, 1.0, axis=1)
    # data[:, -1] -= .5
    # data[:, -1] *= 2

    return data[ : , :-1], data[ : , -1]


def get_model(input_width, num_hidden_layers, hidden_layer_width, activation, kernel_initializer, bias_initializer):



    model = keras.Sequential()

    model.add( keras.layers.Dense(hidden_layer_width, input_dim=input_width, activation=activation, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer ) )

    for i in range(1, num_hidden_layers):
        model.add(keras.layers.Dense(hidden_layer_width, activation=activation, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer))

    # Output layer: linear because it's the output
    model.add( keras.layers.Dense( 1, activation='linear', bias_initializer=bias_initializer, kernel_initializer=kernel_initializer ) )

    return model


if __name__ == '__main__':


    bank_train_file = "Data/bank-note/train.csv"
    bank_test_file = "Data/bank-note/test.csv"

    training_dataX, training_dataY = get_and_append_data(bank_train_file)
    testing_dataX, testing_dataY = get_and_append_data(bank_test_file)

    print(testing_dataX[:10])
    print(testing_dataY[:10])

    print(tf.version.VERSION);

    model = get_model(input_width=5, num_hidden_layers=5, hidden_layer_width=10, activation='tanh', bias_initializer='glorot_normal', kernel_initializer='glorot_normal')

    model.compile('adam', loss='mean_squared_error', metrics=['accuracy', 'binary_accuracy'])

    model.fit(training_dataX, training_dataY, epochs=25, batch_size=70, verbose=0)

    train_scores = model.evaluate(training_dataX, training_dataY)
    # print("train error: ", train_scores)
    test_scores = model.evaluate(testing_dataX, testing_dataY)
    # print("test error: ", test_scores)

    print(model.predict(testing_dataX[:5]))



    model = get_model(input_width=5, num_hidden_layers=5, hidden_layer_width=10, activation='relu', bias_initializer='he_normal', kernel_initializer='he_normal')

    model.compile('adam', loss='mean_squared_error', metrics=['accuracy', 'binary_accuracy'])

    model.fit(training_dataX, training_dataY, epochs=25, batch_size=70, verbose=0)

    train_scores = model.evaluate(training_dataX, training_dataY)
    # print("train error: ", train_scores)
    test_scores = model.evaluate(testing_dataX, testing_dataY)
    # print("test error: ", test_scores)


    print(model.predict(testing_dataX[:5]))

    results = "{}\t&{:<7.0f}\t&{:<7.0f}\t&{:>10.15f}\t&{:>10.15f}\t\\\\\hline"
    print("method\t&width\t&depth\t&train error\t&test error\t\\\\\hline")
    for method in [ ['tanh','glorot_normal' ], ['relu', 'he_normal'] ]:
        for width in [5, 10, 25, 50, 100]:
            for depth in [3,5,9]:


                model = get_model(input_width=5, num_hidden_layers=depth, hidden_layer_width=width, activation=method[0], bias_initializer=method[1], kernel_initializer=method[1])

                model.compile('adam', loss='mean_squared_error', metrics=['accuracy', 'binary_accuracy'])

                model.fit(training_dataX, training_dataY, epochs=25, batch_size=70, verbose=0)

                train_scores = model.evaluate(training_dataX, training_dataY, verbose=0)
                # print("train error: ", train_scores)
                test_scores = model.evaluate(testing_dataX, testing_dataY, verbose=0)

                print(results.format(method[0], width, depth, 1-train_scores[2], 1-test_scores[2]))


