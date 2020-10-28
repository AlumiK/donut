import os
import donut
import numpy as np
import tensorflow as tf

from testing import load_kpi, adjust_scores, ignore_missing, best_f1score


def main():
    # read the raw data
    timestamp, values, labels = load_kpi(INPUT)
    file = os.path.basename(INPUT)

    # complete the timestamp, and obtain the missing point indicators
    timestamp, missing, (values, labels) = donut.preprocessing.complete_timestamp(timestamp, (values, labels))

    # split the training and testing data
    test_portion = 0.3
    test_n = int(len(values) * test_portion)
    train_values, test_values = values[:-test_n], values[-test_n:]
    train_labels, test_labels = labels[:-test_n], labels[-test_n:]
    train_missing, test_missing = missing[:-test_n], missing[-test_n:]

    # standardize the training and testing data
    train_values, mean, std = donut.preprocessing.standardize_kpi(train_values,
                                                                  excludes=np.logical_or(train_labels, train_missing))
    test_values, _, _ = donut.preprocessing.standardize_kpi(test_values, mean=mean, std=std)

    # we build the entire model within the scope of `model_vs`,
    # it should hold exactly all the variables of `model`, including
    # the variables created by Keras layers
    with tf.variable_scope('model') as model_vs:
        model = donut.Donut(
            h_for_p_x=tf.keras.Sequential([
                tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
                tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
            ]),
            h_for_q_z=tf.keras.Sequential([
                tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
                tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
            ]),
            x_dims=120,
            z_dims=8,
        )

    trainer = donut.DonutTrainer(model=model, model_vs=model_vs, max_epoch=EPOCHS)
    predictor = donut.DonutPredictor(model)

    with tf.Session().as_default():
        trainer.fit(train_values, train_labels, train_missing, mean, std)
        test_scores = -predictor.get_score(test_values, test_missing)

    adjusted_scores = adjust_scores(test_scores, test_labels[119:])
    adjusted_scores, adjusted_labels = ignore_missing([adjusted_scores, test_labels[119:]], missing=test_missing[119:])
    threshold, precision, recall, f1score = best_f1score(labels=adjusted_labels, scores=adjusted_scores)

    print(f'file: {file}\n'
          f'threshold: {threshold}\n'
          f'precision: {precision:.3f}\n'
          f'recall: {recall:.3f}\n'
          f'f1score: {f1score:.3f}\n')

    os.makedirs(OUTPUT, exist_ok=True)
    with open(f'{os.path.join(OUTPUT, os.path.splitext(file)[0])}.txt', 'w') as output:
        output.write(f'file={file}\n\n'
                     f'threshold={threshold}\n'
                     f'precision={precision:.3f}\n'
                     f'recall={recall:.3f}\n'
                     f'f1_score={f1score:.3f}\n')


if __name__ == '__main__':
    INPUT = os.path.join('data', 'cpu4.csv')
    OUTPUT = 'out'
    EPOCHS = 50
    main()
