import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.optimizers import Adam
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from models import (
    get_generator,
    get_discriminator,
    get_generator_containing_discriminator,
    GeneratorLoss
)

DATA_DIR = 'data/cmu_arctic_vc2/'
ids_train = [os.path.basename(p) for p in glob.glob(DATA_DIR + 'X/*')]
ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=114514)

batch_size = 50

input_static_dim = 59

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))
import random


def train_generator():
    random.shuffle(ids_train_split)
    for start in tqdm(range(0, len(ids_train_split), batch_size)):
        x_batch = []
        y_batch = []
        end = min(start + batch_size, len(ids_train_split))
        ids_train_batch = ids_train_split[start:end]
        for id in ids_train_batch:
            from_data_path = DATA_DIR + 'X/{id}'.format(id=id)
            to_data_path = DATA_DIR + 'Y/{id}'.format(id=id)

            from_data = np.load(from_data_path)
            to_data_static = np.load(to_data_path)[:, :input_static_dim]

            x_batch.append(from_data)
            y_batch.append(to_data_static)
        x_batch = np.vstack(x_batch)
        y_batch = np.vstack(y_batch)

        yield x_batch, y_batch


def valid_generator():
    for start in range(0, len(ids_valid_split), batch_size):
        x_batch = []
        y_batch = []
        end = min(start + batch_size, len(ids_train_split))
        ids_train_batch = ids_train_split[start:end]
        for id in ids_train_batch:
            from_data_path = DATA_DIR + 'X/{id}'.format(id=id)
            to_data_path = DATA_DIR + 'Y/{id}'.format(id=id)

            from_data = np.load(from_data_path)
            to_data_static = np.load(to_data_path)[:, :input_static_dim]

            x_batch.append(from_data)
            y_batch.append(to_data_static)
        x_batch = np.vstack(x_batch)
        y_batch = np.vstack(y_batch)
        yield x_batch, y_batch


def main():
    generator = get_generator()
    discriminator = get_discriminator()
    gan = get_generator_containing_discriminator(generator, discriminator)

    model_path = 'weights_0219/generator_16600.hdf5'
    generator.load_weights(filepath=model_path)

    model_path = 'weights_0219/discriminator_16600.hdf5'
    discriminator.load_weights(filepath=model_path)

    opt_d = Adam(lr=1e-4, beta_1=0.5)
    discriminator.compile(optimizer=opt_d,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

    generator_loss = GeneratorLoss()
    opt_g = Adam(lr=2e-4, beta_1=0.5)
    gan.compile(optimizer=opt_g,
                loss=generator_loss)

    list_all_train_metric = []
    list_all_valid_metric = []

    for epoch in range(100000):
        print('epoch:', epoch)
        list_train_metric = []
        for x_batch, y_batch in train_generator():

            gen_data = generator.predict(x_batch)

            X = np.append(y_batch, gen_data, axis=0)
            y = np.array([1] * y_batch.shape[0] + [0] * gen_data.shape[0])
            loss_d, acc_d = discriminator.train_on_batch(X, y)

            X = x_batch
            y = np.hstack([np.ones((y_batch.shape[0], 1)), y_batch])

            loss_g = gan.train_on_batch(X, y)
            loss_g_mge = generator_loss.mge_loss
            loss_g_adv = generator_loss.adv_loss
            list_train_metric.append([loss_d, acc_d, loss_g, loss_g_adv, loss_g_mge])
        mge_adv_loss_weight = generator_loss.mge_adv_loss_weight
        train_metric = np.mean(list_train_metric, axis=0)
        list_all_train_metric.append(train_metric)
        pd.DataFrame(list_all_train_metric).to_csv('logs/train_metric.csv')
        print('train loss:', train_metric)
        print('mge_adv_loss_weight:', mge_adv_loss_weight, train_metric[4] / train_metric[3])

        list_valid_metric = []
        for x_batch, y_batch in valid_generator():
            generated = generator.predict(x_batch)
            loss_g_mge = mean_squared_error(y_batch, generated)

            X = np.append(y_batch, generated, axis=0)
            y = np.array([0] * len(x_batch) + [1] * len(generated))
            pred = discriminator.predict(X)
            loss_d = log_loss(y, pred)
            acc_d = accuracy_score(y, pred > 0.5)
            roc_d = roc_auc_score(y, pred)

            list_valid_metric.append([loss_d, acc_d, roc_d, loss_g_mge])

        valid_metric = np.mean(list_valid_metric, axis=0)
        list_all_valid_metric.append(valid_metric)
        pd.DataFrame(list_all_valid_metric).to_csv('logs/valid_metric.csv')

        print('valid loss: ', valid_metric)
        print('==============')
        if epoch % 100 == 0:
            print('save model')
            generator.save_weights('weights/generator_{}.hdf5'.format(epoch), True)
            discriminator.save_weights('weights/discriminator_{}.hdf5'.format(epoch), True)


if __name__ == '__main__':
    main()
