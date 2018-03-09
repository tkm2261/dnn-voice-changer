from keras.models import Model, Sequential
from keras.losses import mean_squared_error, categorical_crossentropy

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Dropout, multiply, add, Lambda
from keras.layers.merge import concatenate

import keras.backend as K


class UpdateModel(Model):
    @property
    def updates(self):
        updates = super().updates
        if hasattr(self, 'loss_functions'):
            for loss_func in self.loss_functions:
                if hasattr(loss_func, 'updates'):
                    updates += loss_func.updates
        return updates


def get_generator(input_static_dim=59,
                  input_dynamic_dim=118,
                  num_hidden=5,
                  hidden_dim=512,
                  dropout_rate=0.5):

    inputs_all = x = Input(shape=(input_static_dim + input_dynamic_dim,))
    x_static = Lambda(lambda x: x[:, :input_static_dim])(inputs_all)

    t_x = Dense(input_static_dim, activation='sigmoid')(x_static)

    for _ in range(num_hidden):
        act = LeakyReLU()
        x = Dense(hidden_dim, activation=act)(x)
        x = Dropout(dropout_rate)(x)

    g_x = Dense(input_static_dim)(x)

    outputs = add([x_static, multiply([t_x, g_x])])

    model = Model(inputs=inputs_all, outputs=outputs, name='generator')

    return model


def get_discriminator(input_dim=59,
                      output_dim=1,
                      num_hidden=2,
                      hidden_dim=256,
                      dropout_rate=0.5,
                      last_activation='sigmoid'):

    inputs = x = Input(shape=(input_dim,))

    for _ in range(num_hidden):
        act = LeakyReLU()
        x = Dense(hidden_dim, activation=act)(x)
        x = Dropout(dropout_rate)(x)

    x = Dense(output_dim, activation=last_activation)(x)

    outputs = x

    model = Model(inputs=inputs, outputs=outputs, name='discriminator')

    return model


def get_generator_containing_discriminator(generator, discriminator):
    input_static_dim = 59
    input_dynamic_dim = 118

    inputs_all = Input(shape=(input_static_dim + input_dynamic_dim,))

    gen_data = generator(inputs_all)

    discriminator.trainable = False
    pred_label = discriminator(gen_data)

    outputs = concatenate([pred_label, gen_data])
    model = UpdateModel(inputs=inputs_all, outputs=outputs, name='gan')

    return model


class GeneratorLoss:
    __name__ = 'generator_loss'

    def __init__(self):
        self._mge_adv_loss_weight = K.variable(0, name='mge_adv_loss_weight')
        self._mge_loss = K.variable(0, name='mge_loss')  # for observation
        self._adv_loss = K.variable(0, name='adv_loss')    # for observation
        self.updates = []

    def __call__(self, y_true, y_pred):
        y_true_label, y_true_data = y_true[:, 0], y_true[:, 1:]
        y_pred_label, y_pred_data = y_pred[:, 0], y_pred[:, 1:]

        mge_loss = K.mean(K.square(y_true_data - y_pred_data))
        self.updates.append(K.update(self._mge_loss, mge_loss))

        adv_loss = K.mean(- K.log(y_pred_label))
        self.updates.append(K.update(self._adv_loss, adv_loss))

        self.updates.append(K.update(self._mge_adv_loss_weight, mge_loss / (adv_loss + 1.0e-8)))

        ret = mge_loss + 0.01 * self._mge_adv_loss_weight * adv_loss
        return ret

    @property
    def mge_loss(self):
        return K.get_value(self._mge_loss)

    @property
    def adv_loss(self):
        return K.get_value(self._adv_loss)

    @property
    def mge_adv_loss_weight(self):
        return K.get_value(self._mge_adv_loss_weight)
