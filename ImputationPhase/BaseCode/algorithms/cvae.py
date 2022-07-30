import gc

from tensorflow.keras.layers import Lambda, Input, Dense, Dropout, Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from dotmap import DotMap
import numpy as np

DEFAULT_CVAE_CONFIGURATIONS = {
    "convolutional_number_hidden_layers": 2,
    "convolutional_filters": 32,
    "convolutional_kernel_size": 3,
    "dense_hidden_layers_encoder": [128],
    "dense_hidden_layers_decoder": [],
    "latent_dim": 32,
    "hidden_layers_activation": "relu",
    "output_layer_activation": "sigmoid",
    "batch_size": 64,
    "epochs": 200,
    "learning_rate": 0.001,
    "dropout_rate": 0.0,
    "l2_lambda": 0.0,
    "weights_initializer": "glorot_normal",
    "reduce_learning_rate_factor": 1.0,
    "reduce_learning_rate_patience": 10,
    "early_stopping_patience": 0,
    "reconstruction_missing_values_weight": 1,
    "kullback_leibler_weight": 1,
    "validation_size": 0.1428,
    "verbose": 1
}

class CVAE:

    def __init__(self, custom_configurations, dataset_complete, dataset_pre_imputed, masks):

        configurations = custom_configurations.copy()
        temp_configurations = configurations.copy()
        configurations.update(DEFAULT_CVAE_CONFIGURATIONS)
        configurations.update(temp_configurations)
        self.configs = DotMap(configurations)

        self.dataset_complete = dataset_complete.copy()
        self.dataset_pre_imputed = dataset_pre_imputed.copy()
        self.masks = masks.copy()

        self.vae = None

    def _sampling(self, args):

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def create_and_train(self):

        x_train, x_val = train_test_split(self.dataset_complete,
                                          test_size=self.configs.validation_size, shuffle=False)
        x_train_noisy, x_val_noisy = train_test_split(self.dataset_pre_imputed,
                                                      test_size=self.configs.validation_size, shuffle=False)
        masks_train, masks_val = train_test_split(self.masks, test_size=self.configs.validation_size, shuffle=False)

        image_width = x_train.shape[1]
        image_height = x_train.shape[2]
        number_channels = x_train.shape[3]

        x_train = np.reshape(x_train, [-1, image_width, image_height, number_channels])
        x_val = np.reshape(x_val, [-1, image_width, image_height, number_channels])
        x_train_noisy = np.reshape(x_train_noisy, [-1, image_width, image_height, number_channels])
        x_val_noisy = np.reshape(x_val_noisy, [-1, image_width, image_height, number_channels])

        input_shape = (image_width, image_height, number_channels)

        inputs = Input(shape=input_shape, name='encoder_input')
        masks = Input(shape=input_shape, name='masks')

        x = inputs

        if self.configs.convolutional_number_hidden_layers > 0:

            filters = self.configs.convolutional_filters / 2

            for i in range(self.configs.convolutional_number_hidden_layers):
                filters *= 2
                x = Conv2D(filters=round(filters),
                           kernel_size=self.configs.convolutional_kernel_size,
                           activation=self.configs.hidden_layers_activation,
                           strides=2,  # 2 instead of 1 to avoid using MaxPooling2D
                           padding='same',
                           kernel_regularizer=l2(self.configs.l2_lambda),
                           kernel_initializer=self.configs.weights_initializer)(x)

        shape = K.int_shape(x)

        x = Flatten()(x)

        for hidden_dim in self.configs.dense_hidden_layers_encoder:
            hidden_dim = hidden_dim if hidden_dim >= 2 else 2
            x = Dense(hidden_dim,
                      activation=self.configs.hidden_layers_activation,
                      kernel_regularizer=l2(self.configs.l2_lambda),
                      kernel_initializer=self.configs.weights_initializer)(x)
            if self.configs.dropout_rate > 0:
                x = Dropout(self.configs.dropout_rate)(x)

        z_mean = Dense(self.configs.latent_dim, kernel_initializer=self.configs.weights_initializer, name='z_mean')(x)
        z_log_var = Dense(self.configs.latent_dim, kernel_initializer=self.configs.weights_initializer, name='z_log_var')(x)

        z = Lambda(self._sampling, output_shape=(self.configs.latent_dim,), name='z')([z_mean, z_log_var])

        encoder = Model([inputs, masks], [z_mean, z_log_var, z], name='encoder')

        latent_inputs = Input(shape=(self.configs.latent_dim,), name='z_sampling')
        x = latent_inputs

        for hidden_dim in self.configs.dense_hidden_layers_decoder:
            x = Dense(hidden_dim,
                      activation=self.configs.hidden_layers_activation,
                      kernel_regularizer=l2(self.configs.l2_lambda),
                      kernel_initializer=self.configs.weights_initializer)(x)
            if self.configs.dropout_rate > 0:
                x = Dropout(self.configs.dropout_rate)(x)

        if self.configs.convolutional_number_hidden_layers > 0:

            x = Dense(shape[1] * shape[2] * shape[3],
                      activation=self.configs.hidden_layers_activation,
                      kernel_regularizer=l2(self.configs.l2_lambda),
                      kernel_initializer=self.configs.weights_initializer)(x)

            x = Reshape((shape[1], shape[2], shape[3]))(x)

            for i in range(self.configs.convolutional_number_hidden_layers):
                filters /= 2
                x = Conv2DTranspose(filters=round(filters),
                                    kernel_size=self.configs.convolutional_kernel_size,
                                    activation=self.configs.hidden_layers_activation,
                                    strides=2,
                                    padding='same',
                                    kernel_regularizer=l2(self.configs.l2_lambda),
                                    kernel_initializer=self.configs.weights_initializer)(x)

            outputs = Conv2DTranspose(filters=number_channels,
                                      kernel_size=self.configs.convolutional_kernel_size,
                                      activation=self.configs.output_layer_activation,
                                      padding='same',
                                      name='decoder_output',
                                      kernel_regularizer=l2(self.configs.l2_lambda),
                                      kernel_initializer=self.configs.weights_initializer)(x)

        else:

            x = Dense(shape[1] * shape[2] * shape[3],
                      activation=self.configs.output_layer_activation,
                      kernel_regularizer=l2(self.configs.l2_lambda),
                      kernel_initializer=self.configs.weights_initializer)(x)

            x = Reshape((shape[1], shape[2], shape[3]))(x)

            outputs = x

        decoder = Model(latent_inputs, outputs, name='decoder')

        outputs = decoder(encoder([inputs, masks])[2])

        self.vae = Model([inputs, masks], outputs, name='vae')

        def custom_vae_loss_func(y_true, y_pred):
            bce_loss_mv = binary_crossentropy(K.flatten(y_true * masks), K.flatten(y_pred * masks)) * image_width * image_height
            bce_loss_ov = binary_crossentropy(K.flatten(y_true * (masks * -1 + 1)), K.flatten(y_pred * (masks * -1 + 1))) * image_width * image_height
            bce_loss = bce_loss_ov + self.configs.reconstruction_missing_values_weight * bce_loss_mv
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            vae_loss = K.mean(bce_loss + self.configs.kullback_leibler_weight * kl_loss)
            return vae_loss

        def vae_loss_func(y_true, y_pred):
            bce_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred)) * image_width * image_height
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            vae_loss = K.mean(bce_loss + self.configs.kullback_leibler_weight * kl_loss)
            return vae_loss

        def VAL_MAE(y_true, y_pred):
            return K.mean(K.abs(y_pred - y_true), axis=-1)

        def VAL_MASK_MAE(y_true, y_pred):
            return K.sum(K.abs(masks * y_pred - masks * y_true)) / K.sum(masks)

        if self.configs.reconstruction_missing_values_weight == 1:
            loss_func = vae_loss_func
        else:
            loss_func = custom_vae_loss_func

        self.vae.compile(optimizer=Adam(self.configs.learning_rate), loss=loss_func,
                         metrics=[VAL_MASK_MAE, VAL_MAE])

        all_callbacks = []

        if self.configs.reduce_learning_rate_factor < 1:
            all_callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                 factor=self.configs.reduce_learning_rate_factor,
                                 patience=self.configs.reduce_learning_rate_patience, min_lr=0))

        if self.configs.early_stopping_patience > 0:
            all_callbacks.append(EarlyStopping(monitor='val_loss', mode='min', verbose=0,
                                 patience=self.configs.early_stopping_patience))

        gc.collect()
        self.vae.fit([x_train_noisy, masks_train], x_train,
                     epochs=self.configs.epochs,
                     batch_size=self.configs.batch_size,
                     validation_data=([x_val_noisy, masks_val], x_val),
                     callbacks=all_callbacks,
                     verbose=self.configs.verbose)

        return self.vae

    def predict(self, external_input=None):

        if external_input is None:
            return self.vae.predict([self.dataset_pre_imputed, self.masks])

        return self.vae.predict(external_input)