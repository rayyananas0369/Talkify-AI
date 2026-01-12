from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, ZeroPadding3D, TimeDistributed, Flatten, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model

def get_lipnet_model(img_c=3, img_w=100, img_h=50, frames_n=75, output_size=28):
    input_shape = (frames_n, img_h, img_w, img_c)
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')

    x = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(input_data)
    x = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
    x = BatchNormalization(name='batc1')(x)
    x = Activation('relu', name='actv1')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(x)

    x = ZeroPadding3D(padding=(1, 1, 1), name='zero2')(x)
    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='valid', kernel_initializer='he_normal', name='conv2')(x)
    x = BatchNormalization(name='batc2')(x)
    x = Activation('relu', name='actv2')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(x)

    x = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(x)
    x = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), padding='valid', kernel_initializer='he_normal', name='conv3')(x)
    x = BatchNormalization(name='batc3')(x)
    x = Activation('relu', name='actv3')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(x)

    # Reshape for GRU
    x = TimeDistributed(Flatten(), name='time_distributed_1')(x)

    x = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal'), merge_mode='concat', name='bidirectional_1')(x)
    x = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal'), merge_mode='concat', name='bidirectional_2')(x)

    x = Dense(output_size, kernel_initializer='he_normal', name='dense1')(x)
    # Note: CTC decoding will be handled in inference
    
    model = Model(inputs=input_data, outputs=x)
    return model
