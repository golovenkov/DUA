import argparse
import numpy as np
np.random.seed(42)
import joblib
from keras.layers import Dense, Input, Flatten, dot, concatenate, Reshape, Lambda, Concatenate, Multiply, Activation, Add
from keras.layers import Conv2D, MaxPooling2D, Embedding, GRU
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, model_from_json

from keras.activations import softmax
from keras import backend as K
import tensorflow as tf
from keras import initializers
from model import build_DUA_2, build_DUA_3
import tensorflow as tf
tf.set_random_seed(42)


def main():
    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3

    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))


    psr = argparse.ArgumentParser()
    psr.add_argument('--maxlen', default=50, type=int)
    psr.add_argument('--max_turn', default=10, type=int)
    psr.add_argument('--num_words', default=50000, type=int)
    psr.add_argument('--word_dim', default=200, type=int)
    psr.add_argument('--sent_dim', default=200, type=int)
    psr.add_argument('--session_hidden_size', default=50, type=int)
    psr.add_argument('--embedding_matrix', default='embedding_matrix.joblib')
    psr.add_argument('--train_data', default='train.joblib')
    psr.add_argument('--valid_data', default='valid.joblib')
    psr.add_argument('--model_name', default='DUA_K.dev')
    psr.add_argument('--batch_size', default=50, type=int)
    args = psr.parse_args()

    print('load embedding matrix')
    embedding_matrix = joblib.load(args.embedding_matrix)

    print('build model')
    model = build_DUA_3(args.max_turn, args.maxlen, args.word_dim, args.sent_dim, args.session_hidden_size, args.num_words, embedding_matrix)

    # json_string = model.to_json()
    # open(args.model_name + '.json', 'w').write(json_string)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    model_checkpoint = ModelCheckpoint(args.model_name + '.h5', save_best_only=True, save_weights_only=True)

    print('load train data')
    train_data = joblib.load(args.train_data)
    context = np.array(train_data['context'])
    response = np.array(train_data['response'])
    labels = train_data['labels']

    print('loda valid data')
    valid_data = joblib.load(args.valid_data)
    valid_context = np.array(valid_data['context'])
    valid_response = np.array(valid_data['response'])
    valid_labels = valid_data['labels']

    print('fitting')
    model.fit(
        [context, response],
        labels,
        validation_data=([valid_context, valid_response], valid_labels),
        batch_size=args.batch_size,
        epochs=10,
        callbacks=[early_stopping, model_checkpoint]
    )

if __name__ == '__main__': main()
