import argparse
import joblib
import numpy as np
np.random.seed(42)
import keras.backend as K
from model import build_DUA_3
import tensorflow as tf
tf.set_random_seed(42)

def evaluate_recall(y, k=1):
    num_examples = float(len(y))
    num_correct = 0

    for predictions in y:
        if 0 in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples

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
    psr.add_argument('--test_data', default='test.joblib')
    psr.add_argument('--model_name', default='DUA_K.dev')
    psr.add_argument('--embedding_matrix', default='embedding_matrix.joblib')
    args = psr.parse_args()

    print('load data')
    test_data = joblib.load(args.test_data)

    print('load embedding matrix')
    embedding_matrix = joblib.load(args.embedding_matrix)

    # json_string = open(args.model_name + '.json').read()
    # model = model_from_json(json_string)

    model = build_DUA_3(max_turn=10, maxlen=50, word_dim=200, sent_dim=200, session_hidden_size=50,
              num_words=282132, embedding_matrix=embedding_matrix)
    model.load_weights(args.model_name + '.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    context = np.array(test_data['context'])
    response = np.array(test_data['response'])

    print('predict')
    y = model.predict([context, response], verbose=1, batch_size=50)
    y = np.array(y).reshape(50000, 10)
    y = [np.argsort(y[i], axis=0)[::-1] for i in range(len(y))]
    for n in [1, 2, 5]:
        print('Recall @ ({}, 10): {:g}'.format(n, evaluate_recall(y, n)))

if __name__ == '__main__': main()
