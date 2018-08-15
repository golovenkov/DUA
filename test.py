import argparse
import joblib
import numpy as np
from keras.models import model_from_json
from model import build_DUA_2

def evaluate_recall(y, k=1):
    num_examples = float(len(y))
    num_correct = 0

    for predictions in y:
        if 0 in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument('--test_data', default='test.joblib')
    psr.add_argument('--model_name', default='SMN_last')
    args = psr.parse_args()

    print('load data')
    test_data = joblib.load(args.test_data)

    # json_string = open(args.model_name + '.json').read()
    # model = model_from_json(json_string)

    model = build_DUA_2(max_turn=10, maxlen=50, word_dim=200, sent_dim=200, session_hidden_size=50,
              num_words=50000, embedding_matrix=None)
    model.load_weights(args.model_name + '.h5')

    context = np.array(test_data['context'])
    response = np.array(test_data['response'])

    print('predict')
    y = model.predict([context, response], batch_size=50)
    y = np.array(y).reshape(50000, 10)
    y = [np.argsort(y[i], axis=0)[::-1] for i in range(len(y))]
    for n in [1, 2, 5]:
        print('Recall @ ({}, 10): {:g}'.format(n, evaluate_recall(y, n)))

if __name__ == '__main__': main()
