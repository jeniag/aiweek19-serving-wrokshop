import argparse
import logging
import os
import re

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from config.spring import create_config_client


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                         "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        "aclImdb", "test"))

    return train_df, test_df


def serving_input_receiver_fn():
    sentence = tf.compat.v1.placeholder(dtype=tf.string, shape=[None], name='sentence')

    features = {"sentence": sentence}

    return tf.estimator.export.ServingInputReceiver(features=features, receiver_tensors=sentence)


def main(args):
    config_client = create_config_client(profile="training",
                             app_name="sentiment")
    config_client.get_config()

    train_df, test_df = download_and_load_datasets()
    logging.info(train_df.head())

    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["sentiment"].astype(float), num_epochs=None, shuffle=True)

    # Prediction on the whole training set.
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["sentiment"].astype(float), shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df["sentiment"].astype(float), shuffle=False)

    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    estimator = tf.estimator.DNNRegressor(
        hidden_units=config_client.config["training"]["hidden_units"],
        feature_columns=[embedded_text_feature_column],
        optimizer=tf.train.AdagradOptimizer(learning_rate=config_client.config["training"]["learning_rate"]))

    estimator.train(input_fn=train_input_fn, steps=1000)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    export_input_fn = serving_input_receiver_fn
    estimator.export_savedmodel(args.export_path, export_input_fn, as_text=True)

    logging.info(train_eval_result)
    logging.info(test_eval_result)


if __name__ == '__main__':
    # Reduce logging output.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.basicConfig(level=logging.DEBUG)
    default_path = os.path.abspath('{}/../client/model'.format(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-e', '--export_path', dest='export_path', type=str, default=default_path,
                        help='Output path for the model')
    parsed_args = parser.parse_args()
    main(parsed_args)


# docker pull tensorflow/serving
# docker run -p 8500:8500 -p 8501:8501  --mount type=bind,source=/Users/Charles.s/exports/,target=/models/my_model \ -e MODEL_NAME=my_model -t tensorflow/serving