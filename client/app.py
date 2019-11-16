import logging

from config.spring import create_config_client
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

import grpc
import tensorflow as tf


from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'you_have_to_have_some_secret_for_validation'


class SentimentForm(Form):
    text = StringField('Review Text:', validators=[validators.required()])
    channel = grpc.insecure_channel("localhost:8500")

    @app.route("/", methods=['GET', 'POST'])
    def hello():
        form = SentimentForm(request.form)
        if request.method == 'POST':
            text = request.form['text']
            logging.debug(text)

        if form.validate():
            # Save the comment here.
            positive_prob = SentimentForm.analyze_sentiment(text)
            message = SentimentForm.message_for_sentiment(positive_prob)
            flash('Thanks for the review: {}'.format(message))
        else:
            logging.debug("Something wrong: {}".format(form.errors))
            flash('Something is wrong with the form')

        return render_template('index.jinja2', form=form)

    @staticmethod
    def message_for_sentiment(positive_prob):
        c = create_config_client(profile="training,serving",
                                 app_name="sentiment")
        if positive_prob > c.config['serving']['voucher_threshold']:
            return "it is very useful!"
        else:
            return "please enjoy a discount of 5% for your next purchase"

    @staticmethod
    def analyze_sentiment(text):
        stub = prediction_service_pb2_grpc.PredictionServiceStub(SentimentForm.channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'my_model'
        request.model_spec.signature_name = 'serving_default'
        ###request.model_spec.version = 0 # Can use this for A/B testing
        text_proto = tf.compat.v1.make_tensor_proto(text)
        request.inputs['inputs'].CopyFrom(text_proto)
        response = stub.Predict(request)
        scores = tf.make_ndarray(response.outputs['scores'])
        positive_prob = scores[0][1]
        return positive_prob



if __name__ == "__main__":
    c = create_config_client(profile="serving",
                             app_name="sentiment")
    c.get_config() # initialize once
    app.run()
