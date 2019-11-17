# aiweek19-serving-workshop
Materials for the AI Week 19' workshop - "Serving Deep Learning Models from a Data Science and Engineering Perspective" workshop 

## Installation
You will need Python installed - we used Python 3.6 for testing.

You will also need the required packages installed. Try `pip install -r requirements.txt`

## Usefull Things to run
### Cloud Config
```
docker run -it  --rm \
-p 8888:8888 \
hyness/spring-cloud-config-server \
--spring.cloud.config.server.git.uri=https://github.com/jeniag/aiweek19-serving-workshop.git \
--spring.cloud.config.server.git.searchPaths=config
```


### Running training
`python training/train.py --export-path=<some_export_path>`

### Tensorflow Serving
```
docker run -p 8500:8500 -p 8501:8501  --rm \
-v <some_export_path>:/models/my_model \
-e MODEL_NAME=my_model \
-t tensorflow/serving
```

### Running the client
`python client/app.py`


