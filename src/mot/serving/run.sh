#! /bin/bash

python3 /mot/serving/app.py &
/usr/bin/tf_serving_entrypoint.sh 
