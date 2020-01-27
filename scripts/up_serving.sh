#! /bin/bash

python3 -m mot.serving.app &
/usr/bin/tf_serving_entrypoint.sh
