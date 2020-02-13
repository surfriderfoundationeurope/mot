#!/bin/bash

model_url=http://files.heuritech.com/raw_files/surfrider/serving.zip


if [ -a serving ]; then
    rm -r serving
fi

if [ $MODEL_FOLDER ]; then
    if [ -d $MODEL_FOLDER ]; then
        echo "Copying $MODEL_FOLDER to ./serving"
        cp -r $MODEL_FOLDER serving
    else
        echo "$MODEL_FOLDER isn't a valid folder. Aborting launch of serving." 1>&2
        exit 64
    fi
else
    echo "No MODEL_FOLDER specified. Downloading $model_url"

    # Removing files and folders that may induce naming conflicts
    if [ -a serving.zip ]; then
        rm -r serving.zip
    fi
    if [ -a tmp ]; then
        rm -r tmp
    fi

    wget $model_url
    unzip serving.zip -d tmp
    mv $(find tmp/ -maxdepth 1 -mindepth 1 -type d) serving
    rm -r tmp
    rm -r serving.zip
fi
