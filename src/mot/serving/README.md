# Serving

The serving is the interface to peform inferences on images (object detection) or on videos (object tracking).

## SavedModel

For this step, you need a SavedModel on disk. You can either
- export one of your models following the instructions in the principal README
- use this [pretrained model](http://files.heuritech.com/raw_files/surfrider/serving.zip). You need to unzip it, this will be your `MODEL_FOLDER` in the next step.

## Launch

```bash
NVIDIA_VISIBLE_DEVICES=2 RATIO_GPU=0.3 MODEL_FOLDER=/path/to/serving PORT=the_port_you_want_to_expose make docker-serving
```

- `NVIDIA_VISIBLE_DEVICES` allows you to specify the GPU you want to use for inferences.
- `RATIO_GPU` is used to specify which fraction of your GPU tou want to allow to your model. By default, it is set at 0.45.
- For `MODEL_FOLDER`, you have to specify the path to the folder where the `saved_model.pb` file and `variables` folder are stored. If you don't specify a `MODEL_FOLDER`, [this one](http://files.heuritech.com/raw_files/surfrider/serving.zip) will be automatically downloaded and used.
- The `PORT` is the one you'll use to make inference requests.


## Requests

Here are the different ways to perform inference requests.

### Web interface

You can access a basic web interface to manually upload pictures or videos to do inference.
In your browser, access the address `host:port`, with port being the one you specified in the previous step.

### cURL

#### Json

This only works for images.

```bash
curl -d @/path/to/json --header "Content-Type: application/json" host:port
```

#### File

```bash
curl -F "file=@/path/to/file" host:port
```

For videos you can add parameters such as fps and resolution like that:

```bash
curl -F "file=@/path/to/video.mp4" -F "fps=2" -F "resolution=(10,10)" host:port
```

You don't have to specify those parameters and you can find their default value in [this file](inference.py).
