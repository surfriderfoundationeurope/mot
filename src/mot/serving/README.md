# Serving

The serving is the interface to peform inferences on images (object detection) or on videos (object tracking). 

## Launch

Once you have a SavedModel on disk, you can launch the serving with:

```bash
NVIDIA_VISIBLE_DEVICES=2 MODEL_FOLDER=/path/to/serving PORT=the_port_you_want_to_expose make docker-serving
```

`NVIDIA_VISIBLE_DEVICES`allows you to specify the GPU you want to use for inferences.
For `MODEL_FOLDER`, you specify the path to the folder where the `saved_model.pb` file and `variables` folder stored.
The `PORT` is the one you'll use to make inference requests.


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
