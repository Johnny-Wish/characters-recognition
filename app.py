"""
Flask Serving

This file is a sample flask app that can be used to test your model with an API.

This app does the following:
    - Handles uploads and looks for an image file send as "file" parameter
    - Stores the image at ./images dir
    - Invokes ffwd_to_img function from evaluate.py with this image
    - Returns the output file generated at /output

Additional configuration:
    - You can also choose the checkpoint file name to use as a request parameter
    - Parameter name: checkpoint
    - It is loaded from /input
"""
import os
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from preprocess import Reshape
from pytorch_models.lenet import LeNetBuilder, transformer
from visual import DataPointVisualizer

mapping = "0123456789" + \
          "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + \
          "abcdefghijklmnopqrstuvwxyz"

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app = Flask(__name__)

device = "gpu" if torch.cuda.is_available() else "cpu"
model = LeNetBuilder(1, 62, pretrained_path="/pretrained/LeNet.pth", train_features=False, device=device)()


class ImageLoader:
    def __init__(self, path):
        self.path = path

        with Image.open(self.path) as img:
            data = img.getdata()

        data = np.asarray(data)[:, -1].astype(np.uint8)
        reshape = Reshape(28, 28)
        data = reshape(data)
        data = np.stack([data])
        data = np.stack([data])
        self._data = data
        self._data_tensor = torch.Tensor(self._data)

    @property
    def data(self):
        return self._data

    @property
    def tensor(self):
        return self._data_tensor

    def show(self):
        DataPointVisualizer((self._data[0][0],))


@app.route('/', methods=["POST"])
def classify():
    """
    Take the input image and style transfer it
    """
    # check if the post request has the file part
    input_file = request.files.get('file')
    if not input_file:
        return BadRequest("File not present in request")

    filename = secure_filename(input_file.filename)
    if filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(filename):
        return BadRequest("Invalid file type")

    input_filepath = os.path.join('./dataset/uploaded', filename)
    input_file.save(input_filepath)

    input_tensor = ImageLoader(input_file).tensor
    with torch.no_grad():
        output_tensor = model(input_tensor)[0]
        probs = nn.Softmax()(output_tensor)  # type: torch.Tensor

    sorted_probs, sorted_labels = probs.sort(descending=True)
    sorted_probs = [float(prob) for prob in sorted_probs][:5]
    sorted_labels = [mapping[int(label)] for label in sorted_labels][:5]

    # Get checkpoint filename from la_muse
    return jsonify({"probs": sorted_probs, "labels": sorted_labels}), 200


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    print("starting server ...")
    app.run(host='0.0.0.0')
