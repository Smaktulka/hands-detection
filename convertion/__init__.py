import numpy
import torch

import blaze
from blaze.model import BlazeModel
from convertion.tflite import SaveableModel
import tensorflow as tf


def convert_pytorch_to_tflite(torch_model_path, tflite_model_path):
    from ai_edge_torch import convert
    print("THIS CODE SHOULD ONLY BE RUN IN Google Colab")
    print("Check this link: https://ai.google.dev/edge/litert/models/pytorch_to_tflite")
    model = BlazeModel()
    model.load_anchors(blaze.ANCHORS_PATH)
    model.load_state_dict(torch.load(torch_model_path, map_location=torch.device('cpu')))

    sample_inputs = (torch.randn(1, 3, 128, 128),)

    torch_output = model(*sample_inputs)
    edge_model = convert(model.eval(), sample_inputs)

    edge_output = edge_model(*sample_inputs)

    if (numpy.allclose(
            torch_output.detach().numpy(),
            edge_output,
            atol=1e-5,
            rtol=1e-5,
    )):
        print("Inference result with Pytorch and TfLite was within tolerance")
    else:
        print("Something wrong with Pytorch --> TfLite")

    edge_model.export(tflite_model_path)


def convert_tf_to_tflite(model: SaveableModel, tflite_model_path):
    inputs = model.get_model_input()

    output = model(iter(inputs))

    keras_model = tf.keras.Model(inputs=inputs, outputs=output)

    keras_model.save('out.keras', include_optimizer=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    post_interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

    post_interpreter.allocate_tensors()

    post_input_details = post_interpreter.get_input_details()
    post_output_details = post_interpreter.get_output_details()

    print("Post Input details:", post_input_details)
    print("Post Output details:", post_output_details)
