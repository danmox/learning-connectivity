from cnn_results import connectivity_from_CNN, connectivity_from_opt
from hdf5_dataset_utils import cnn_image_parameters, kernelized_config_img, plot_image
from cnn import load_model_for_eval
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.jit
from argparse import ArgumentParser


def checkpoint_to_script(original_filename, script_filename):
    params = cnn_image_parameters()
    model = load_model_for_eval(original_filename)
    script = torch.jit.trace_module(
        model, {"evaluate": torch.zeros(params["img_size"])}
    )
    torch.jit.save(script, script_filename)


def plot(original_filename, script_filename):
    params = cnn_image_parameters()
    model = load_model_for_eval(original_filename)
    model_jit = torch.jit.load(script_filename)

    x_task = np.array([[-70, -70], [70, 70]])
    input_img = kernelized_config_img(x_task, params)

    _, x_cnn, _, _ = connectivity_from_CNN(input_img, model, x_task, params, samples=10)
    _, x_jit, _, _ = connectivity_from_CNN(
        input_img, model_jit, x_task, params, samples=10
    )
    _, x_opt = connectivity_from_opt(x_task, params)

    fig, ax = plt.subplots()
    plot_image(input_img, params, ax)
    ax.plot(x_task[:, 0], x_task[:, 1], "ro", label="task")
    ax.plot(
        x_opt[:, 0], x_opt[:, 1], "rx", label=f"opt ({x_opt.shape[0]})", ms=9, mew=3
    )
    ax.plot(
        x_cnn[:, 0], x_cnn[:, 1], "bx", label=f"CNN ({x_cnn.shape[0]})", ms=9, mew=3
    )
    ax.plot(
        x_jit[:, 0], x_jit[:, 1], "gx", label=f"CNN ({x_jit.shape[0]})", ms=9, mew=3
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert the cnn.py model to a torchscript model. "
        + "The converted model will have a method called evaluate that processes a torch image."
    )
    parser.add_argument("checkpoint", help="The filename of the checkpoint.")
    parser.add_argument(
        "script", help="The filename where the torchscript should be saved. Ex: mid/models/model.ts."
    )
    parser.add_argument(
        "--no-convert", help="Should conversion be run.", action="store_true"
    )
    parser.add_argument(
        "--no-plot", help="Should plots be made for testing.", action="store_true"
    )
    args = parser.parse_args()

    if not args.no_convert:
        checkpoint_to_script(args.checkpoint, args.script)
    if not args.no_plot:
        plot(args.checkpoint, args.script)