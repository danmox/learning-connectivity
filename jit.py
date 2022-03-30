from cnn_results import connectivity_from_CNN, connectivity_from_opt
from hdf5_dataset_utils import cnn_image_parameters, plot_image
from mid import lloyd
from cnn import load_model_for_eval
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.jit
from argparse import ArgumentParser
import json


def checkpoint_to_script(original_filename, script_filename):

    params = cnn_image_parameters()
    model = load_model_for_eval(original_filename)
    script = torch.jit.trace_module(
        model, {"evaluate": torch.zeros(params["img_size"])}
    )
    torch.jit.save(script, script_filename)
    print(f'saved torchscript model to {script_filename}')

    # save the parameters as json
    # we do not want all the parameters since some of them are np arrays
    required_params = (
            "comm_range",
            "img_res",
            "kernel_std",
            "meters_per_pixel",
            "min_area_factor",
    )
    params = {k: params[k] for k in required_params}
    params_filename = script_filename + ".json"
    with open(params_filename, "w") as f:
        json.dump(params, f)
        print(f'saved torchscript model parameters to {params_filename}')


def plot(original_filename, script_filename):
    params = cnn_image_parameters()
    model = load_model_for_eval(original_filename)
    model_jit = torch.jit.load(script_filename)

    x_task = np.array([[-50, -50], [50, 50]])
    input_img = lloyd.kernelized_config_img(x_task, params)

    _, x_cnn, _, cnn_img = connectivity_from_CNN(input_img, model, x_task, params)
    _, x_jit, _, ts_img = connectivity_from_CNN(input_img, model_jit, x_task, params)
    _, x_opt = connectivity_from_opt(x_task, params)

    fig, ax = plt.subplots()
    plot_image(ax, np.maximum(input_img, cnn_img), params=params)
    ax.plot(x_task[:, 0], x_task[:, 1], "ro", label="task")
    ax.plot(
        x_opt[:, 0], x_opt[:, 1], "rx", label=f"opt ({x_opt.shape[0]})", ms=9, mew=3
    )
    ax.plot(
        x_cnn[:, 0], x_cnn[:, 1], "bx", label=f"CNN ({x_cnn.shape[0]})", ms=9, mew=3
    )
    ax.plot(
        x_jit[:, 0], x_jit[:, 1], "gx", label=f"JIT ({x_jit.shape[0]})", ms=9, mew=3
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert a pytorch model to a torchscript model. "
        + "The converted model will have a method called evaluate that processes a torch image."
    )
    parser.add_argument("model", help="The filename of the checkpoint.")
    parser.add_argument("script", help="The filename where the torchscript should be saved.")
    parser.add_argument("--convert", help="Only run the conversion method. Do not plot the results.",
                        action="store_true")
    parser.add_argument("--plot", help="Only run plot the results. Do not run the conversion method.",
                        action="store_true")
    args = parser.parse_args()

    if not args.plot:
        checkpoint_to_script(args.model, args.script)
    if not args.convert:
        plot(args.model, args.script)
