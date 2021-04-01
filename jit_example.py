from cnn_results import connectivity_from_CNN, connectivity_from_opt
from hdf5_dataset_utils import cnn_image_parameters, kernelized_config_img, plot_image
from cnn import load_model_for_eval
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.jit

model_file = "models/best.ckpt"
num_agents = 5

params = cnn_image_parameters()
model = load_model_for_eval(model_file)

script = torch.jit.trace_module(model, {"inference": torch.zeros(params["img_size"])})
torch.jit.save(script, "models/best.ts")
model_jit = torch.jit.load("models/best.ts")

x_task = np.array([
    [-70,-70],
    [70,70]
])
input_img = kernelized_config_img(x_task, params)
cnn_conn, x_cnn, _, _ = connectivity_from_CNN(input_img, model, x_task, params, samples=10)
jit_conn, x_jit, _, _ = connectivity_from_CNN(input_img, model_jit, x_task, params, samples=10)
opt_conn, x_opt = connectivity_from_opt(x_task, params)

fig, ax = plt.subplots()
plot_image(input_img, params, ax)
ax.plot(x_task[:,0], x_task[:,1], 'ro', label='task')
ax.plot(x_opt[:,0], x_opt[:,1], 'rx', label=f'opt ({x_opt.shape[0]})', ms=9, mew=3)
ax.plot(x_cnn[:,0], x_cnn[:,1], 'bx', label=f'CNN ({x_cnn.shape[0]})', ms=9, mew=3)
ax.plot(x_jit[:,0], x_jit[:,1], 'gx', label=f'CNN ({x_jit.shape[0]})', ms=9, mew=3)
ax.legend()
plt.show()