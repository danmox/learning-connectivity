import numpy as np
import h5py
from hdf5_dataset_utils import cnn_image_parameters
import matplotlib.pyplot as plt
import matplotlib as mpl
from feasibility import connect_graph

mpl.rcParams['figure.dpi'] = 150
mpl.rcParams.update({'font.size': 16})

mode = 'test'
sample = 3267
dataset = 'data/connectivity_100000s_4t_20201014-012608.hdf5'

params = cnn_image_parameters()
img_bbx = params['img_size'][0] * params['meters_per_pixel'] / 2.0 * np.asarray([-1,1,-1,1])

hdf5_file = h5py.File(dataset, mode='r')

task_image = hdf5_file[mode]['task_img'][sample,...]
comm_image = hdf5_file[mode]['comm_img'][sample,...]
task_config = hdf5_file[mode]['task_config'][sample,...]
mst_config = connect_graph(task_config, params['comm_range'])
comm_config = hdf5_file[mode]['comm_config'][sample,...]

# task image
fig, ax = plt.subplots()
plt.imshow(task_image)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.tight_layout()
# plt.savefig('results/dataset_task_image.png')

# task configuration
fig, ax = plt.subplots()
ax.plot(task_config[:,1], task_config[:,0], 'g.', ms=24, label='task')
ax.axis('scaled')
ax.axis(img_bbx)
ax.set_ylabel('distance (m)')
ax.set_xlabel('distance (m)')
ax.invert_yaxis()
ax.legend()
fig.tight_layout()
plt.savefig('results/dataset_task_config.pdf')

# MST configuration
fig, ax = plt.subplots()
ax.plot(task_config[:,1], task_config[:,0], 'g.', ms=24, label='task')
ax.plot(mst_config[:,1], mst_config[:,0], 'r.', ms=24, label='network')
ax.axis('scaled')
ax.axis(img_bbx)
ax.set_ylabel('distance (m)')
ax.set_xlabel('distance (m)')
ax.invert_yaxis()
ax.legend()
fig.tight_layout()
plt.savefig('results/dataset_mst_config.pdf')

# final configuration
fig, ax = plt.subplots()
ax.plot(task_config[:,1], task_config[:,0], 'g.', ms=24, label='task')
ax.plot(comm_config[:,1], comm_config[:,0], 'r.', ms=24, label='network')
ax.axis('scaled')
ax.axis(img_bbx)
ax.set_ylabel('distance (m)')
ax.set_xlabel('distance (m)')
ax.invert_yaxis()
ax.legend()
fig.tight_layout()
plt.savefig('results/dataset_final_config.pdf')

# comm image
fig, ax = plt.subplots()
plt.imshow(comm_image)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
fig.tight_layout()
plt.savefig('results/dataset_comm_image.png')

plt.show()

hdf5_file.close()
