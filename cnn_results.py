from pathlib import Path
from hdf5_dataset_utils import cnn_image_parameters, plot_image
from math import ceil
from cnn import load_model_for_eval, get_file_name
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import numpy as np
from connectivity_maximization import circle_points
from mid.connectivity_planner.src.connectivity_planner.connectivity_optimization import ConnectivityOpt as ConnOpt, round_sf
from mid.connectivity_planner.src.connectivity_planner.channel_model import PiecewisePathLossModel
from mid.connectivity_planner.src.connectivity_planner.feasibility import connect_graph, adaptive_bbx, min_feasible_sample
from mid.connectivity_planner.src.connectivity_planner import lloyd

import torch
from hdf5_dataset_utils import ConnectivityDataset
import h5py
import os
import time
import datetime


def scale_from_filename(filename):
    parts = filename.split('_')
    if '256' in parts:
        return 2
    return 1

def compute_peaks(image, threshold_val=80, blur_sigma=1, region_size=7, view=False):
    out_peaks, blurred_img = lloyd.compute_peaks(image, threshold_val, blur_sigma, region_size)
    if view:
        fig, ax = plt.subplots()
        ax.plot(out_peaks[:,0], out_peaks[:,1], 'ro')
        ax.imshow(blurred_img.T)
        ax.invert_yaxis()
        plt.show()
    return out_peaks


def connectivity_from_CNN(input_image, model, x_task, params, samples=1, viz=False, variable_power=True):

    conn = np.zeros((samples,))
    power = np.zeros((samples,))
    cnn_imgs = np.zeros((samples,) + input_image.shape, dtype=input_image.dtype)
    agents = np.zeros((samples,), dtype=int)
    x_comm = []
    for i in range(samples):

        # run inference and extract the network team configuration

        cnn_imgs[i] = model.evaluate(torch.from_numpy(input_image)).cpu().detach().numpy()
        x_comm += [compute_coverage(cnn_imgs[i], params, viz=viz)]
        agents[i] = x_comm[i].shape[0]

        # find connectivity
        conn[i] = ConnOpt.connectivity(params['channel_model'], x_task, x_comm[i])

        # increase transmit power until the network is connected
        power[i] = params['channel_model'].t
        while variable_power and conn[i] < 5e-4:
            power[i] += 0.2
            cm = PiecewisePathLossModel(print_values=False, l0=power[i])
            conn[i] = ConnOpt.connectivity(cm, x_task, x_comm[i])

    # return the best result
    # prioritize: lowest power > fewer agents > highest connectivity
    mask = power == np.min(power)
    mask &= agents == np.min(agents[mask])
    best_idx = np.where(conn == np.max(conn[mask]))[0][0]

    return conn[best_idx], x_comm[best_idx], power[best_idx], cnn_imgs[best_idx]


def connectivity_from_opt(task_config, p, viz=False):
    comm_config = connect_graph(task_config, p['comm_range'])
    opt = ConnOpt(p['channel_model'], task_config, comm_config)
    conn = opt.maximize_connectivity(viz=viz)
    return conn, opt.get_comm_config()


def compute_coverage(image, params, viz=False):
    """compute coverage using Lloyd's Algorithm"""

    # extract peaks of intensity image
    config_subs = compute_peaks(image, threshold_val=60, view=viz)
    config = lloyd.sub_to_pos(params['meters_per_pixel'], params['img_size'][0], config_subs)
    peaks = np.copy(config)

    # Lloyd's algorithm
    coverage_range = params['coverage_range']
    it = 1
    while True:
        voronoi_cells = lloyd.compute_voronoi(config, params['bbx'])
        new_config = lloyd.lloyd_step(image, params['xy'], config, voronoi_cells, coverage_range)
        update_dist = np.sum(np.linalg.norm(config - new_config, axis=1))

        cell_patches = []
        circle_patches = []
        if viz:
            for i, cell in enumerate(voronoi_cells):
                cell_patches.append(mpl.patches.Polygon(cell.points, True))
                circle_patches.append(mpl.patches.Circle(config[i], radius=coverage_range))

            p = mpl.collections.PatchCollection(cell_patches, alpha=0.2)
            p.set_array(np.arange(len(cell_patches))*255/(len(cell_patches)-1))
            p.set_cmap('jet')

            fig, ax = plt.subplots()
            plot_image(image, params, ax)
            ax.add_collection(p)
            ax.add_collection(mpl.collections.PatchCollection(circle_patches, ec='r', fc='none'))
            ax.plot(peaks[:,0], peaks[:,1], 'rx', label='peaks')
            ax.plot(new_config[:,0], new_config[:,1], 'bo', label='centroids')
            # ax.plot(config[:,1], config[:,0], 'bx', color=(0,1,0), label='prev config')
            ax.set_title(f'it {it}, cond = {round_sf(update_dist,3)}')
            ax.legend()
            plt.show()

        # exit if the configuration hasn't appreciably changed
        if update_dist < 1e-5:
            break

        config = new_config
        it += 1

    return config


def segment_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')
    dataset_len = hdf5_file['test']['task_img'].shape[0]

    if args.sample is None:
        idx = np.random.randint(dataset_len)
    elif args.sample > dataset_len:
        print(f'provided sample index {args.sample} out of range of dataset with length {dataset_len}')
        return
    else:
        idx = args.sample

    input_image = hdf5_file['test']['task_img'][idx,...]
    output_image = hdf5_file['test']['comm_img'][idx,...]
    model_image = model.inference(input_image)

    img_scale_factor = int(hdf5_file['train']['task_img'].shape[1] / 128)
    params = cnn_image_parameters(img_scale_factor)

    coverage_points = compute_coverage(model_image, params, viz=args.view)

    fig, ax = plt.subplots()
    if args.isolate:
        plot_image(model_image, params, ax)
    else:
        plot_image(np.maximum(model_image, input_image), params, ax)
    ax.plot(coverage_points[:,0], coverage_points[:,1], 'ro')
    # ax.axis('off')
    plt.show()
    print(f'optimal coverage computed for {dataset_file.name} test partition sample {idx}')


def extract_128px_center_image(image, scale):
    center_idx = 128 * scale // 2
    return image[center_idx-64:center_idx+64, center_idx-64:center_idx+64]


def line_main(args):
    line_test(args.model, args.draws, args.save)


def line_test(model_file, draws=1, save=True):

    model = load_model_for_eval(model_file)
    if model is None:
        return
    model_name = get_file_name(model_file)

    scale = scale_from_filename(model_name)
    params = cnn_image_parameters(scale)

    start_config = np.asarray([[20., 0.], [-20., 0.]])
    step = 2*np.asarray([[1., 0.],[-1., 0.]])
    for i in range(22):
        x_task = start_config + i*step
        img = lloyd.kernelized_config_img(x_task, params)

        cnn_conn, x_cnn, cnn_pwr, cnn_img = connectivity_from_CNN(img, model, x_task, params, draws)
        opt_conn, x_opt = connectivity_from_opt(x_task, params)

        disp_img = extract_128px_center_image(np.maximum(cnn_img, img), scale)
        img_extents = params['img_side_len'] / 2.0 * np.asarray([-1,1,1,-1]) / scale

        fig, ax = plt.subplots()
        ax.imshow(disp_img.T, extent=img_extents)
        ax.invert_yaxis()
        ax.plot(x_task[:,0], x_task[:,1], 'ro', label='task')
        ax.plot(x_opt[:,0], x_opt[:,1], 'rx', label=f'opt ({x_opt.shape[0]})', ms=9, mew=3)
        ax.plot(x_cnn[:,0], x_cnn[:,1], 'bx', label=f'CNN ({x_cnn.shape[0]})', ms=9, mew=3)

        ax.set_yticks(np.arange(-80, 80, 20))
        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.legend(loc='best', fontsize=14)

        pwr_diff = cnn_pwr - params['channel_model'].t
        pwr_str = '0 dBm' if pwr_diff < 0.01 else f'{pwr_diff:.1f} dBm'
        ax.set_title(f'opt. ({opt_conn:.3f}, 0 dBm), CNN ({cnn_conn:.3f}, {pwr_str})', fontsize=18)

        plt.tight_layout()

        if save:
            filename = f'line_{i:02d}_{model_name}.png'
            plt.savefig(filename, dpi=150)
            np.save(filename[:-4], cnn_img)
            plt.close()
            print(f'saved image and array {filename[:-3]+"{png,npy}"}')
        else:
            plt.show()


def circle_main(args):
    circle_test(args.model, args.agents, args.draws, args.save)


def circle_test(model_file, task_agents=3, draws=1, save=True):

    model = load_model_for_eval(model_file)
    if model is None:
        return
    model_name = get_file_name(model_file)

    scale = scale_from_filename(model_name)
    params = cnn_image_parameters(scale)

    min_rad = (params['comm_range']+2.0) / (2.0 * np.sin(np.pi / task_agents))
    rads = np.linspace(min_rad, 60, 15)
    for i, rad in enumerate(rads):
        x_task = circle_points(rad, task_agents)
        img = lloyd.kernelized_config_img(x_task, params)

        cnn_conn, x_cnn, cnn_pwr, cnn_img = connectivity_from_CNN(img, model, x_task, params, draws)
        opt_conn, x_opt = connectivity_from_opt(x_task, params)

        print(f'it {i+1:2d}: rad = {rad:.1f}m, cnn # = {x_cnn.shape[0]}, '
              f'cnn conn = {cnn_conn:.4f}, opt # = {x_opt.shape[0]}, '
              f'opt conn = {opt_conn:.4f}')

        disp_img = extract_128px_center_image(np.maximum(cnn_img, img), scale)
        img_extents = params['img_side_len'] / 2.0 * np.asarray([-1,1,1,-1]) / scale

        fig, ax = plt.subplots()
        ax.imshow(disp_img.T, extent=img_extents)
        ax.invert_yaxis()
        ax.plot(x_task[:,0], x_task[:,1], 'ro', label='task')
        ax.plot(x_opt[:,0], x_opt[:,1], 'rx', label=f'opt ({x_opt.shape[0]})', ms=9, mew=3)
        ax.plot(x_cnn[:,0], x_cnn[:,1], 'bx', label=f'CNN ({x_cnn.shape[0]})', ms=9, mew=3)

        ax.set_yticks(np.arange(-80, 80, 20))
        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.legend(loc='best', fontsize=14)

        pwr_diff = cnn_pwr - params['channel_model'].t
        pwr_str = '0 dBm' if pwr_diff < 0.01 else f'{pwr_diff:.1f} dBm'
        ax.set_title(f'opt. ({opt_conn:.3f}, 0 dBm), CNN ({cnn_conn:.3f}, {pwr_str})', fontsize=18)

        plt.tight_layout()

        if save:
            filename = f'circle_{i:02d}_agents_{task_agents}_{model_name}.png'
            plt.savefig(filename, dpi=150)
            np.save(filename[:-4], cnn_img)
            plt.close()
            print(f'saved image and array {filename[:-3]+"{png,npy}"}')
        else:
            plt.show()


def extrema_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    dataset = ConnectivityDataset(dataset_file, train=False)

    if args.best:
        extrema_test = lambda l : l < extreme_loss
        extreme_loss = np.Inf
        print(f'seeking the best performing sample')
    else:
        extrema_test = lambda l : l > extreme_loss
        extreme_loss = 0.0
        print(f'seeking the worst performing sample')
    with torch.no_grad():
        print(f'looping through {len(dataset)} test samples in {dataset_file}')
        for i in range(len(dataset)):
            print(f'\rprocessing sample {i+1} of {len(dataset)}\r', end="")
            batch = [torch.unsqueeze(ten, 0) for ten in dataset[i]]
            loss = model.validation_step(batch, None)
            if extrema_test(loss):
                extreme_idx = i
                extreme_loss = loss

    hdf5_file = h5py.File(dataset_file, mode='r')
    input_image = hdf5_file['test']['task_img'][extreme_idx,...]
    output_image = hdf5_file['test']['comm_img'][extreme_idx,...]
    model_image = model.inference(input_image)

    img_scale_factor = int(hdf5_file['train']['task_img'].shape[1] / 128)
    params = cnn_image_parameters(img_scale_factor)

    extrema_type = 'best' if args.best else 'worst'
    print(f'{extrema_type} sample is {extreme_idx}{20*" "}')
    ax = plt.subplot(1,3,1)
    plot_image(input_image, params, ax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('input')
    ax = plt.subplot(1,3,2)
    plot_image(output_image, params, ax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('target')
    ax = plt.subplot(1,3,3)
    plot_image(model_image, params, ax)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('output')
    plt.tight_layout()
    plt.show()

    hdf5_file.close()


def connectivity_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    mode = 'train' if args.train else 'test'

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')
    dataset_len = hdf5_file[mode]['task_img'].shape[0]

    if args.sample is None:
        idx = np.random.randint(dataset_len)
    elif args.sample >= dataset_len:
        print(f'sample index {args.sample} out of dataset index range: [0-{dataset_len-1}]')
        return
    else:
        idx = args.sample

    task_img = hdf5_file[mode]['task_img'][idx,...]
    opt_conn = hdf5_file[mode]['connectivity'][idx]
    x_task = hdf5_file[mode]['task_config'][idx,...]
    x_opt = hdf5_file[mode]['comm_config'][idx,...]
    x_opt = x_opt[~np.isnan(x_opt[:,0])]

    img_scale_factor = int(hdf5_file['train']['task_img'].shape[1] / 128)
    params = cnn_image_parameters(img_scale_factor)

    cnn_conn, x_cnn, cnn_L0, cnn_img = connectivity_from_CNN(task_img, model, x_task, params, args.draws)

    ax = plt.subplot()
    plot_image(np.maximum(task_img, cnn_img), params, ax)
    ax.plot(x_task[:,0], x_task[:,1], 'ro', label='task')
    ax.plot(x_opt[:,0], x_opt[:,1], 'rx', label=f'opt ({x_opt.shape[0]})', ms=9, mew=3)
    ax.plot(x_cnn[:,0], x_cnn[:,1], 'bx', label=f'CNN ({x_cnn.shape[0]})', ms=9, mew=3)
    # ax.set_yticks(np.arange(-80, 80, 20))
    ax.legend(loc='best', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=16)

    pwr_diff = cnn_L0 - params['channel_model'].t
    pwr_str = '0 dBm' if pwr_diff < 0.01 else f'{pwr_diff:.1f} dBm'
    ax.set_title(f'{idx}: opt. ({opt_conn:.3f}, 0 dBm), CNN ({cnn_conn:.3f}, {pwr_str})', fontsize=14)

    plt.tight_layout()

    if not args.save:
        print(f'showing sample {idx} from {dataset_file.name}')
        plt.show()
    else:
        filename = str(idx) + '_' + dataset_file.stem + '.png'
        plt.savefig(filename, dpi=150)
        print(f'saved image {filename}')

    hdf5_file.close()


def compute_stats_main(args):
    compute_stats_test(args.model, args.dataset, args.train, args.samples, args.nosave, args.draws)


def compute_stats_test(model_file, dataset_file, train=False, samples=None, nosave=False, draws=1):

    model = load_model_for_eval(model_file)
    if model is None:
        return
    model_name = get_file_name(model_file)

    dataset_file = Path(dataset_file)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')

    mode = 'train' if train else 'test'
    dataset_len = hdf5_file[mode]['task_img'].shape[0]
    if samples is not None:
        if samples > dataset_len:
            print(f'requested sample count ({samples}) > dataset length ({dataset_len})')
            return
        else:
            dataset_len = samples

    img_scale_factor = int(hdf5_file['train']['task_img'].shape[1] / 128)
    p = cnn_image_parameters(img_scale_factor)

    opt_conn = hdf5_file[mode]['connectivity'][:dataset_len]
    cnn_conn = np.zeros_like(opt_conn)
    cnn_pwr = np.zeros_like(opt_conn)
    cnn_count = np.zeros_like(opt_conn, dtype=int)
    opt_count = np.zeros_like(opt_conn, dtype=int)

    for i in range(dataset_len):
        print(f'\rprocessing sample {i+1} of {dataset_len}\r', end="")
        task_img = hdf5_file[mode]['task_img'][i,...]
        x_task = hdf5_file[mode]['task_config'][i,...]
        cnn_conn[i], x_cnn, cnn_pwr[i], _ = connectivity_from_CNN(task_img, model, x_task, p, draws)
        cnn_count[i] = x_cnn.shape[0]
        opt_count[i] = np.sum(~np.isnan(hdf5_file[mode]['comm_config'][i,:,0]))
    print(f'processed {dataset_len} test samples in {dataset_file.name}')

    if not nosave:
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = f'{dataset_len}_samples_{model_name}_{dataset_file.stem}_stats_{timestamp}'
        stats = np.vstack((opt_conn, cnn_conn, cnn_pwr, opt_count, cnn_count)).T
        np.save(filename, stats)
        print(f'saved data to {filename}.npy')
    else:
        print(f'NOT saving data')

    eps = 1e-10
    opt_feasible = opt_conn > eps
    cnn_feasible = cnn_conn > eps
    cnn_morepower = cnn_pwr > p['channel_model'].t
    both_feasible = opt_feasible & cnn_feasible

    agent_count_diff = cnn_count - opt_count

    absolute_error = opt_conn[both_feasible] - cnn_conn[both_feasible]
    percent_error = absolute_error / opt_conn[both_feasible]

    print(f'{np.sum(opt_feasible)}/{dataset_len} feasible with optimization')
    print(f'{np.sum(cnn_feasible)}/{dataset_len} feasible with CNN')
    print(f'{np.sum(cnn_feasible & ~opt_feasible)} cases where only the CNN was feasible')
    print(f'{np.sum(cnn_morepower)} cases where the CNN required more transmit power')
    print(f'cnn power use:  mean = {np.mean(cnn_pwr):.2f}, std = {np.std(cnn_pwr):.4f}')
    print(f'cnn agent diff: mean = {np.mean(agent_count_diff):.3f}, std = {np.std(agent_count_diff):.4f}')
    print(f'absolute error: mean = {np.mean(absolute_error):.4f}, std = {np.std(absolute_error):.4f}')
    print(f'percent error:  mean = {100*np.mean(percent_error):.2f}%, std = {100*np.std(percent_error):.2f}%')

    return filename + '.npy' if not nosave else None


def parse_stats_main(args):
    parse_stats_test(args.stats, args.labels, args.save)


def parse_stats_test(stats_files, labels, save=True):

    if len(stats_files) != len(labels):
        print(f'number of stats files ({len(stats_files)}) must match number of labels ({len(labels)})')
        return

    stats = {}
    for filename, label in zip(stats_files, labels):
        stats_file = Path(filename)
        if not stats_file.exists():
            print(f'{stats_file} does not exist')
            return
        data = np.load(stats_file)
        stats[label] = {'power': data[:,2], 'opt_count': data[:,3], 'cnn_count': data[:,4]}

    # histogram of transmit powers

    powers = [np.round(stats[label]['power'], 1) for label in labels]
    bins = np.hstack(([0, 0.1], np.arange(1, 15, 1.0)))

    fig, ax = plt.subplots()
    ax.hist(powers, bins=bins, stacked=False, log=True, label=labels)
    ax.legend(loc='best', fontsize=16)
    ax.set_xticks(np.arange(0, 15, 3))
    ax.set_xlabel('$P_T$ dBm', fontsize=18)
    ax.set_ylabel('# test cases', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    if save:
        plt.savefig('power_histogram.pdf', dpi=150)
        print('saved power_histogram.pdf')


    # histogram of difference between CNN agent count and opt agent count

    p = cnn_image_parameters()
    agent_diffs = []
    for label in labels:
        mask = np.round(stats[label]['power'], 1) == p['channel_model'].t  # all samples where CNN used default power
        agent_diffs += [(stats[label]['cnn_count'][mask] - stats[label]['opt_count'][mask]).astype(int)]

    bins = np.arange(np.min(np.hstack(agent_diffs)), np.max(np.hstack(agent_diffs))+2, 1)

    fig, ax = plt.subplots()
    ax.hist(agent_diffs, bins=bins, stacked=False, log=True, align='left', label=labels)
    ax.legend(loc='best', fontsize=18)
    ax.set_xticks(bins[:-1])
    ax.set_xlabel('# CNN agents $-$ # opt agents', fontsize=18)
    ax.set_ylabel('# test cases', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    if save:
        plt.savefig('agent_count_histogram.pdf', dpi=150)
        print('saved agent_count_histogram.pdf')
    else:
        plt.show()

    print(f'power: mean = {np.mean(np.hstack(powers)):.2f}, std = {np.std(np.hstack(powers)):.3f}')
    print(f'diffs: mean = {np.mean(np.hstack(agent_diffs)):.4f}, std = {np.std(np.hstack(agent_diffs)):.3f}'
          f' ({sum([len(d) for d in agent_diffs])} / {sum([len(p) for p in powers])})')


def variation_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return

    dataset_file = Path(args.dataset)
    if not dataset_file.exists():
        print(f'provided dataset {dataset_file} not found')
        return
    hdf5_file = h5py.File(dataset_file, mode='r')
    dataset_len = hdf5_file['test']['task_img'].shape[0]

    if args.sample is None:
        idx = np.random.randint(dataset_len)
    elif args.sample > dataset_len:
        print(f'provided sample index {args.sample} out of range of dataset with length {dataset_len}')
        return
    else:
        idx = args.sample

    x = hdf5_file['test']['task_img'][idx,...]

    with torch.no_grad():
        x1 = torch.from_numpy(np.expand_dims(x / 255.0, axis=(0,1))).float()
        y_hat1, _, _ = model(x1)
        y_out1 = torch.clamp(255*y_hat1, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()

        x2 = torch.from_numpy(np.expand_dims(x / 255.0, axis=(0,1))).float()
        y_hat2, _, _ = model(x2)
        y_out2 = torch.clamp(255*y_hat1, 0, 255).cpu().detach().numpy().astype(np.uint8).squeeze()

    diff_x = x1 - x2
    diff_y_hat = y_hat1 - y_hat2
    diff_y_out = y_out1 - y_out2

    print(f'diff_x: min = {diff_x.min()}, max = {diff_x.max()}')
    print(f'diff_y_hat: min = {diff_y_hat.min()}, max = {diff_y_hat.max()}')
    print(f'diff_y_out: min = {diff_y_out.min()}, max = {diff_y_out.max()}')

    # ax = plt.subplot(1,3,1)
    # ax.imshow(model_image1, vmax=255)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.set_title('model_image1')
    # ax = plt.subplot(1,3,2)
    # ax.imshow(model_image2, vmax=255)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.set_title('model_image2')
    # ax = plt.subplot(1,3,3)
    # ax.imshow(np.abs(diff), vmax=255)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.set_title('abs(diff)')
    # plt.tight_layout()
    # plt.show()


def time_test(args):

    model = load_model_for_eval(args.model)
    if model is None:
        return
    model_name = get_file_name(args.model)

    min_agents = 3
    max_agents = 18
    team_sizes = np.arange(min_agents, max_agents+1)
    samples = 10

    img_scale_factor = 2
    params = cnn_image_parameters(img_scale_factor)

    cnn_time = np.zeros((team_sizes.shape[0], samples))
    opt_time = np.zeros_like(cnn_time)

    for team_idx, total_agents in enumerate(team_sizes):

        for sample_idx in range(samples):

            # generate sample with fixed number of agents
            task_agents = ceil(total_agents / 2.0)
            bbx = adaptive_bbx(task_agents, params['comm_range'])
            while True:
                x_task, x_comm = min_feasible_sample(task_agents, params['comm_range'], bbx)
                if x_task.shape[0] + x_comm.shape[0] == total_agents:
                    break
            input_image = lloyd.kernelized_config_img(x_task, params)

            print(f'\rtiming sample {sample_idx}/{samples} for team {total_agents} agent team\r', end="")

            # run CNN
            t0 = time.time()
            cnn_image = model.inference(input_image)
            connectivity_from_CNN(input_image, model, x_task, params, args.draws)
            dt = time.time() - t0
            cnn_time[team_idx, sample_idx] = dt

            # run optimization
            opt = ConnOpt(params['channel_model'], x_task, x_comm)
            t0 = time.time()
            opt.maximize_connectivity(max_its=20)
            dt = time.time() - t0
            opt_time[team_idx, sample_idx] = dt

    # computation time with error bars
    fig, ax = plt.subplots()
    print('cnn mean times:')
    print(np.mean(cnn_time, axis=1))
    print('opt mean times:')
    print(np.mean(opt_time, axis=1))
    ax.errorbar(team_sizes, np.mean(cnn_time, axis=1), yerr=np.std(cnn_time, axis=1),
                color='r', lw=2, label='CNN')
    ax.errorbar(team_sizes, np.mean(opt_time, axis=1), yerr=np.std(opt_time, axis=1),
                color='b', lw=2, label='opt')
    ax.set_xlabel('total agents', fontsize=16)
    ax.set_ylabel('computation time (s)', fontsize=16)
    ax.set_xticks(team_sizes[::2])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN network tests')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # TODO remove draws (since reparameterization is disabled in eval mode)

    line_parser = subparsers.add_parser('line', help='run line test on provided model')
    line_parser.add_argument('model', type=str, help='model to test')
    line_parser.add_argument('--save', action='store_true')
    line_parser.add_argument('--draws', metavar='N', type=int, default=1, help='use best of N model samples')

    circ_parser = subparsers.add_parser('circle', help='run circle test on provided model')
    circ_parser.add_argument('model', type=str, help='model to test')
    circ_parser.add_argument('--save', action='store_true')
    circ_parser.add_argument('--agents', type=int, default=3, help='number of agents in the circle')
    circ_parser.add_argument('--draws', metavar='N', type=int, default=1, help='use best of N model samples')

    extrema_parser = subparsers.add_parser('extrema', help='show examples where the provided model performs well/poorly on the given dataset')
    extrema_parser.add_argument('model', type=str, help='model to test')
    extrema_parser.add_argument('dataset', type=str, help='test dataset')
    extrema_parser.add_argument('--best', action='store_true', help='look for best results instead of worst')

    seg_parser = subparsers.add_parser('segment', help='test out segmentation method for extracting distribution from image')
    seg_parser.add_argument('model', type=str, help='model')
    seg_parser.add_argument('dataset', type=str, help='test dataset')
    seg_parser.add_argument('--sample', type=int, help='sample to test')
    seg_parser.add_argument('--isolate', action='store_true', help='show the CNN output without the input image overlayed')
    seg_parser.add_argument('--view', action='store_true', help="show each iteration of Lloyd's algorithm")

    conn_parser = subparsers.add_parser('connectivity', help='compute connectivity for a CNN output')
    conn_parser.add_argument('model', type=str, help='model')
    conn_parser.add_argument('dataset', type=str, help='test dataset')
    conn_parser.add_argument('--sample', type=int, help='sample to test')
    conn_parser.add_argument('--save', action='store_true')
    conn_parser.add_argument('--train', action='store_true', help='draw sample from training data')
    conn_parser.add_argument('--draws', metavar='N', type=int, default=1, help='use best of N model samples')

    comp_parser = subparsers.add_parser('compute_stats', help='compute performance statistics for a dataset')
    comp_parser.add_argument('model', type=str, help='model')
    comp_parser.add_argument('dataset', type=str, help='test dataset')
    comp_parser.add_argument('--train', action='store_true', help='run stats on training data partition')
    comp_parser.add_argument('--samples', type=int, help='number of samples to process; if omitted all samples in the dataset will be used')
    comp_parser.add_argument('--nosave', action='store_true', help='don\'t save connectivity data')
    comp_parser.add_argument('--draws', metavar='N', type=int, default=1, help='use best of N model samples')

    parse_parser = subparsers.add_parser('parse_stats', help='parse performance statistics saved by compute_stats')
    parse_parser.add_argument('--stats', type=str, help='stats.npy files generated from compute_stats', nargs='+')
    parse_parser.add_argument('--labels', type=str, help='labels to use with each stats file', nargs='+')
    parse_parser.add_argument('--save', action='store_true')

    var_parser = subparsers.add_parser('variation', help='show variation in model outputs')
    var_parser.add_argument('model', type=str, help='model')
    var_parser.add_argument('dataset', type=str, help='test dataset')
    var_parser.add_argument('--sample', type=int, help='sample to test')

    time_parser = subparsers.add_parser('time', help='compare CNN inference time with optimization time')
    time_parser.add_argument('model', type=str, help='model')
    time_parser.add_argument('--draws', metavar='N', type=int, default=1, help='use best of N model samples')

    mpl.rcParams['figure.dpi'] = 150

    args = parser.parse_args()
    if args.command == 'line':
        line_main(args)
    elif args.command == 'circle':
        circle_main(args)
    elif args.command == 'extrema':
        extrema_test(args)
    elif args.command == 'segment':
        segment_test(args)
    elif args.command == 'connectivity':
        connectivity_test(args)
    elif args.command == 'compute_stats':
        compute_stats_main(args)
    elif args.command == 'parse_stats':
        parse_stats_main(args)
    elif args.command == 'variation':
        variation_test(args)
    elif args.command == 'time':
        time_test(args)
