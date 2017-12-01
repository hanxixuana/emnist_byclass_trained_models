#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py as h5


label_dict = {
    '7a': 'z', '79': 'y', '78': 'x', '77': 'w', '76': 'v',
    '75': 'u', '74': 't', '73': 's', '72': 'r', '71': 'q',
    '70': 'p', '6f': 'o', '6e': 'n', '6d': 'm', '6c': 'l',
    '6b': 'k', '6a': 'j', '69': 'i', '68': 'h', '67': 'g',
    '66': 'f', '65': 'e', '64': 'd', '63': 'c', '62': 'b',
    '61': 'a',
    '5a': 'Z', '59': 'Y', '58': 'X', '57': 'W', '56': 'V',
    '55': 'U', '54': 'T', '53': 'S', '52': 'R', '51': 'Q',
    '50': 'P', '4f': 'O', '4e': 'N', '4d': 'M', '4c': 'L',
    '4b': 'K', '4a': 'J', '49': 'I', '48': 'H', '47': 'G',
    '46': 'F', '45': 'E', '44': 'D', '43': 'C', '42': 'B',
    '41': 'A',
    '39': '9', '38': '8', '37': '7', '36': '6', '35': '5',
    '34': '4', '33': '3', '32': '2', '31': '1', '30': '0'
}


def rnd_test(all_pics, all_labels, label_names, n_row=3, n_col=3, save_pic=False):
    picked_idx = np.random.randint(0, all_labels.__len__(), (n_row + 1) * (n_col + 1))
    fig = plt.figure(figsize=(16.0, 12.0))
    loc_idx = 1
    for row_idx in xrange(n_row):
        for col_idx in xrange(n_col):
            ax = fig.add_subplot(n_row, n_col, loc_idx)
            ax.imshow(
                all_pics[picked_idx[loc_idx - 1]].transpose(1, 2, 0).astype('uint8'),
            )
            ax.set_title(
                label_names[all_labels[picked_idx[loc_idx - 1]]]
                +
                '-'
                +
                str(picked_idx[loc_idx - 1])
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            loc_idx += 1
    fig.suptitle('Samples After Standardization')
    if save_pic:
        fig.savefig('sample_' + str(n_row) + '_' + str(n_col) + '.png')


# parameters
crop_percentage = 0.7
input_height = 128
input_width = 128
output_height = 32
output_width = 32

emnist_path = '/home/xixuan/emnist/emnist_by_class'
output_hdf5_dir = 'data'
output_hdf5_file_name = 'hwc_' + str(output_height) + '_x_' + str(output_width) + '.h5'

if __name__ == '__main__':

    class_label = os.listdir(emnist_path)
    data_dir_in_path = [emnist_path + '/' + item + '/train_' + item for item in class_label]

    n_pic_per_class = np.array([len(os.listdir(item)) for item in data_dir_in_path])
    total_pic_n = np.sum(n_pic_per_class)

    crop_x = [
        int(input_height * (1.0 - crop_percentage) / 2.0),
        int(input_height - input_height * ((1.0 - crop_percentage) / 2.0))
    ]
    crop_y = [
        int(input_width * (1.0 - crop_percentage) / 2.0),
        int(input_width - input_width * ((1.0 - crop_percentage) / 2.0))
    ]

    all_pics = np.empty([total_pic_n, 3, output_height, output_width], dtype='float32')
    all_labels = np.empty(total_pic_n, dtype='int')
    all_label_names = []

    # go through all pictures

    count = 0l

    for label_idx, data_dir in enumerate(data_dir_in_path):
        file_names = os.listdir(data_dir)
        all_label_names.append(
            label_dict[data_dir[-2:]]
        )
        print([label_dict[data_dir[-2:]], data_dir])
        for single_file in file_names:
            file_path = os.path.join(
                data_dir,
                single_file
            )
            input_pic = cv2.imread(file_path)
            cropped_pic = input_pic[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]]
            resized_pic = cv2.resize(cropped_pic, (output_height, output_width))
            retyped_pic = resized_pic.astype('float32')
            # rescaled_pic = retyped_pic / resized_pic.max()

            all_pics[count] = retyped_pic.transpose(2, 0, 1)
            all_labels[count] = label_idx

            count += 1

    # global mean and std
    channel_mean = np.mean(all_pics.transpose([1, 0, 2, 3]).reshape(3, -1), 1)
    channel_std = np.std(all_pics.transpose([1, 0, 2, 3]).reshape(3, -1), 1)

    standardized_pics = (
        (
            all_pics
            -
            channel_mean[np.newaxis, :, np.newaxis, np.newaxis]
        )
        /
        channel_std[np.newaxis, :, np.newaxis, np.newaxis]
    )

    if not os.path.exists(os.path.join(os.getcwd(), output_hdf5_dir)):
        os.makedirs(os.path.join(os.getcwd(), output_hdf5_dir))

    with h5.File(os.path.join(os.getcwd(), output_hdf5_dir, output_hdf5_file_name), 'w') as h5_file:
        h5_file['standardized_pics'] = standardized_pics
        h5_file['labels'] = all_labels
        h5_file['label_names'] = np.array(all_label_names).astype('S')
        h5_file['n_pic_per_class'] = n_pic_per_class
        h5_file['channel_mean'] = channel_mean
        h5_file['channel_std'] = channel_std

    rnd_test(standardized_pics, all_labels, all_label_names, 6, 10, True)
