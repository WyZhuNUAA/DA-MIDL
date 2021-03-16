import SimpleITK as sitk
import numpy as np
import torch


def data_flow(img_path, sample_name, sample_labels, center_cors,
              batch_size, patch_size, num_patches, shuffle_flag=True):

    margin = int(np.floor((patch_size - 1) / 2.0))
    input_shape = (batch_size, 1, patch_size, patch_size, patch_size)
    output_shape = (batch_size, 1)

    while True:
        if shuffle_flag:
            sample_name = np.array(sample_name)
            sample_labels = np.array(sample_labels)
            permut = np.random.permutation(len(sample_name))
            np.take(sample_name, permut, out=sample_name)
            np.take(sample_labels, permut, out=sample_labels)
            sample_name = sample_name.tolist()
            sample_labels = sample_labels.tolist()

        inputs = []
        for i_input in range(num_patches):
            inputs.append(np.zeros(input_shape, dtype='float32'))
        outputs = np.ones(output_shape, dtype=np.long)

        i_batch = 0
        img_path = img_path + '{}'
        for i_iter in range(len(sample_name)):

            img_dir = img_path.format(sample_name[i_iter].strip())
            I = sitk.ReadImage(img_dir)
            img = np.array(sitk.GetArrayFromImage(I))

            for i_patch in range(center_cors.shape[1]):
                x_cor = center_cors[0, i_patch]
                y_cor = center_cors[1, i_patch]
                z_cor = center_cors[2, i_patch]
                img_patch = img[x_cor - margin: x_cor + margin + 1,
                                y_cor - margin: y_cor + margin + 1,
                                z_cor - margin: z_cor + margin + 1]

                inputs[i_patch][i_batch, 0, :, :, :] = img_patch

            outputs[i_batch, :] = sample_labels[i_iter] * outputs[i_batch, :]

            i_batch += 1

            if i_batch == batch_size:
                yield (torch.from_numpy(np.array(inputs)), outputs)
                inputs = []
                for i_input in range(num_patches):
                    inputs.append(np.zeros(input_shape, dtype='float32'))
                outputs = np.ones(output_shape, dtype=np.long)
                i_batch = 0


def tst_data_flow(img_path, sample_name, sample_lbl, center_cors, patch_size, num_patches):
    input_shape = (1, 1, patch_size, patch_size, patch_size)
    output_shape = (1, 1)

    margin = int(np.floor((patch_size - 1) / 2.0))

    img_path = img_path + '{}'
    img_dir = img_path.format(sample_name.strip())
    I = sitk.ReadImage(img_dir)
    img = np.array(sitk.GetArrayFromImage(I))

    inputs = []
    for i_input in range(num_patches):
        inputs.append(np.zeros(input_shape, dtype='float32'))

    for i_patch in range(center_cors.shape[1]):
        x_cor = center_cors[0, i_patch]
        y_cor = center_cors[1, i_patch]
        z_cor = center_cors[2, i_patch]

        img_patch = img[x_cor - margin: x_cor + margin + 1,
                        y_cor - margin: y_cor + margin + 1,
                        z_cor - margin: z_cor + margin + 1]

        inputs[i_patch][0, 0, :, :, :] = img_patch

    outputs = sample_lbl * np.ones(output_shape, dtype='long')

    return torch.from_numpy(np.array(inputs)), outputs

