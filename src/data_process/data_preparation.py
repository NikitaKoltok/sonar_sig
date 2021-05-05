import numpy as np
import cv2
import shutil
import random
import os


def object_data_sampler(data_folder,
                        big_crop_folder,
                        amp_folder,
                        im_folder,
                        re_folder,
                        res_folder,
                        save_amp=True,
                        save_complex=True,
                        patch_length=5000,
                        obj_size=0.8,
                        validation_part=0.2,
                        test_num=100,
                        classes=[]):
    if not os.listdir(res_folder):
        os.mkdir(os.path.join(res_folder, "patches"))
        os.mkdir(os.path.join(res_folder, "learning"))
        os.mkdir(os.path.join(res_folder, "validation"))
        os.mkdir(os.path.join(res_folder, "test"))
        os.mkdir(os.path.join(res_folder, "test", "objects"))
        if save_amp:
            os.mkdir(os.path.join(res_folder, "patches", "amp"))
            os.mkdir(os.path.join(res_folder, "learning", "amp"))
            os.mkdir(os.path.join(res_folder, "validation", "amp"))
            os.mkdir(os.path.join(res_folder, "test", "objects", "amp"))
        if save_complex:
            os.mkdir(os.path.join(res_folder, "patches", "im"))
            os.mkdir(os.path.join(res_folder, "learning", "im"))
            os.mkdir(os.path.join(res_folder, "validation", "im"))
            os.mkdir(os.path.join(res_folder, "test", "objects", "im"))

            os.mkdir(os.path.join(res_folder, "patches", "re"))
            os.mkdir(os.path.join(res_folder, "learning", "re"))
            os.mkdir(os.path.join(res_folder, "validation", "re"))
            os.mkdir(os.path.join(res_folder, "test", "objects", "re"))

    names = os.listdir(data_folder)

    for name in names:
        if name.endswith('.txt'):
            labels = open(os.path.join(data_folder, name), 'r').readlines()
            full_img = cv2.imread(big_crop_folder + '/' + name.replace('txt', 'png'), cv2.IMREAD_GRAYSCALE)
            x_scale_factor = full_img.shape[1] / 512

            if save_amp:
                amp_data = np.fromfile(amp_folder + '/' + name.replace('txt', 'bin'), dtype=np.float)
                amp_data = np.reshape(amp_data, full_img.shape)
            if save_complex:
                re_data = np.fromfile(re_folder + '/' + name.replace('txt', 'bin'), dtype=np.float)
                im_data = np.fromfile(im_folder + '/' + name.replace('txt', 'bin'), dtype=np.float)
                re_data = np.reshape(re_data, full_img.shape)
                im_data = np.reshape(im_data, full_img.shape)

            for i in range(len(labels)):
                if int(labels[i].split(" ")[0]) in classes:
                    x = int(float(labels[i].split(" ")[1]) * 512 * x_scale_factor)
                    y = int(float(labels[i].split(" ")[2]) * 512)
                    width = int(float(labels[i].split(" ")[3]) * 512 * x_scale_factor)
                    height = int(float(labels[i].split(" ")[4]) * 512)
                    for m in range(int(y - int((height * obj_size)/2)), int(y + int((height * obj_size)/2))):
                        if save_amp:
                            if len(amp_data[m]) > patch_length:
                                amp_patch = amp_data[m][:patch_length]
                                if not os.listdir(os.path.join(res_folder, "patches", "amp")):
                                    amp_patch.tofile(os.path.join(res_folder, "patches", "amp", "0.bin"))
                                else:
                                    file_names = os.listdir(os.path.join(res_folder, "patches", "amp"))

                                    numbers = []
                                    for name in file_names:
                                        numbers.append(int(name.split('.')[0]))

                                    numbers.sort()
                                    new_name = numbers[-1] + 1
                                    new_name = str(new_name) + '.bin'
                                    amp_patch.tofile(os.path.join(res_folder, "patches", "amp", new_name))

                        if save_complex:
                            if len(im_data[m]) > patch_length:
                                re_patch = re_data[m][:patch_length]
                                im_patch = im_data[m][:patch_length]
                                if not os.listdir(os.path.join(res_folder, "patches", "im")):
                                    re_patch.tofile(os.path.join(res_folder, "patches", "re", "0.bin"))
                                    im_patch.tofile(os.path.join(res_folder, "patches", "im", "0.bin"))
                                else:
                                    names = os.listdir(os.path.join(res_folder, "patches", "im"))

                                    numbers = []
                                    for name in names:
                                        numbers.append(int(name.split('.')[0]))

                                    numbers.sort()
                                    new_name = numbers[-1] + 1
                                    new_name = str(new_name) + '.bin'
                                    re_patch.tofile(os.path.join(res_folder, "patches", "re", new_name))
                                    im_patch.tofile(os.path.join(res_folder, "patches", "im", new_name))

    if save_amp and not save_complex:
        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        test_names = random.sample(obj_filenames, test_num)
        for name in test_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "test", "objects", "amp", name))
            os.remove(os.path.join(res_folder, "patches", "amp", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        val_names = random.sample(obj_filenames, int(validation_part * len(obj_filenames)))

        iter_name = 0
        for name in val_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "validation", "amp", str(iter_name) + ".bin"))

            iter_name += 1
            os.remove(os.path.join(res_folder, "patches", "amp", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        iter_name = 0
        for name in obj_filenames:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "learning", "amp", str(iter_name) + ".bin"))
            iter_name += 1

    if save_complex and not save_amp:
        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "im"))
        test_names = random.sample(obj_filenames, test_num)
        for name in test_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "test", "objects", "im", name))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "test", "objects", "re", name))
            os.remove(os.path.join(res_folder, "patches", "im", name))
            os.remove(os.path.join(res_folder, "patches", "re", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "im"))
        val_names = random.sample(obj_filenames, int(validation_part * len(obj_filenames)))

        iter_name = 0
        for name in val_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "validation", "im", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "validation", "re", str(iter_name) + ".bin"))

            iter_name += 1
            os.remove(os.path.join(res_folder, "patches", "im", name))
            os.remove(os.path.join(res_folder, "patches", "re", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "im"))
        iter_name = 0
        for name in obj_filenames:
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "learning", "im", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "learning", "re", str(iter_name) + ".bin"))
            iter_name += 1

    if save_amp and save_complex:
        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        test_names = random.sample(obj_filenames, test_num)
        for name in test_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "test", "objects", "amp", name))
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "test", "objects", "im", name))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "test", "objects", "re", name))
            os.remove(os.path.join(res_folder, "patches", "im", name))
            os.remove(os.path.join(res_folder, "patches", "re", name))
            os.remove(os.path.join(res_folder, "patches", "amp", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        val_names = random.sample(obj_filenames, int(validation_part * len(obj_filenames)))

        iter_name = 0
        for name in val_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "validation", "amp", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "validation", "im", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "validation", "re", str(iter_name) + ".bin"))

            iter_name += 1
            os.remove(os.path.join(res_folder, "patches", "amp", name))
            os.remove(os.path.join(res_folder, "patches", "im", name))
            os.remove(os.path.join(res_folder, "patches", "re", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        iter_name = 0
        for name in obj_filenames:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "learning", "amp", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "learning", "im", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "learning", "re", str(iter_name) + ".bin"))
            iter_name += 1

    shutil.rmtree(os.path.join(res_folder, "patches"), ignore_errors=True)


def background_data_sampler(data_folder,
                            big_crop_folder,
                            amp_folder,
                            im_folder,
                            re_folder,
                            res_folder,
                            save_amp=True,
                            save_complex=True,
                            total_num=4000,
                            patch_length=5000,
                            validation_part=0.2,
                            test_num=100,
                            classes=[0, 2]):

    if not os.listdir(res_folder):
        os.mkdir(os.path.join(res_folder, "patches"))
        os.mkdir(os.path.join(res_folder, "learning"))
        os.mkdir(os.path.join(res_folder, "validation"))
        os.mkdir(os.path.join(res_folder, "test"))
        os.mkdir(os.path.join(res_folder, "test", "background"))
        if save_amp:
            os.mkdir(os.path.join(res_folder, "patches", "amp"))
            os.mkdir(os.path.join(res_folder, "learning", "amp"))
            os.mkdir(os.path.join(res_folder, "validation", "amp"))
            os.mkdir(os.path.join(res_folder, "test", "background", "amp"))
        if save_complex:
            os.mkdir(os.path.join(res_folder, "patches", "im"))
            os.mkdir(os.path.join(res_folder, "learning", "im"))
            os.mkdir(os.path.join(res_folder, "validation", "im"))
            os.mkdir(os.path.join(res_folder, "test", "background", "im"))

            os.mkdir(os.path.join(res_folder, "patches", "re"))
            os.mkdir(os.path.join(res_folder, "learning", "re"))
            os.mkdir(os.path.join(res_folder, "validation", "re"))
            os.mkdir(os.path.join(res_folder, "test", "background", "re"))

    names = os.listdir(data_folder)
    random.shuffle(names)

    if save_amp:
        target_folder = os.path.join(res_folder, "patches", "amp")
    if save_complex:
        target_folder = os.path.join(res_folder, "patches", "im")

    for name in names:
        if len(os.listdir(target_folder)) < total_num:
            if name.endswith('.txt'):
                labels = open(os.path.join(data_folder, name), 'r').readlines()
                if len(labels):
                    full_img = cv2.imread(big_crop_folder + '/' + name.replace('txt', 'png'), cv2.IMREAD_GRAYSCALE)
                    x_scale_factor = full_img.shape[1] / 512

                    if save_amp:
                        amp_data = np.fromfile(amp_folder + '/' + name.replace('txt', 'bin'), dtype=np.float)
                        amp_data = np.reshape(amp_data, full_img.shape)
                    if save_complex:
                        re_data = np.fromfile(re_folder + '/' + name.replace('txt', 'bin'), dtype=np.float)
                        im_data = np.fromfile(im_folder + '/' + name.replace('txt', 'bin'), dtype=np.float)
                        re_data = np.reshape(re_data, full_img.shape)
                        im_data = np.reshape(im_data, full_img.shape)

                    obj_lines = []
                    for i in range(len(labels)):
                        if int(labels[i].split(" ")[0]) in classes:
                            x = int(float(labels[i].split(" ")[1]) * 512 * x_scale_factor)
                            y = int(float(labels[i].split(" ")[2]) * 512)
                            width = int(float(labels[i].split(" ")[3]) * 512 * x_scale_factor)
                            height = int(float(labels[i].split(" ")[4]) * 512)

                            for m in range(int(y - int(height / 2)), int(y + int(height / 2))):
                                obj_lines.append(m)

                            for n in range(512):
                                if n not in obj_lines:
                                    if save_amp:
                                        if len(amp_data[m]) > patch_length:
                                            amp_patch = amp_data[m][:patch_length]
                                            if not os.listdir(os.path.join(res_folder, "patches", "amp")):
                                                if len(os.listdir(target_folder)) < total_num:
                                                    amp_patch.tofile(os.path.join(res_folder, "patches", "amp", "0.bin"))
                                            else:
                                                file_names = os.listdir(os.path.join(res_folder, "patches", "amp"))

                                                numbers = []
                                                for name in file_names:
                                                    numbers.append(int(name.split('.')[0]))

                                                numbers.sort()
                                                new_name = numbers[-1] + 1
                                                new_name = str(new_name) + '.bin'
                                                if len(os.listdir(target_folder)) < total_num:
                                                    amp_patch.tofile(os.path.join(res_folder, "patches", "amp", new_name))

                                    if save_complex:
                                        if len(im_data[m]) > patch_length:
                                            re_patch = re_data[m][:patch_length]
                                            im_patch = im_data[m][:patch_length]
                                            if not os.listdir(os.path.join(res_folder, "patches", "im")):
                                                if len(os.listdir(target_folder)) < total_num:
                                                    re_patch.tofile(os.path.join(res_folder, "patches", "re", "0.bin"))
                                                    im_patch.tofile(os.path.join(res_folder, "patches", "im", "0.bin"))
                                            else:
                                                names = os.listdir(os.path.join(res_folder, "patches", "im"))

                                                numbers = []
                                                for name in names:
                                                    numbers.append(int(name.split('.')[0]))

                                                numbers.sort()
                                                new_name = numbers[-1] + 1
                                                new_name = str(new_name) + '.bin'
                                                if len(os.listdir(target_folder)) < total_num:
                                                    re_patch.tofile(os.path.join(res_folder, "patches", "re", new_name))
                                                    im_patch.tofile(os.path.join(res_folder, "patches", "im", new_name))
                else:
                    full_img = cv2.imread(big_crop_folder + '/' + name.replace('txt', 'png'), cv2.IMREAD_GRAYSCALE)
                    x_scale_factor = full_img.shape[1] / 512

                    if save_amp:
                        amp_data = np.fromfile(amp_folder + '/' + name.replace('txt', 'bin'), dtype=np.float)
                        amp_data = np.reshape(amp_data, full_img.shape)
                    if save_complex:
                        re_data = np.fromfile(re_folder + '/' + name.replace('txt', 'bin'), dtype=np.float)
                        im_data = np.fromfile(im_folder + '/' + name.replace('txt', 'bin'), dtype=np.float)
                        re_data = np.reshape(re_data, full_img.shape)
                        im_data = np.reshape(im_data, full_img.shape)

                    for m in range(512):
                        if save_amp:
                            if len(amp_data[m]) > patch_length:
                                amp_patch = amp_data[m][:patch_length]
                                if not os.listdir(os.path.join(res_folder, "patches", "amp")):
                                    if len(os.listdir(target_folder)) < total_num:
                                        amp_patch.tofile(os.path.join(res_folder, "patches", "amp", "0.bin"))
                                else:
                                    file_names = os.listdir(os.path.join(res_folder, "patches", "amp"))

                                    numbers = []
                                    for name in file_names:
                                        numbers.append(int(name.split('.')[0]))

                                    numbers.sort()
                                    new_name = numbers[-1] + 1
                                    new_name = str(new_name) + '.bin'
                                    if len(os.listdir(target_folder)) < total_num:
                                        amp_patch.tofile(os.path.join(res_folder, "patches", "amp", new_name))

                            if save_complex:
                                if len(im_data[m]) > patch_length:
                                    re_patch = re_data[m][:patch_length]
                                    im_patch = im_data[m][:patch_length]
                                    if not os.listdir(os.path.join(res_folder, "patches", "im")):
                                        if len(os.listdir(target_folder)) < total_num:
                                            re_patch.tofile(os.path.join(res_folder, "patches", "re", "0.bin"))
                                            im_patch.tofile(os.path.join(res_folder, "patches", "im", "0.bin"))
                                    else:
                                        names = os.listdir(os.path.join(res_folder, "patches", "im"))

                                        numbers = []
                                        for name in names:
                                            numbers.append(int(name.split('.')[0]))

                                        numbers.sort()
                                        new_name = numbers[-1] + 1
                                        new_name = str(new_name) + '.bin'
                                        if len(os.listdir(target_folder)) < total_num:
                                            re_patch.tofile(os.path.join(res_folder, "patches", "re", new_name))
                                            im_patch.tofile(os.path.join(res_folder, "patches", "im", new_name))

    if save_amp and not save_complex:
        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        test_names = random.sample(obj_filenames, test_num)
        for name in test_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "test", "background", "amp", name))
            os.remove(os.path.join(res_folder, "patches", "amp", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        val_names = random.sample(obj_filenames, int(validation_part * len(obj_filenames)))

        iter_name = 0
        for name in val_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "validation", "amp", str(iter_name) + ".bin"))

            iter_name += 1
            os.remove(os.path.join(res_folder, "patches", "amp", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        iter_name = 0
        for name in obj_filenames:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "learning", "amp", str(iter_name) + ".bin"))
            iter_name += 1

    if save_complex and not save_amp:
        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "im"))
        test_names = random.sample(obj_filenames, test_num)
        for name in test_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "test", "background", "im", name))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "test", "background", "re", name))
            os.remove(os.path.join(res_folder, "patches", "im", name))
            os.remove(os.path.join(res_folder, "patches", "re", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "im"))
        val_names = random.sample(obj_filenames, int(validation_part * len(obj_filenames)))

        iter_name = 0
        for name in val_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "validation", "im", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "validation", "re", str(iter_name) + ".bin"))

            iter_name += 1
            os.remove(os.path.join(res_folder, "patches", "im", name))
            os.remove(os.path.join(res_folder, "patches", "re", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "im"))
        iter_name = 0
        for name in obj_filenames:
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "learning", "im", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "learning", "re", str(iter_name) + ".bin"))
            iter_name += 1

    if save_complex and save_amp:
        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        test_names = random.sample(obj_filenames, test_num)
        for name in test_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "test", "background", "amp", name))
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "test", "background", "im", name))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "test", "background", "re", name))

            os.remove(os.path.join(res_folder, "patches", "im", name))
            os.remove(os.path.join(res_folder, "patches", "re", name))
            os.remove(os.path.join(res_folder, "patches", "amp", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        val_names = random.sample(obj_filenames, int(validation_part * len(obj_filenames)))

        iter_name = 0
        for name in val_names:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "validation", "amp", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "validation", "im", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "validation", "re", str(iter_name) + ".bin"))

            iter_name += 1
            os.remove(os.path.join(res_folder, "patches", "amp", name))
            os.remove(os.path.join(res_folder, "patches", "im", name))
            os.remove(os.path.join(res_folder, "patches", "re", name))

        obj_filenames = os.listdir(os.path.join(res_folder, "patches", "amp"))
        iter_name = 0
        for name in obj_filenames:
            shutil.copyfile(os.path.join(res_folder, "patches", "amp", name),
                            os.path.join(res_folder, "learning", "amp", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "im", name),
                            os.path.join(res_folder, "learning", "im", str(iter_name) + ".bin"))
            shutil.copyfile(os.path.join(res_folder, "patches", "re", name),
                            os.path.join(res_folder, "learning", "re", str(iter_name) + ".bin"))
            iter_name += 1

    shutil.rmtree(os.path.join(res_folder, "patches"), ignore_errors=True)


def folder_creator(object_folder,
                   background_folder,
                   concat_amp=True,
                   concat_complex=True):
    if concat_amp:
        learn_folder_obj = os.path.join(object_folder, "learning", "amp")
        learn_folder_back = os.path.join(background_folder, "learning", "amp")

        val_folder_obj = os.path.join(object_folder, "validation", "amp")
        val_folder_back = os.path.join(background_folder, "validation", "amp")

    if concat_complex:
        learn_folder_obj = os.path.join(object_folder, "learning", "im")
        learn_folder_back = os.path.join(background_folder, "learning", "im")

        val_folder_obj = os.path.join(object_folder, "validation", "im")
        val_folder_back = os.path.join(background_folder, "validation", "im")

    learning_obj_list = list(np.ones(len(os.listdir(learn_folder_obj))))
    learning_back_list = list(np.zeros(len(os.listdir(learn_folder_back))))

    val_obj_list = list(np.ones(len(os.listdir(val_folder_obj))))
    val_back_list = list(np.zeros(len(os.listdir(val_folder_back))))

    final_learn_list = learning_obj_list + learning_back_list
    final_val_list = val_obj_list + val_back_list

    learning_arr = np.asarray(final_learn_list)
    val_arr = np.asarray(final_val_list)

    learning_arr.tofile(os.path.join(object_folder, "learning", "labels.bin"))
    val_arr.tofile(os.path.join(object_folder, "validation", "labels.bin"))

    if concat_amp:

        names = os.listdir(os.path.join(object_folder, "learning", "amp"))

        numbers = []
        for name in names:
            numbers.append(int(name.split('.')[0]))

        numbers.sort()
        new_name = numbers[-1] + 1

        for filename in os.listdir(os.path.join(background_folder, "learning", "amp")):
            shutil.copyfile(os.path.join(background_folder, "learning", "amp", filename),
                            os.path.join(object_folder, "learning", "amp", str(new_name) + ".bin"))
            new_name += 1

        names = os.listdir(os.path.join(object_folder, "validation", "amp"))

        numbers = []
        for name in names:
            numbers.append(int(name.split('.')[0]))

        numbers.sort()
        new_name = numbers[-1] + 1

        for filename in os.listdir(os.path.join(background_folder, "validation", "amp")):
            shutil.copyfile(os.path.join(background_folder, "validation", "amp", filename),
                            os.path.join(object_folder, "validation", "amp", str(new_name) + ".bin"))
            new_name += 1

    if concat_complex:
        names = os.listdir(os.path.join(object_folder, "learning", "im"))

        numbers = []
        for name in names:
            numbers.append(int(name.split('.')[0]))

        numbers.sort()
        new_name = numbers[-1] + 1

        for filename in os.listdir(os.path.join(background_folder, "learning", "im")):
            shutil.copyfile(os.path.join(background_folder, "learning", "im", filename),
                            os.path.join(object_folder, "learning", "im", str(new_name) + ".bin"))
            shutil.copyfile(os.path.join(background_folder, "learning", "re", filename),
                            os.path.join(object_folder, "learning", "re", str(new_name) + ".bin"))
            new_name += 1

        names = os.listdir(os.path.join(object_folder, "validation", "im"))

        numbers = []
        for name in names:
            numbers.append(int(name.split('.')[0]))

        numbers.sort()
        new_name = numbers[-1] + 1

        for filename in os.listdir(os.path.join(background_folder, "validation", "im")):
            shutil.copyfile(os.path.join(background_folder, "validation", "im", filename),
                            os.path.join(object_folder, "validation", "im", str(new_name) + ".bin"))
            shutil.copyfile(os.path.join(background_folder, "validation", "re", filename),
                            os.path.join(object_folder, "validation", "re", str(new_name) + ".bin"))
            new_name += 1

    os.mkdir(os.path.join(object_folder, "test", "background"))
    if concat_amp:
        os.mkdir(os.path.join(object_folder, "test", "background", "amp"))
        for filename in os.listdir(os.path.join(background_folder, "test", "background", "amp")):
            shutil.copyfile(os.path.join(background_folder, "test", "background", "amp", filename),
                            os.path.join(object_folder, "test", "background", "amp", filename))
    if concat_complex:
        os.mkdir(os.path.join(object_folder, "test", "background", "im"))
        os.mkdir(os.path.join(object_folder, "test", "background", "re"))

        for filename in os.listdir(os.path.join(background_folder, "test", "background", "im")):
            shutil.copyfile(os.path.join(background_folder, "test", "background", "im", filename),
                            os.path.join(object_folder, "test", "background", "im", filename))
            shutil.copyfile(os.path.join(background_folder, "test", "background", "re", filename),
                            os.path.join(object_folder, "test", "background", "re", filename))

    shutil.rmtree(background_folder, ignore_errors=True)


object_data_sampler(data_folder='../../signal_labels_new/obj_train_data',
                    big_crop_folder='../../signal_labels_new/big_crop',
                    amp_folder='../../signal_labels_new/amp_crop_norm',
                    im_folder='../../signal_labels_new/im_crop',
                    re_folder='../../signal_labels_new/re_crop',
                    res_folder='../../signal_labels_new/patches_5000_grand',
                    save_amp=True,
                    save_complex=False,
                    patch_length=5000,
                    obj_size=0.8,
                    test_num=100,
                    validation_part=0.2,
                    classes=[0, 2])

background_data_sampler(data_folder='../../signal_labels_new/obj_train_data',
                        big_crop_folder='../../signal_labels_new/big_crop',
                        amp_folder='../../signal_labels_new/amp_crop_norm',
                        im_folder='../../signal_labels_new/im_crop',
                        re_folder='../../signal_labels_new/re_crop',
                        res_folder='../../signal_labels_new/back_patches_5000_grand',
                        save_amp=True,
                        save_complex=False,
                        total_num=6000,
                        patch_length=5000,
                        test_num=100,
                        validation_part=0.2,
                        classes=[0, 1, 2])

folder_creator(object_folder='../../signal_labels_new/patches_5000_grand',
               background_folder='../../signal_labels_new/back_patches_5000_grand',
               concat_amp=True,
               concat_complex=False)


