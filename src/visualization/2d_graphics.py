# import pandas as pd
# import matplotlib.cm as cm
import itertools
import os
import pickle
from multiprocessing import cpu_count

# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D, axes3d
# import mpl_toolkits.mplot3d.axes3d as p3
# plt.rcParams['animation.html'] = 'html5'
# import pylab
# from mpl_toolkits.mplot3d import Axes3D
# import csv
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.ticker import (AutoMinorLocator)
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as colors
# import collections
# import itertools
# from collections import Counter
# from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from tqdm import tqdm

PATH = 'D:\\My_Data\\_Desktop\\distance_portraits\\data'
num_points = 100


def Max_RCS(data):
    max_RCS = 0
    for (phi, theta) in data:
        max_RCS = max(max(data[phi, theta]['RCS']), max_RCS)
    return max_RCS


def Max_len(data):
    max_len = 0
    for (phi, theta) in data:
        max_len = max(max(data[phi, theta]['distance']), max_len)
    return max_len


def Min_len(data):
    min_len = []
    for [phi, theta] in data:
        all_dist = data[phi, theta]['distance']
        min_len.append(min(all_dist))
    return min(min_len)


def grid_spacing(data):
    main_min = []
    for [phi, theta] in data:
        main_min.append(min(abs(np.diff(data[phi, theta]['distance']))))
    return min(main_min)


def collect(model):
    """
    Передает файл в функцию для формирования данных
     в зависимости от того, содержит ли он отметки углов
    :param model: str: имя файла модели данных
    :return: dict{(phi, theta): 'distance':[], 'RCS':[]}
    """
    path = os.path.join(PATH, model, model + '.output')
    with open(path) as data_file:
        if '.' in data_file.readline():
            data = collect_without_angles(model)
            print(model, 'without')
        else:
            print(model, 'with')
            data = collect_with_angles(model)
    return data


def collect_with_angles(model):
    """
    Формирует данные из файла, содержащего отметки углов
    :param model: str: имя файла модели данных
    :return: dict{(phi, theta): 'distance':[], 'RCS':[]}
    """
    count_angles = 0
    data = {}
    path = os.path.join(PATH, model, model + '.output')
    last_angles = (0, 0)
    with open(path) as data_file:
        for num, line in enumerate(data_file):
            line_array = list(map(float, line.split()))
            if not (line_array):
                continue
            if num > 0 and count_angles < num_points:
                count_angles += 1
                data[last_angles]['distance'].append(line_array[0])
                data[last_angles]['RCS'].append(line_array[1])
            else:
                last_angles = (line_array[0], line_array[1])
                data[last_angles] = {}
                data[last_angles]['distance'] = []
                data[last_angles]['RCS'] = []
                count_angles = 0
    return data


def collect_without_angles(model):
    """
    Формирует данные из файла, не содержащего отметки углов
    :param model: str: имя файла модели данных
    :return: dict{(phi, theta): 'distance':[], 'RCS':[]}
    """
    data = {}
    angle1_ = np.arange(-180, 185, 5)
    angle2_ = np.arange(0, 185, 5)
    point_count = num_points

    x, y = np.meshgrid(angle2_, angle1_)
    angle_pairs = np.vstack([y.flatten(), x.flatten()]).T
    angles = np.array([[pair] * point_count for pair in angle_pairs])
    angles = np.array(list(itertools.chain.from_iterable(angles)))
    all_angles = []
    for i in range(len(angles)):
        phi, theta = angles[i]
        if not (phi, theta) in all_angles:
            all_angles.append((phi, theta))
            data[(phi, theta)] = {}
            data[(phi, theta)]['RCS'] = np.zeros(num_points)
            data[(phi, theta)]['distance'] = np.zeros(num_points)

    path = os.path.join(PATH, model, model + '.output')
    with open(path) as data_file:
        for num, line in enumerate(data_file):
            angle_point = num // num_points
            angle = all_angles[angle_point]
            line_array = list(map(float, line.split()))
            if not line_array:
                continue
            data[angle]['RCS'][num % num_points] = line_array[1]
            data[angle]['distance'][num % num_points] = line_array[0]
    return data


'''
#  Если заданы углы в файле с ЭПР-дистанцией для модели

def collect(model): #Собирает информацию из файла
    data={}
    for address, dirs, files in os.walk(os.path.join(PATH, model)):
        for file in files:
            if file.endswith(".output"):
                path = os.path.join(address, file)
                last_angles = (0, 0)
                with open(path) as data_file:
                    next(data_file)
                    for line in data_file:
                        line_array = list(map(float, line.split()))
                        if not(line_array):
                            continue
                        if '.' in line:
                            data[last_angles]['distance'].append(line_array[0])
                            data[last_angles]['RCS'].append(line_array[1])
                        else:
                            last_angles = (line_array[0], line_array[1])
                            data[last_angles] = {}
                            data[last_angles]['distance'] = []
                            data[last_angles]['RCS'] = []
    return data
'''


def pkl(data):
    with open('D:\\My_Data\\_Downloads\\II\\data.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('D:\\My_Data\\_Downloads\\II\\data.pkl', 'rb') as f:
        return (pickle.load(f))


def numpy_data_angles(data, phi, theta, new_data):
    new_data[phi, theta] = {}
    new_data[phi, theta]['distance'] = []
    new_data[phi, theta]['RCS'] = []
    y = np.array(data[phi, theta]['RCS'])
    x = np.array(data[phi, theta]['distance'])
    new_data[phi, theta]['distance'] = x
    new_data[phi, theta]['RCS'] = y


def num_data(angle1, angle2):
    new_data = {}
    data_keys = [(angle1[i], angle2[i]) for i in range(len(angle1))]
    for [phi, theta] in data_keys:
        numpy_data_angles(data, phi, theta, new_data)
    return new_data


def numpy_data(data):
    data_keys = np.array(list(data.keys()), dtype=float)
    n_workers = cpu_count() - 1
    cur_scope = Parallel(n_workers)(delayed(num_data)(data_keys.T[0, i::n_workers],
                                                      data_keys.T[1, i::n_workers])
                                    for i in range(n_workers))
    for i in range(n_workers):
        cur_scope[0].update(cur_scope[i])
    return cur_scope[0]


def normalized_data_angles(data, phi, theta, max_len, new_data):  # нормирует данные
    max_dist = max(data[phi, theta]['distance'])
    while round(max_dist) != round(max_len):
        data[phi, theta]['distance'].append(round(max_dist) + 1)
        data[phi, theta]['RCS'].append(0)
    if round(max_dist) == round(max_len):
        data[phi, theta]['distance'].append(round(max_dist) + 1)
        data[phi, theta]['RCS'].append(0)
    new_data[phi, theta] = {}
    new_data[phi, theta]['distance'] = []
    new_data[phi, theta]['RCS'] = []
    new_data[phi, theta]['distance'] = data[phi, theta]['distance']
    new_data[phi, theta]['RCS'] = data[phi, theta]['RCS']


def normalized_data(angle1, angle2):
    new_data = {}
    max_len = Max_len(data)
    data_keys = [(angle1[i], angle2[i]) for i in range(len(angle1))]
    for [phi, theta] in data_keys:
        normalized_data_angles(data, phi, theta, max_len, new_data)
    return new_data


def norm_data(data):
    data_keys = np.array(list(data.keys()), dtype=float)
    n_workers = cpu_count() - 1
    cur_scope = Parallel(n_workers)(delayed(normalized_data)(data_keys.T[0, i::n_workers],
                                                             data_keys.T[1, i::n_workers])
                                    for i in range(n_workers))
    for i in range(n_workers):
        cur_scope[0].update(cur_scope[i])
    return cur_scope[0]


def interpolated_data_angles(data, phi, theta, new_data):  # выдает интерполированные данные
    new_data[phi, theta] = {}
    new_data[phi, theta]['distance'] = []
    new_data[phi, theta]['RCS'] = []
    x = data[phi, theta]['distance']
    y = data[phi, theta]['RCS']
    Y_inter = interp1d(x, y, fill_value='extrapolate', kind='cubic')
    x_mesh = np.linspace(Min_len(data), Max_len(data), 100)  # grid_spacing(data) = 2 * radius of sphere
    new_data[phi, theta]['distance'].append(x_mesh)
    new_data[phi, theta]['RCS'].append(Y_inter(x_mesh))


def interpolated_data(angle1, angle2):
    new_data = {}
    data_keys = [(angle1[i], angle2[i]) for i in range(len(angle1))]
    for [phi, theta] in data_keys:
        interpolated_data_angles(data, phi, theta, new_data)
    return new_data


def interpolat_data(data):
    data_keys = np.array(list(data.keys()), dtype=float)
    n_workers = cpu_count() - 1
    cur_scope = Parallel(n_workers)(delayed(interpolated_data)(data_keys.T[0, i::n_workers],
                                                               data_keys.T[1, i::n_workers])
                                    for i in range(n_workers))
    for i in range(n_workers):
        cur_scope[0].update(cur_scope[i])
    return cur_scope[0]


def data_without_zero_angles(data, phi, theta, new_data):  # накладывает ограничения на данные по значению RCS
    new_data[phi, theta] = {}
    new_data[phi, theta]['distance'] = []
    new_data[phi, theta]['RCS'] = []
    for i in range(len(data[phi, theta]['RCS'])):
        if data[phi, theta]['RCS'][i] <= 1e150:
            new_data[phi, theta]['RCS'].append(data[phi, theta]['RCS'][i])
            new_data[phi, theta]['distance'].append(data[phi, theta]['distance'][i])
    if new_data[phi, theta]['RCS'] == []:
        new_data.pop((phi, theta))


# def data_without_zero(angle1, angle2):
def data_without_zero(data):
    new_data = {}
    # data_keys = [(angle1[i], angle2[i]) for i in range(len(angle1))]
    data_keys = data
    for [phi, theta] in data_keys:
        data_without_zero_angles(data, phi, theta, new_data)
    return new_data


def without_zero(data):
    data_keys = np.array(list(data.keys()), dtype=float)
    n_workers = cpu_count() - 1
    cur_scope = Parallel(n_workers)(delayed(data_without_zero)(data_keys.T[0, i::n_workers],
                                                               data_keys.T[1, i::n_workers])
                                    for i in range(n_workers))
    for i in range(n_workers):
        cur_scope[0].update(cur_scope[i])
    return cur_scope[0]


# def animate(i):
#     ax.view_init(elev=10., azim=i)
#     return fig,
#
#
# def Animation():
#     anim = animation.FuncAnimation(fig, animate, init_func = RCS_sphere(),
#                                    frames=360, interval=20, blit=True)
#     anim.save('animation.mp4', fps=15, extra_args=['-vcodec','h264'])


def graph(data, model, phi=0, theta=45):  # Строит график зависимости ЭПР от амплитуды для заданных углов
    # if not os.path.exists(os.path.join(PATH, model, 'graphics')):
    #     os.makedirs(os.path.join(PATH, model, 'graphics'))
    # os.chdir(os.path.join(PATH, model, 'graphics'))
    models_name = {'Aerob': 'Аэроб', 'Harpoon': 'AGM-84 Harpoon', 'Tomahawk_BGM': 'BGM-109 Tomahawk',
                   'Jassm': 'AGM-158 JASSM', 'ExocetAM39': 'AM-39 Exocet', 'Mig29': 'МиГ-29', 'F16': 'F-16',
                   'F22_raptor': 'F-22 Raptor', 'F35A': 'F-35A', 'EuroFighterTyphoon': 'Eurofighter Typhoon',
                   'Dassaultrafale': 'Dassault Rafale', 'AH-1Cobra': 'Bell AH-1 Cobra', 'AH-1WSuperCobra':
                       'Bell AH-1 Super Cobra', 'Orlan': 'Orlan'}
    os.chdir('D:\\My_Data\\_Desktop\\distance_portraits\\results\\hedgehog_pic\\2d')
    dist, RCS = data[phi, theta].items()
    _, x = np.array(dist)
    _, y = RCS
    # sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.rcParams.update({'font.size': 15})  # размер шрифта
    plt.rcParams["font.family"] = "Times New Roman"  # шрифт
    y = y / np.max(y)
    Y_inter = interp1d(x, y, fill_value='extrapolate', kind='cubic')
    x_mesh = np.linspace(min(x), max(x), num_points * 4)
    y = Y_inter(x_mesh)
    ax.plot(x_mesh, y, 'k')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.xaxis.set_major_locator(MultipleLocator(20))
    # ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.grid(which='both')
    # ax.grid(which='minor')
    ax.set_xlabel('Дальность, м')
    ax.set_ylabel('A/|A max|')
    plt.title('Дальностный портрет ({}; {})'.format(phi, theta))
    plt.suptitle(models_name[model])
    plt.savefig(model + '.png')
    plt.close(fig)


def all_graph(angle1, angle2):  # Строит графики для всех углов
    data_keys = [(angle1[i], angle2[i]) for i in range(len(angle1))]
    for [phi, theta] in data_keys:
        print(phi, theta)
        graph(data, model, phi, theta)


def graphics(data):
    data_keys = np.array(list(data.keys()), dtype=float)
    n_workers = cpu_count() - 1
    cur_scope = Parallel(n_workers)(delayed(all_graph)(data_keys.T[0, i::n_workers],
                                                       data_keys.T[1, i::n_workers])
                                    for i in range(n_workers))


def new_RCS_Vector(data, phi, theta, ax):
    x, y, z = [], [], []
    for i in range(len(data[phi, theta]['distance'])):
        R = data[phi, theta]['distance'][i] + 4  # сдвиг для масштабирования
        x.append(R * np.sin(phi) * np.cos(theta))
        y.append(R * np.sin(phi) * np.sin(theta))
        z.append(R * np.cos(phi))
    # max_RCS = Max_RCS(data) #относительная окраска
    max_RCS = max(data[phi, theta]['RCS'])
    all_RCS = data[phi, theta]['RCS']
    color = [0] * len(all_RCS)
    max_y = [max_RCS] * len(all_RCS)
    for i in range(len(all_RCS)):
        color[i] = (round(max_y[i], 1) - round(all_RCS[i], 1))
    c = ax.scatter(x, y, z, c=data[phi, theta]['RCS'], cmap='hsv', vmin=0, vmax=max_RCS, s=2)
    return c


def the_New_rcs_sphere(angle1, angle2):  # прорисовывает сферу
    data_keys = [(angle1[i], angle2[i]) for i in range(len(angle1))]
    fig = plt.figure()
    ax = Axes3D(fig)
    i = 0
    for [phi, theta] in data_keys:
        if phi % 40 == 0 and theta % 40 == 0:
            # if i % 1000 == 0: #разреженное представление
            new_RCS_Vector(data, phi, theta, ax)
            i += 1
            if i == 1:
                fig.colorbar(new_RCS_Vector(data, phi, theta, ax))
    # plt.savefig('D:\\My_Data\\_Downloads\\II\\'+model+'\\parallel.png')


def new_rcs_sphere(data):  # прорисовывает сферу
    fig = plt.figure()
    ax = Axes3D(fig)
    i = 0
    for [phi, theta] in data:
        if phi % 40 == 0 and theta % 40 == 0:
            # if i % 1000 == 0: #разреженное представление
            new_RCS_Vector(data, phi, theta, ax)
            i += 1
            # if i == 1:
            #     fig.colorbar(new_RCS_Vector(data, phi, theta,ax))
    # plt.savefig('D:\\My_Data\\_Downloads\\II\\'+model+'\\parallel.png')
    plt.show()


def parallel_sphere(data):
    data_keys = np.array(list(data.keys()), dtype=float)
    n_workers = cpu_count() - 1
    cur_scope = Parallel(n_workers)(delayed(new_rcs_sphere)(data_keys.T[0, i::n_workers],
                                                            data_keys.T[1, i::n_workers])
                                    for i in range(n_workers))


def fix_graph(data, phi=-10, theta=35):
    # if not os.path.exists(os.path.join(PATH, 'graphics')):
    #     os.makedirs(os.path.join(PATH, 'graphics'))
    # os.chdir(os.path.join(PATH, 'graphics'))
    dist, RCS = data[phi, theta].items()
    _, x = dist
    _, y = RCS
    sns.set()
    plt.figure()
    # data = {'dist':x}
    # df = pd.DataFrame(data)
    # sns.relplot(kind='line', ci='sd', data=df)
    # plt.ylim(0, Max_RCS(data))
    plt.plot(y, x)
    # plt.savefig((str(int(phi))+'_'+str(int(theta)))+'.png')
    plt.show()


def all_fix_graph(data, model):
    os.chdir('D:\\My_Data\\_Downloads\\II\\' + model)
    for [phi, theta] in data:
        fix_graph(data, model)


def find_peak(data):
    keys = []
    for [phi, theta] in data:
        sign = 0
        for i in range(len(data[phi, theta]['RCS'])):
            if data[phi, theta]['RCS'][i] >= Max_RCS(data) // 2:
                sign = 1
        if sign == 1:
            keys.append((phi, theta))
    return keys


def max_RCS_by_angles(data, model):  # зависимость максимального ЭПР от угла
    x1, x2, y = [], [], []
    for [phi, theta] in data:
        x1.append(phi)
        x2.append(theta)
        y.append(max(data[phi, theta]['RCS']))
    plt.figure()
    plt.subplot(211)
    plt.plot(x1, y)
    plt.subplot(212)
    plt.plot(x2, y)
    plt.savefig('D:\\My_Data\\_Downloads\\II\\' + model + '\\maxRCS.png')


def RCS_by_angles(data, model):  # зависимость ЭПР от угла
    angle1, angle2, RCS = [], [], []
    for [phi, theta] in data:
        angle1 += [phi] * len((data[phi, theta]['RCS']))
        RCS.extend(data[phi, theta]['RCS'])
        angle2 += [theta] * len((data[phi, theta]['RCS']))
    plt.figure()
    plt.subplot(211)
    plt.plot(angle1, RCS)
    plt.subplot(212)
    plt.plot(angle2, RCS)
    plt.savefig('D:\\My_Data\\_Downloads\\II\\' + model + '\\RCS.png')


def histogram_RCS_angles(data):  # считает количество значений с разными ЭПР
    RCS = []
    for [phi, theta] in data:
        RCS.extend(map(int, data[phi, theta]['RCS']))
    plt.hist(RCS)
    plt.savefig('D:\\My_Data\\_Downloads\\II\\' + model + '\\hist_with0.png')
    myDictionary = dict(Counter(RCS))
    new_RCS = []
    for k in myDictionary.keys():
        if k != 0:
            new_RCS.append(k)
    # print(sorted(myDictionary.items(), key=lambda kv: kv[0]))
    plt.figure()
    plt.hist(new_RCS)
    plt.savefig('D:\\My_Data\\_Downloads\\II\\' + model + '\\hist_without0.png')


def threeD_histogram_dist(data):
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.grid()
    x, y, z = [], [], []
    for [phi, theta] in data:
        y.extend(map(int, data[phi, theta]['RCS']))
        x.extend(data[phi, theta]['distance'])
    count_pair = dict(Counter(zip(x, y)))
    z = list(count_pair.values())
    new_x, new_y = [], []
    for k in count_pair.keys():
        new_x.append(k[0])
        new_y.append(k[1])
    ax.bar3d(new_x, new_y, np.zeros_like(z), 0.05, 1, z, zsort='average', shade=True, color='lightpink')
    ax.set_xlabel('дальность')
    ax.set_ylabel('интенсивность')
    plt.savefig('D:\\My_Data\\_Downloads\\II\\' + model + '\\3D_hist.png')


# def threeD_histogram_angle(data):
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     plt.grid()
#     x, y, z = [], [], []
#     for [phi, theta] in data:
#         y.extend(map(int, data[phi, theta]['RCS']))
#         for i in range(len(data[phi, theta]['RCS'])):
#             x.append(phi)
#     count_pair = dict(Counter(zip(x, y)))
#     z = list(count_pair.values())
#     new_x, new_y = [], []
#     for k in count_pair.keys():
#         new_x.append(k[0])
#         new_y.append(k[1])
#     ax.bar3d(new_x, new_y, np.zeros_like(z), 0.05, 1, z, zsort='average', shade=True, color='lightpink')
#     ax.set_xlabel('дальность')
#     ax.set_ylabel('интенсивность')
#     plt.savefig('D:\\My_Data\\_Downloads\\II\\' + model + '\\3D_hist_angle.png')


def discrete_data(data, phi, theta, new_data):
    new_data[phi, theta] = {}
    new_data[phi, theta]['distance'] = []
    new_data[phi, theta]['RCS'] = []
    delta = 1
    prev = data[phi, theta]['distance'][0]
    next = data[phi, theta]['distance'][0] + delta
    dist, RCS = [], []
    for i in range(len(data[phi, theta]['distance'])):
        if data[phi, theta]['distance'][i] >= prev and data[phi, theta]['distance'][i] <= next:
            dist.append(data[phi, theta]['distance'][i])
            RCS.append(data[phi, theta]['RCS'][i])
        elif data[phi, theta]['distance'][i] > next:
            new_data[phi, theta]['RCS'].append(max(RCS))
            # new_data[phi, theta]['distance'].append(dist[np.argmax(RCS)])
            new_data[phi, theta]['distance'].append(next)
            prev += delta
            next += delta
            dist, RCS = [], []
    if new_data[phi, theta]['RCS'] == []:
        new_data.pop((phi, theta))


def discrete_data_loop(angle1, angle2):
    new_data = {}
    data_keys = [(angle1[i], angle2[i]) for i in range(len(angle1))]
    for [phi, theta] in data_keys:
        discrete_data(data, phi, theta, new_data)
    return new_data


def discrete_data_parallel(data):
    data_keys = np.array(list(data.keys()), dtype=float)
    n_workers = cpu_count() - 1
    cur_scope = Parallel(n_workers)(delayed(discrete_data_loop)(data_keys.T[0, i::n_workers],
                                                                data_keys.T[1, i::n_workers])
                                    for i in range(n_workers))
    for i in range(n_workers):
        cur_scope[0].update(cur_scope[i])
    return cur_scope[0]


def checking_discrete(data):
    max_len = 0
    max_angles = (0, 0)
    for [phi, theta] in data:
        if len(data[phi, theta]['distance']) > max_len:
            max_len = len(data[phi, theta]['distance'])
            max_angles = (phi, theta)
    return max_len, max_angles


def all_discrete(data):
    new_data = {}
    for [phi, theta] in data:
        if phi % 20 == 0 and theta % 20 == 0:
            discrete_data(data, phi, theta, new_data)
    return new_data


def model_points_2d(model):  # , ax):
    # model_colors = {'harpoon': (1, 0, 0), 'f22': (0, 0, 1)}
    # models = {'harpoon': 10, 'f22': 20, 'tomahawk':30, 'aerob': 40, 'AH-1Cobra':50,
    #           'eurofightertyphoon':60, 'f35a':70, 'jassm':80}
    models_name = {'Aerob': 'Аэроб', 'Harpoon': 'AGM-84 Harpoon', 'Tomahawk_BGM': 'BGM-109 Tomahawk',
                   'Jassm': 'AGM-158 JASSM', 'ExocetAM39': 'AM-39 Exocet', 'Mig29': 'МиГ-29', 'F16': 'F-16',
                   'F22_raptor': 'F-22 Raptor', 'F35A': 'F-35A', 'EuroFighterTyphoon': 'Eurofighter Typhoon',
                   'Dassaultrafale': 'Dassault Rafale', 'AH-1Cobra': 'Bell AH-1 Cobra', 'AH-1WSuperCobra':
                       'Bell AH-1 Super Cobra', 'Orlan': 'Orlan'}
    os.chdir('D:\\My_Data\\_Desktop\\distance_portraits\\results\\hedgehog_pic\\2d_points')
    data = data_without_zero(collect(model))
    x, y = [], []
    plt.rcParams.update({'font.size': 15})  # размер шрифта
    plt.rcParams["font.family"] = "Times New Roman"  # шрифт
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for (phi, theta) in data:
        x.append(data[phi, theta]['distance'])
        y.append(data[phi, theta]['RCS'])
    x = np.array(list(itertools.chain.from_iterable(x)))
    y = np.array(list(itertools.chain.from_iterable(y)))
    y = y / np.max(y)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax.xaxis.set_major_locator(MultipleLocator(20))
    # ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.grid(which='both')
    ax.set_xlabel('Дальность, м')
    ax.set_ylabel('A/|A max|')
    plt.title(models_name[model])
    # sns.set()
    c = plt.scatter(x, y, c='k', s=2)
    # c = ax.scatter(x, models[model], y, s=2, label=model)
    # ax.contour(Z=y)
    # plt.legend()
    plt.savefig(model + '.png')


# def model_points(model, ax):
#     # model_colors = {'harpoon': (1, 0, 0), 'f22': (0, 0, 1)}
#     # models = {'harpoon': 10, 'f22': 20, 'tomahawk':30, 'aerob': 40, 'AH-1Cobra':50,
#     #           'eurofightertyphoon':60, 'f35a':70, 'jassm':80}
#     os.chdir('D:\\My_Data\\_Desktop\\distance_portraits\\results\\hedgehog_pic\\2d_points')
#     data = data_without_zero(collect(model))
#     x, y = [], []
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     for (phi, theta) in data:
#         x.append(data[phi, theta]['distance'])
#         y.append(data[phi, theta]['RCS'])
#     x = np.array(list(itertools.chain.from_iterable(x)))
#     y = np.array(list(itertools.chain.from_iterable(y)))
#     y = y / np.max(y)
#     ax.xaxis.set_minor_locator(AutoMinorLocator(2))
#     ax.yaxis.set_minor_locator(AutoMinorLocator(2))
#     # ax.xaxis.set_major_locator(MultipleLocator(20))
#     # ax.yaxis.set_major_locator(MultipleLocator(20))
#     ax.grid(which='both')
#     ax.set_xlabel('Дальность, м')
#     ax.set_ylabel('A/|A max|')
#     ax.title(model)
#     # sns.set()
#     c = plt.scatter(x, y, c='k', s=2)
#     # c = ax.scatter(x, models[model], y, s=2, label=model)
#     # ax.contour(Z=y)
#     # plt.legend()
#     plt.savefig(model + '.png')


# def all_model_points():
#     models = ['harpoon', 'f22', 'tomahawk', 'aerob', 'AH-1Cobra',
#               'eurofightertyphoon', 'f35a', 'jassm']
#     sns.set()
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     plt.rcParams.update({'font.size': 15})  # размер шрифта
#     plt.rcParams["font.family"] = "Times New Roman"  # шрифт
#     for model in models:
#         model_points(model, ax)
#     plt.show()


def added_graph(data, model, phi, theta):  # Строит график зависимости ЭПР от амплитуды для заданных углов
    # if not os.path.exists('D:\\My_Data\\_Downloads\\II\\'+model+'\\graphics'):
    #     os.makedirs('graphics')
    os.chdir('D:\\My_Data\\_Downloads\\II\\' + model + '\\graphics')
    dist, RCS = data[phi, theta].items()
    _, x = dist
    _, y = RCS
    min_peaks, _ = find_peaks(x)
    peaks, _ = find_peaks(x, height=max(x) / 2)
    dif_y = y[peaks][0]
    new_y = np.array(y) + dif_y
    add_zeros = np.zeros(3)
    start_add_coord = np.linspace(0, new_y[0], 3)
    end_add_coord = np.linspace(new_y[-1], new_y[-1] + dif_y, 3)

    fig = plt.figure()
    plt.subplot(211)
    plt.plot(y, x)
    plt.plot(y[peaks], x[peaks], "x")
    plt.plot(y[min_peaks][0], x[min_peaks][0], 'x')
    plt.subplot(212)
    new_y = np.append(start_add_coord, new_y)
    new_y = np.append(new_y, end_add_coord)
    x = np.append(add_zeros, x)
    x = np.append(x, add_zeros)
    plt.plot(new_y, x)
    plt.xlim(0, max(new_y))
    plt.suptitle(model)
    plt.xlabel('distance')
    plt.ylabel('RCS')
    plt.show()
    # plt.savefig((str(int(phi))+'_'+str(int(theta)))+'parallel.png')
    plt.close(fig)


def all_angles_added_graph(model, ax, maximum_of_all_RCS,
                           models):  # Строит график зависимости ЭПР от амплитуды для заданных углов
    # models = {'Harpoon': 70, 'F22_raptor': 30, 'Tomahawk_BGM':50, 'Aerob': 60, 'AH-1Cobra':20, 'F35A':10, 'Jassm':40}
    # color = {'Harpoon': 'b', 'F22_raptor': 'g', 'Tomahawk_BGM':'r', 'Aerob': 'c', 'AH-1Cobra':'m', 'F35A':'y', 'Jassm':'crimson'}
    data = collect(model)
    all_values = [*data.values()]
    RCS = np.array(list(object['RCS'] for object in all_values))
    dist = np.array(all_values[0]['distance'])
    # all_RCS = np.array(list(itertools.chain.from_iterable(np.array(list(object['RCS'] for object in all_values)))))
    # all_dist = np.array(list(itertools.chain.from_iterable(object['distance'] for object in all_values)))
    # line_RCS = RCS.reshape(-1)
    # peaks, _ = find_peaks(line_RCS, height=max(line_RCS)/ 4)
    max_RCS = []
    for item in range(RCS.shape[1]):
        max_RCS.append(max(RCS[:, item]))
    max_RCS = np.array(max_RCS)
    if (max_RCS > 1e100).any():
        new_RCS, new_dist = [], []
        for i in range(len(max_RCS)):
            if max_RCS[i] < 1e100:
                new_RCS.append(max_RCS[i])
                new_dist.append(dist[i])
        max_RCS, dist = new_RCS, new_dist
    Y_inter = interp1d(dist, max_RCS, fill_value='extrapolate', kind='cubic')
    x_mesh = np.linspace(0, max(dist), len(dist) * 4)
    Y_inter_res = Y_inter(x_mesh) / maximum_of_all_RCS
    # verts = [(x_mesh[i], np.repeat(models[model], len(x_mesh))[i], Y_inter_res [i]) for i in range(len(x_mesh))]
    ax.add_collection3d(ax.fill_between(x_mesh, Y_inter_res, 0, alpha=0.6), zdir='y', zs=models[model])
    # ax.add_collection3d(ax.fill_between(x_mesh, Y_inter_res, 0, facecolor=color[model], alpha=0.8), zdir='y', zs=models[model])
    ax.plot(x_mesh, np.repeat(models[model], x_mesh.shape[0]), Y_inter_res, label=model)  # , c=color[model])
    # ax.scatter(all_dist, all_RCS, s=2, label=model)
    # ax.scatter(dist[peaks%num_points], np.repeat(models[model], len(peaks)), line_RCS[peaks], marker = "x")
    # ax.legend()


def all_models_all_angles_added_graph(root_path):
    """

    """
    models = ['AH-1Cobra', 'Aerob', 'AH-1WSuperCobra', 'Dassaultrafale',
              'EuroFighterTyphoon', 'ExocetAM39', 'F16', 'F22_raptor', 'F35A',
              'Harpoon', 'Jassm', 'Mig29', 'Tomahawk_BGM', 'Orlan']
    models_name = {'Aerob': 'Аэроб', 'Harpoon': 'AGM-84 Harpoon', 'Tomahawk_BGM': 'BGM-109 Tomahawk',
                   'Jassm': 'AGM-158 JASSM', 'ExocetAM39': 'AM-39 Exocet', 'Mig29': 'МиГ-29', 'F16': 'F-16',
                   'F22_raptor': 'F-22 Raptor', 'F35A': 'F-35A', 'EuroFighterTyphoon': 'Eurofighter Typhoon',
                   'Dassaultrafale': 'Dassault Rafale', 'AH-1Cobra': 'Bell AH-1 Cobra',
                   'AH-1WSuperCobra': 'Bell AH-1 Super Cobra', 'Orlan': 'Orlan'}
    maximum_RCS = 0
    maxs = []
    for model in tqdm(models):
        data = data_without_zero(collect(model))
        Max = Max_RCS(data)
        maxs.append(Max)
        if Max > maximum_RCS:
            maximum_RCS = Max_RCS(data)
    sns.set()
    sns.set_style('white')
    # fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 15})  # размер шрифта
    plt.rcParams["font.family"] = "Times New Roman"  # шрифт
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.grid(False)
    ax.set_xlabel('Дальность, м')
    ax.set_zlabel('A/|A max|')
    args = zip(maxs, models)
    sorted_classes = [i[1] for i in sorted(args, key=lambda x: x[0])]
    args = np.linspace(10 * len(sorted_classes), 10, len(sorted_classes))
    sorted_classes_names = []
    for i in sorted_classes:
        sorted_classes_names.append(models_name[i])
    ax.set_yticks(args)
    ax.set_yticklabels(sorted_classes_names, rotation='45')
    args = dict(zip(sorted_classes, args))
    for model in models:
        all_angles_added_graph(model, ax, maximum_RCS, args)

    out_path = os.path.join(root_path, 'hedgehog_pic')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
        # plt.savefig(out_path)
    plt.show()


if __name__ == "__main__":
    ROOT_PATH = '../../input'
    PATH = os.path.join(ROOT_PATH, 'eda_data')
    all_models_all_angles_added_graph(ROOT_PATH)  # отрисовать все данные

    models = ['AH-1Cobra']  # , 'Aerob', 'AH-1WSuperCobra', 'Dassaultrafale',
    # 'EuroFighterTyphoon', 'ExocetAM39', 'F16', 'F22_raptor', 'F35A',
    # 'Harpoon', 'Jassm', 'Mig29', 'Tomahawk_BGM']
    # all_models_all_angles_added_graph()
    # model = 'Orlan'
    # model_points_2d(model)
    # data = collect(model)
    # graph(data, model)
    # all_model_points()
    # graph(data, model)
    # graphics(data)
    # data = all_discrete(data)
