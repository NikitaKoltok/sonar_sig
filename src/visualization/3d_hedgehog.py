import os
import itertools
import numpy as np
# from scipy.signal import find_peaks
# import pickle
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# from mpl_toolkits.mplot3d import Axes3D
# import collections
# from collections import Counter
# from matplotlib.collections import LineCollection
# from scipy import interpolate
# from scipy.interpolate import interp1d
# from joblib import Parallel, delayed
# from multiprocessing import cpu_count
# from mpl_toolkits.mplot3d import Axes3D, axes3d
# import mpl_toolkits.mplot3d.axes3d as p3
# plt.rcParams['animation.html'] = 'html5'
# import pylab
# from mpl_toolkits.mplot3d import Axes3D
# import csv

num_points = 100
PATH = 'D:\\My_Data\\_Desktop\\distance_portraits\\data'

def define_data():
    global model
    model = 'Harpoon'
    global data
    data = collect(model)

'''
def collect(model):
    """
    Собирать информацию из файла, если углы не заданы 
    Структура данных на выходе: [дистанция, ЭПР, 1 угол, 2 угол, класс]
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
'''
def collect(model): #Собирает информацию из файла
    data={}
    for address, dirs, files in os.walk(os.path.join(PATH, model)):
        for file in files:
            if file.endswith(".output"):
                path = os.path.join(address, file)
                last_angles = (-180, 0)
                with open(path) as data_file:
                    # next(data_file)
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
    return data'''

def collect(model):
    '''
    Передает файл в функцию для формирования данных
     в зависимости от того, содержит ли он отметки углов
    :param model:
    :return: dict{(phi, theta): 'distance':[], 'RCS':[]}
    '''
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
    '''
    Формирует данные из файла, содержащего отметки углов
    :param model:
    :return: dict{(phi, theta): 'distance':[], 'RCS':[]}
    '''
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
    '''
    Формирует данные из файла, не содержащего отметки углов
    :param model:
    :return: dict{(phi, theta): 'distance':[], 'RCS':[]}
    '''
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
            #new_data[phi, theta]['distance'].append(dist[np.argmax(RCS)])
            new_data[phi, theta]['distance'].append(next)
            prev += delta
            next += delta
            dist, RCS = [], []
    if new_data[phi, theta]['RCS'] == []:
        new_data.pop((phi, theta))


def all_discrete(data):
    new_data = {}
    for [phi, theta] in data:
        if phi % 20 == 0  and theta % 20 == 0:
            discrete_data(data, phi, theta, new_data)
    return new_data


def added_graph(data, model, phi, theta): #Строит график зависимости ЭПР от амплитуды для заданных углов
    if not os.path.exists('D:\\My_Data\\_Downloads\\II\\'+model+'\\graphics'):
        os.makedirs('graphics')
    os.chdir('D:\\My_Data\\_Downloads\\II\\'+model+'\\graphics')
    dist, RCS = data[phi, theta].items()
    _, x = dist
    _, y = RCS
    min_peaks, _ = find_peaks(x)
    peaks, _ = find_peaks(x, height=max(x)/2)
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
    plt.show()
    # plt.savefig((str(int(phi))+'_'+str(int(theta)))+'parallel.png')
    plt.close(fig)

def graph(data, model, phi, theta): #Строит график зависимости ЭПР от амплитуды для заданных углов
    if not os.path.exists('D:\\My_Data\\_Downloads\\II\\'+model+'\\graphics'):
        os.makedirs('graphics')
    os.chdir('D:\\My_Data\\_Downloads\\II\\'+model+'\\graphics')
    dist, RCS = data[phi, theta].items()
    _, x = dist
    _, y = RCS
    fig = plt.figure()
    #plt.ylim(0, Max_RCS(data))
    plt.plot(x, y)
    plt.savefig((str(int(phi))+'_'+str(int(theta)))+'parallel.png')
    plt.close(fig)


def all_graph(angle1, angle2): #Строит графики для всех углов
    data_keys = [(angle1[i], angle2[i]) for i in range(len(angle1))]
    os.chdir('D:\\My_Data\\_Downloads\\II\\'+model)
    for [phi, theta] in data_keys:
        print(phi, theta)
        graph(data,model,phi,theta)


def graphics(data):
    data_keys = np.array(list(data.keys()), dtype=float)
    n_workers = cpu_count() - 1
    cur_scope = Parallel(n_workers)(delayed(all_graph)(data_keys.T[0, i::n_workers],
                                                       data_keys.T[1, i::n_workers])
                                     for i in range(n_workers))

if __name__ == '__main__':
    define_data()
    # data = all_discrete(data)
    # added_graph(data, model, 175, 180)

'''
# BLENDER SCRIPT
import bpy
import os
import numpy as np
import colorsys

surface_data = bpy.data.curves.new('wook', 'SURFACE')
surface_data.dimensions = '3D'


def RCS_sphere(data):
    for [phi, theta] in data:
        if phi % 40 == 0 and theta % 40 == 0:
            RCS_Vector(data, phi, theta)

def RCS_Vector(data, phi, theta):
    for i in range(len(data[phi, theta]['distance'])):
        R = data[phi, theta]['distance'][i] + 30
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        rgb = (data[phi, theta]['RCS'][i])/max(data[phi, theta]['RCS'])

        bpy.ops.mesh.primitive_cylinder_add(radius = 0.5, depth = 2, location = (x,y,z), rotation = (0, theta, phi))
        activeObject = bpy.context.active_object
        mat = bpy.data.materials.new(name="MaterialName")
        activeObject.data.materials.append(mat)
        col = colorsys.hsv_to_rgb(0, rgb, 1)
        bpy.context.object.active_material.diffuse_color = col 


filename = "D:\\My_Data\\_DeskTop\\distance_portraits\\visualisation\\data_frame.py"
exec(compile(open(filename).read(), filename, 'exec'))

RCS_sphere(data)'''
