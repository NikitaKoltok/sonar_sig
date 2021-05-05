import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

import plotly.express as px


def collect(model, mode):
	"""
	Собирать информацию из файла

	mode == 0: стандартный файл с углами
	mode == 1: файл без указания углов, порядок углов задается оператором
	"""
	data = []
	if mode == 0:
		cur_angles = None

	path = os.path.join('../../input', model + '.output')
	control_points = 0
	is_angle = True
	with open(path) as data_file:
	    for line in tqdm(data_file, ascii=True, desc=model):

	        line_array = list(map(float, line.split()))
	        if not line_array:
	            continue

	        if is_angle and mode == 0:
	        		if control_points != 100 and line_array[0] != -180:
	        			print('line error', line_array, '[angles]:', cur_angles)
	        		cur_angles = (line_array[0], line_array[1])
	        		# print(cur_angles)
	        		control_points = 0
	        		is_angle = False

	        else:
	        	if mode == 0:
	        		if line_array[1] > 1e150:
	        			line_array[1] = 0.0
	        		data.append([line_array[0], line_array[1], model, cur_angles[0], cur_angles[1]])
	        		control_points += 1

	        		if control_points == 100:
	        			is_angle = True 

	        	elif mode == 1:
	        		data.append([line_array[0], line_array[1], model])


	print(f'Data {model} shape:', np.array(data).shape)
	return np.array(data)


def collect_all_data(models, is_normalize=False):
	point_count = 100  # количество точек в портрете
	angle_step = 5  # шаг измерения угла

	# генерируем все углы для данных, где их нет
	angle1_ = np.arange(-180, 185, angle_step)
	angle2_ = np.arange(0, 185, angle_step)
	x, y = np.meshgrid(angle2_, angle1_)
	angle_pairs = np.vstack([y.flatten(), x.flatten()]).T
	angles = np.array([[pair] * point_count for pair in angle_pairs])
	angles = np.array(list(itertools.chain.from_iterable(angles)))

	# имена столбоцов в таблице
	columns = ['Distance', 'RCS', 'Model_name', '1_angle', '2_angle']

	all_data = pd.DataFrame(columns=columns)  # пустой dataframe для всех данных

	# заполнение данных для каждой модели
	for name_model in models:
		mode = models[name_model]  # режим чтения данных из файла
		data = collect(name_model, mode=mode)  # массив данных с [dist, rcs, model_name]
        
		if mode == 1:  # дополнение углов, если их не было
			data = np.hstack([data, angles])

		data = data.T
		# ---> форматирование данных в вид [количество портретов, 1, значения]
		dist_data = data[0]
		dist_data = [[list(dist_data[idx:idx+100])] for idx in range(0, len(dist_data), 100)]
		dist_data = pd.DataFrame(dist_data, columns=['Distance'])

		rcs_data = data[1]
		rcs_data = [[rcs_data[idx:idx+100]] for idx in range(0, len(rcs_data), 100)]
		rcs_data = pd.DataFrame(rcs_data, columns=['RCS'])

		e1_angle_data = data[3]
		e1_angle_data = [e1_angle_data[idx] for idx in range(0, len(e1_angle_data), 100)]
		e1_angle_data = pd.DataFrame(e1_angle_data, columns=['1_angle'])

		e2_angle_data = data[4]
		e2_angle_data = [e2_angle_data[idx] for idx in range(0, len(e2_angle_data), 100)]
		e2_angle_data = pd.DataFrame(e2_angle_data, columns=['2_angle'])

		class_name_data = [data[2][0]] * dist_data.shape[0]
		class_name_data = pd.DataFrame(class_name_data, columns=['Model_name'])

		new_df_data = pd.concat([dist_data, rcs_data, class_name_data,
			                                  e1_angle_data, e2_angle_data], axis=1)

		all_data = all_data.append(new_df_data, ignore_index=True)


	# перевод строковых значений в float
	columns_to_fp = ['Distance', 'RCS','1_angle', '2_angle']
	for i, col in enumerate(columns_to_fp):
		if i in [0,  1]:
			all_data[col] = all_data[col].map(lambda elem: [np.float(el) for el in elem])
		elif i in [2, 3]:
			all_data[col] = all_data[col].map(lambda elem: np.float(elem))

	# добваить id для каждого именя класса (вспомогательное)
	classes = np.unique(all_data.Model_name.values)
	mapping_types = {model: idx for idx, model in enumerate(classes)}
	all_data['Id_class'] = all_data['Model_name']
	all_data = all_data.replace({'Id_class': mapping_types})

	# добавить тип цели: ракета, самолет, вертолет, беспилотник
	mapping_type_models = {'Tomahawk_BGM': 'missile',
						   'F35A': 'fighter', 
						   'F22_raptor': 'fighter', 
						   'Harpoon': 'missile',
						   'AH-1Cobra': 'helicopter',
						   'aerob': 'drone',
						   'Jassm': 'missile',
						   'Eurofightertyphoon': 'fighter',
						   'ExocetAM39': 'missile',
						   'dassaultrafale': 'fighter',
						   'F16': 'fighter',
						   'AH-1WSuperCobra': 'helicopter',
						   'Mig29': 'fighter',
			               'orlan': 'drone'}

	all_data['Model_type'] = all_data['Model_name']
	all_data = all_data.replace({'Model_type': mapping_type_models})

	classes = np.unique(all_data.Model_type.values)
	mapping_types = {model: idx for idx, model in enumerate(classes)}
	all_data['Id_Model_type'] = all_data['Model_type']
	all_data = all_data.replace({'Id_Model_type': mapping_types})

	if is_normalize:  # нормировка данных на максимальное значение в портрете
		for name_class in classes:
			df_class = all_data[all_data.Model_name == name_class]  # опредленный класс
			# найти максимальное значение в портрете
			max_rcs_value = np.max(np.array([elem for elem in df_class.RCS.values]).flatten())
			new_rcs_value = df_class.RCS.map(lambda elem: elem / max_rcs_value)  # нормировать данные
			# заменить данные в исходном dataframe
			all_data.loc[all_data.Model_name == name_class, 'RCS'] = new_rcs_value

	return all_data


def create_dataframe(is_normalize=False, is_plot=False):
	"""
	Сгенерировать DataFrame всех целей.
	Структура данных на выходе: [дистанция, ЭПР, класс, 1 угол, 2 угол, номер класса]
	"""

	# 0 - для данных с углами
	# 1 - для данныз без записанных углов
	models = {'Tomahawk_BGM': 0,
			  'F35A': 0, 
			  'F22_raptor': 1, 
			  'Harpoon': 1,
			  'AH-1Cobra': 0,
			  'aerob': 0,
			  'Jassm': 0,
			  'Eurofightertyphoon': 0,
			  'ExocetAM39': 0,
			  'dassaultrafale': 0,
			  'F16': 0,
			  'AH-1WSuperCobra': 1,
			  'Mig29': 1,
              'orlan': 0}

	all_data = collect_all_data(models, is_normalize=is_normalize)

	print('Final dataframe shape:', all_data.shape)

	if is_plot:
		plot_graphs(all_data)

	return all_data


def plot_graphs(data):
	"""
	Визуализировать информацию и данных
	"""
	data['RCS_log'] = np.log(data['RCS'])
	fig = px.scatter(data, x='Distance', y='RCS', color='Model_name',
		marginal_y='box')
	fig.show()

	fig = px.scatter(data, x='Distance', y='RCS_log', color='Model_name',
		marginal_y='box')
	fig.show()

	fig = px.scatter_matrix(data, dimensions=['Distance', 'RCS','1_angle', '2_angle'], 
		color='Model_name')
	fig.show()

	return True


if __name__ == '__main__':
	
	test_data = create_dataframe(is_plot=False)
