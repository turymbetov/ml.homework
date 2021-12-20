from os import path
from numpy import inf

cur_dir = path.dirname(__file__)
dataset_dir = path.join(cur_dir, '..', 'datasets', 'hw3')
answer_dir = path.join(cur_dir, 'answers')
dataset_names = ['combined_data_1.txt', 'combined_data_2.txt', 'combined_data_3.txt', 'combined_data_4.txt']
dataset_name = 'combined_data_1.txt'
dataset_path = path.join(dataset_dir, dataset_name)
dataset_paths = [path.join(dataset_dir, i) for i in dataset_names]
answer_names = [('w_ans_{0}'.format(i), 'V_ans_{0}'.format(i)) for i in dataset_names]
answer_paths = [(path.join(answer_dir, w), path.join(answer_dir, V)) for (w, V) in answer_names]

csr_paths = [path.join(dataset_dir, 'csr_{0}'.format(i)) for i in dataset_names]

u_count = 2649429
m_count = 17770
k = 2
row_count = 27000000
V_default = 0.5