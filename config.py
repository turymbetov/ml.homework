from os import path

cur_dir = path.dirname(__file__)
edges_path = path.join(cur_dir, 'dataset', 'edges.txt')
checkins_path = path.join(cur_dir, 'dataset', 'checkins.txt')
hidden_checkins_path = path.join(cur_dir, 'dataset', 'hidden_checkins.txt')
clusters_json = path.join(cur_dir, 'answers', 'ap.json')
clusters_path = path.join(cur_dir, 'answers', 'clusters.txt')
vertices_amount = 1000
iterations = 10
hidden_user = 1
top_n = 10