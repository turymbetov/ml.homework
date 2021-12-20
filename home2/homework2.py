import numpy as np
import config
import json

clusters = None
with open(config.clusters_json, 'r') as f:
    clusters = json.load(f)

def get_cluster_users(ap, cluster):
    ans = []
    l = len(ap)
    for i in range(l):
        if ap[i] == cluster:
            ans.append(i)
    return ans

def count_entities(d: dict, entity):
    if entity in d.keys():
        d[entity] += 1
    else:
        d[entity] = 1

def compute_recommendation(target_user: int, clusters_arr, dataset_path, users_count):
    cluster_users = get_cluster_users(clusters_arr, clusters_arr[target_user])
    d = {}
    with open(dataset_path) as f:
        for line in f:
            user, _, _, _, loc_id = line.strip().split('\t')

            if int(user) >= users_count:
                break

            if int(user) in cluster_users:
                print(user)
                count_entities(d, loc_id)

    sorted_d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    ans = []
    for key, _ in sorted_d:
        ans.append(key)
    return ans[:config.top_n]

def compute_actual(target_user: int, dataset_path):
    d = {}
    with open(dataset_path) as f:
        for line in f:
            user, _, _, _, loc_id = line.strip().split('\t')

            if int(user) == target_user:
                count_entities(d, loc_id)

    sorted_d = sorted(d.items(), key=lambda x: x[1], reverse=True)
    ans = []
    for key, _ in sorted_d:
        ans.append(key)
    return ans[:config.top_n]

ans = compute_recommendation(config.hidden_user, clusters, config.checkins_path, config.vertices_amount)
act = compute_actual(config.hidden_user, config.hidden_checkins_path)
accuracy = 0

for i in act:
    if i in ans:
        accuracy += 1

print(ans)
print(act)
print(accuracy / config.top_n)


