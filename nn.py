import numpy as np
import pandas as pd
from math import inf

def get_distance(x1, y1, x2, y2):
    return np.linalg.norm([x1-x2, y1-y2], 2)

def gen_distance_matrix(data):
    m = len(data)
    dist_mat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            x1, y1 = data.iloc[i, 0:2]
            x2, y2 = data.iloc[j, 0:2]
            dist_mat[i, j] = get_distance(x1, y1, x2, y2)

    return dist_mat

def is_termination(df):
    m = len(df)
    if df['visited'].sum() == m:
        return True
    return False

def get_nearest_point(dist_mat, city_index, df):
    distances = dist_mat[city_index].flatten()
    distances[city_index] = inf
    while True:
        min_i = np.argmin(distances)
        if df.iloc[min_i]['visited'] == 0:
            break
        else:
            distances[min_i] = inf

    return min_i, dist_mat[city_index, min_i]

def NN_algo(data):
    data['next_city'] = 0
    data['visited'] = 0
    path = []

    # 1. Generate Distance Matrix
    dist_mat = gen_distance_matrix(data)

    # 2. Define Starting Point
    current_city_idx = 0
    path.append(current_city_idx+1)

    # 3. Check Termination Condition
    cost = 0
    while not is_termination(data):
        # 2. Choose Nearest Point
        new_city_idx, distance = get_nearest_point(dist_mat, current_city_idx, data)
        
        # update the next city for the current city
        data.iloc[current_city_idx, -2] = new_city_idx + 1

        # update visited condition
        data.iloc[new_city_idx, -1] = 1
        
        # 4. Update Cost
        cost += distance

        current_city_idx = new_city_idx
        path.append(current_city_idx+1)

    return data, cost, path

if __name__ == '__main__':
    df = pd.read_csv("15-Points.csv")
    data, cost, path = NN_algo(df)
    data.to_csv('try.csv')
    # print(np.round(cost, 2))
    # print(path)

