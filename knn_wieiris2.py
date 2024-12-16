import os
import os.path as osp
import pandas as pd
import numpy as np
import operator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Funktion zum Laden der Daten aus den .csv-Dateien
def load_data_from_csv(input_path):
    data = []
    list = []
    for file in os.listdir(input_path):
        if file.endswith(".csv"):
            end_path = os.path.join(input_path, file)
            # CSV-Datei laden
            df = pd.read_csv(end_path)
            #Extrahiere Labels
            class_label = df['Klasse'][0]
            # Extrahiere Kategorien (Spalten 1-7, 8-14, usw.)
            for i in range(0, df.shape[1]-1, 7):
                category_data = df.iloc[:, i:i + 7].to_numpy()
                data.append(category_data)
            # Daten in ein konsistentes Format bringen
            dataset = np.vstack(data)
            # Labels an das v-stackte Data-ndarray anhängen# Füge den Wert als zusätzliche Spalte hinzu
            new_column = np.full((dataset.shape[0], 1), class_label)
            # Kombiniere die Daten m\ neuen Spalte
            dataset_with_class = np.hstack((dataset, new_column))

            headers = ('Gelenk', 'Measure', 'x3d', 'y3d', 'z3d', 'x2d', 'y2d', 'Klasse')
            finished_dataframe = pd.DataFrame(dataset_with_class, columns=headers)
            list.append(finished_dataframe)
            combined_df = pd.concat(list, ignore_index=True)
            print("hel1", finished_dataframe)
            print("hel2", combined_df)
    print("o")
    return combined_df

def euclidian_distance(row1, row2, length):
    '''
    Caculate the euclidian distance between rows
    '''
    distance = 0

    for x in range(length):
        distance += np.square(row1[x] - row2[x])

    return np.sqrt(distance)

def get_neighbors(dataset, sorted_distances, k):
    '''
    Get the closest neighbors in the range of k elements
    '''
    neighbors = []

    for x in range(k):
        neighbors.append(sorted_distances[x][0])

    return neighbors


def get_sorted_distances(dataset, row):
    '''
    Get sorted distance between the row and the dataset

    '''
    distances = {}

    for x in range(len(dataset)):
        dist = euclidian_distance(row, dataset.iloc[x], row.shape[1])
        distances[x] = dist[0]

    sorted_distances = sorted(distances.items(), key=operator.itemgetter(1))

    return sorted_distances


def get_sorted_neighbourhood(dataset, neighbors):
    '''
    Get the neighbor that has the most votes
    '''
    neighbourhood = {}

    for x in range(len(neighbors)):
        response = dataset.iloc[neighbors[x]][-1]

        if response in neighbourhood:
            neighbourhood[response] += 1
        else:
            neighbourhood[response] = 1

    sorted_neighbourhood = sorted(neighbourhood.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_neighbourhood


def knn(dataset, testInstance, k):
    '''
    Implementation of k-nearest neighbors algorithm
    '''

    sorted_distances = get_sorted_distances(dataset, testInstance)

    neighbors = get_neighbors(dataset, sorted_distances, k)

    sorted_neighbourhood = get_sorted_neighbourhood(dataset, neighbors)

    neighbors.insert(0, sorted_neighbourhood[0][0])

    return neighbors
"""
def plot_thething():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    correctmovement, maliciousmovement, fiftyfiftymovement 
    setosa_x, setosa_y, setosa_z = train[train[:, 0] == 1][:, 1:][:, 0], train[train[:, 0] == 1][:, 1:][:, 1], train[train[:,0] == 1][:,1:][:,2]
    correctmovement, maliciousmovement, fiftyfiftymovement = train[train[:, 0] == 2][:, 1:][:, 0], train[train[:, 0] == 2][:, 1:][:,1], train[train[:, 0] == 2][:, 1:][:, 2]
    virginica_x, virginica_y, virginica_z = train[train[:, 0] == 3][:, 1:][:, 0], train[train[:, 0] == 3][:, 1:][:, 1], \
    train[train[:, 0] == 3][:, 1:][:, 2]

    ax.scatter(setosa_x, setosa_y, setosa_z, c='r', marker='o')
    ax.scatter(versicolor_x, versicolor_y, versicolor_z, c='b', marker='o')
    ax.scatter(virginica_x, virginica_y, virginica_z, c='g', marker='o')

    ax.set_xlabel('X Neighbor')
    ax.set_ylabel('Y Neighbor')
    ax.set_zlabel('Z Neighbor')

    plt.show()
"""
def predict(sepal_length, sepal_width, petal_length, petal_width):
    row = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]])
    result = knn(mydataframe, row, 3)
    neighbors = result[1:]
    category = categories[result[0]]
    return category, neighbors

if __name__ == '__main__':
    input_path = osp.normpath(osp.join(osp.dirname(__file__), "SkeletonData_nurwenige"))
    mydataframe = load_data_from_csv(input_path)
    print("w")
    species = dict(zip(list(mydataframe['Klasse'].unique()), ([1, 2, 3])))
    print("Klasse:", species)
    print("Klasse:", species)
    # print(mydataframe)
    # species = dict(zip(list(mydataframe['Klasse'].unique()), ([1, 2, 3])))
    # print("Species:", species)
    #
    # categories = {v: k for (k, v) in species.items()}
    # print("Categories:", categories)
    train = []
    categories = {1: 'correctmovement', 2: 'maliciousmovement', 3: 'fiftyfiftymovement'}
    for i in range(len(mydataframe)):
        row = pd.DataFrame([list(mydataframe.iloc[i].to_numpy()[0:-1])])
        train.append(knn(mydataframe, row, 3))

    train = np.array(train)












