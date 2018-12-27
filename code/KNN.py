import numpy as np
import os

from sklearn.model_selection import train_test_split

T = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
L = [1, 0, 1, 1, 1, 1]


class node:
    def __init__(self, point):
        self.left = None
        self.right = None
        self.point = point
        self.parent = None
        self.roof = None
        self.kind = None
        pass

    def set_left(self, left):
        if left is None: pass
        left.parent = self
        self.left = left

    def set_right(self, right):
        if right is None: pass
        right.parent = self
        self.right = right


def median(lst):
    m = len(lst) // 2
    return lst[m], m


def selectSplit(lst):
    stdLst = np.std(lst, axis=0)
    maxIndex = np.argmax(stdLst)
    return maxIndex


def build_kdtree_T(data, max_leaf_num=5000):
    selectedSplit = int(selectSplit([d[0] for d in data]))

    data = sorted(data, key=lambda x: x[0][selectedSplit])

    p, m = median(data)
    tree = node(p[0])
    tree.roof = selectedSplit

    del data[m]
    if m > 0 and len(leafs1) < max_leaf_num:  # 控制叶子节点数
        tree.set_left(build_kdtree_T(data[:m], max_leaf_num))
    if len(data) > 1 and len(leafs1) < max_leaf_num:
        tree.set_right(build_kdtree_T(data[m:], max_leaf_num))
    if tree.left is None and tree.right is None:
        leafs1.append(tree.point)
    return tree


def distance(a, b):
    dist = np.sqrt(np.sum(np.square(a - b)))
    return dist


def printPath(search_path):
    result = []
    for s in search_path:
        result.append(s.point)


def search_kdtree_T1(root, target):
    search_path = []
    pSearch = root

    while pSearch is not None:  # 搜索近似最近点，创建搜索路径
        search_path.append(pSearch)

        if target[pSearch.roof] <= pSearch.point[pSearch.roof]:
            pSearch = pSearch.left
        else:
            pSearch = pSearch.right

    # 用最后一项赋值best
    nearest = [search_path[-1].point, distance(search_path[-1].point, target)]

    # 回溯
    while len(search_path) > 0:
        pBack = search_path[-1]
        search_path = search_path[:-1]

        if pBack.left is None and pBack.right is None:  # 该节点是叶子结点
            if distance(pBack.point, target) < nearest[1]:
                nearest = [pBack.point, distance(pBack.point, target)]
        else:
            if abs(pBack.point[pBack.roof] - target[pBack.roof]) < nearest[1]:
                if nearest[1] > distance(pBack.point, target):
                    nearest = [pBack.point, distance(pBack.point, target)]

                if target[pBack.roof] <= pBack.point[pBack.roof]:
                    pSearch = pBack.right
                else:
                    pSearch = pBack.left
                if pSearch is not None:
                    search_path.append(pSearch)
    return nearest


def deepSearch(root):
    if root.left is None and root.right is None:
        samples.append((root.point, root.kind))
        leafs.append(root.point)
        return

    if root.left is not None:
        deepSearch(root.left)
    if root.right is not None:
        deepSearch(root.right)
    samples.append((root.point, root.kind))


if __name__ == '__main__':
    leafs = []
    leafs1 = []
    indexOrder = []
    samples = []
    kLst = []  # 存储已经找到的邻居点

    FALL = {'bus': ' ', 'bath': ' ', 'teeth': ' ', 'basketball': ' '}

    _dir = os.path.abspath(os.path.dirname(os.getcwd()))

    dataset = np.load(_dir + "/data_mi/spectrogram/fall-128-33-all.npz")

    sample_length = dataset['labels'].shape[0]

    x = dataset['data_padded']
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

    x = dataset['data_padded']
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    y = dataset['user']
    label = dataset['labels']

    for i in range(sample_length):
        if FALL.get(label[i]) is not None:
            m = 1
            y[i] = m
        else:
            m = 0
            y[i] = m

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)

    dataset = [(d, l) for d, l in zip(train_x.tolist(), train_y.tolist())]

    kd_tree = build_kdtree_T(dataset, 30)

    # from sklearn.neighbors import KDTree
    # kd_tree1 = KDTree(train_x, leaf_size=30)

    deepSearch(kd_tree)
    print('node num:', len(samples))
    print('finished')
    print(search_kdtree_T1(kd_tree, test_x[0])[0])
    # dist, ind = kd_tree1.query(np.reshape(test_x[0], [1, -1]), k=1)
