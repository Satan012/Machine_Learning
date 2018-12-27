import numpy as np
import os
from math import ceil

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
        self.node_num = None  # 对节点进行编号

    def set_left(self, left):
        if left is None:
            pass
        left.parent = self
        self.left = left

    def set_right(self, right):
        if right is None:
            pass
        right.parent = self
        self.right = right


def median(lst):
    m = int(len(lst) / 2)
    return lst[m], m


def selectSplit(lst):
    # stdLst = np.std(lst, axis=0)
    maxs = np.max(lst, axis=0)
    mins = np.min(lst, axis=0)
    ranges = maxs -mins
    maxIndex = np.argmax(ranges)
    return maxIndex


def build_kdtree_T(data, max_leaf_num=5000):
    selectedSplit = int(selectSplit([d[0] for d in data]))

    data = sorted(data, key=lambda x: x[0][selectedSplit])

    p, m = median(data)
    tree = node(p[0])
    tree.roof = selectedSplit
    tree.kind = p[1]

    sample_kind.append(tree.kind)  # 为了为每个node打上id，便于k临近搜索
    tree.node_num = len(sample_kind) - 1

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
    # print('dist:', dist)
    return dist


def printPath(search_path):
    result = []
    for s in search_path:
        # print(s.node_num)
        result.append(s.node_num)
    print(result)


def search_kdtree_T1(root, target, kLst):
    search_path = []
    pSearch = root

    while pSearch is not None:  # 搜索近似最近点，创建搜索路径
        search_path.append(pSearch)

        if target[pSearch.roof] < pSearch.point[pSearch.roof]:
            pSearch = pSearch.left
        else:
            pSearch = pSearch.right

    # 用最后一项赋值best
    nearest = [None, 10000, -1]  # init nearest

    if search_path[-1].node_num not in kLst:  # 若近似临近节点已经被认为是k临近中的一个，则忽略它
        nearest = [search_path[-1].point, distance(search_path[-1].point, target), search_path[-1].node_num]

    # 回溯
    while len(search_path) > 0:
        pBack = search_path[-1]
        search_path = search_path[:-1]

        if pBack.left is None and pBack.right is None:  # 该节点是叶子结点且不是已被记录的k临近点
            if distance(pBack.point, target) < nearest[1] and pBack.node_num not in kLst:
                nearest = [pBack.point, distance(pBack.point, target), pBack.node_num]
        else:
            if abs(pBack.point[pBack.roof] - target[pBack.roof]) <= nearest[1]:  # pBack节点在画圆范围内，需要进入子空间搜索
                if nearest[1] > distance(pBack.point, target) and pBack.node_num not in kLst:  # 不是已被记录的k临近点
                    nearest = [pBack.point, distance(pBack.point, target), pBack.node_num]

                # if target[pBack.roof] > pBack.point[pBack.roof]:
                #     pSearch = pBack.right
                # else:
                #     pSearch = pBack.left
                # if pSearch is not None:
                #     search_path.append(pSearch)

                if pBack.right is not None:
                    search_path.append(pBack.right)
                if pBack.left is not None:
                    search_path.append(pBack.left)
    return nearest


def deepSearch(root):
    if root.left is None and root.right is None:
        print(distance(root.point, test_x[0]), root.node_num)
        return

    if root.left is not None:
        deepSearch(root.left)
    if root.right is not None:
        deepSearch(root.right)
    print(distance(root.point, test_x[0]), root.node_num)


def K_neighbors(root, target, sample_kind, k=5):
    kLst = []  # 存储已经找到的邻居点
    for k in range(k):
        tmp_root = root
        near = search_kdtree_T1(tmp_root, target, kLst)
        kLst.append(near[-1])

    if np.sum(np.array(sample_kind)[kLst].astype(int)) >= 3:
        predict = 1
    else:
        predict = 0

    return predict
    # return np.array(sample_kind)[kLst].astype(int)


if __name__ == '__main__':
    leafs = []
    leafs1 = []
    sample_kind = []

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

    kd_tree = build_kdtree_T(dataset, 64)

    from sklearn.neighbors import KDTree

    kd_tree1 = KDTree(train_x, leaf_size=30)
    print('node num:', len(sample_kind))
    print('finished')
    # deepSearch(kd_tree)

    ## test
    # kLst = []
    # for i in range(5):
    #     near = search_kdtree_T1(kd_tree, test_x[2], kLst)
    #     print(near[1])
    #     kLst.append(near[-1])
    #
    # # deepSearch(kd_tree)
    # from sklearn.neighbors import KDTree
    # kd_tree1 = KDTree(train_x, leaf_size=30)
    # dist, ind = kd_tree1.query(np.reshape(test_x[2], [1, -1]), k=5)
    # print(dist)

    predicts = []
    for x in test_x:
        predicts.append(K_neighbors(root=kd_tree, target=x, sample_kind=sample_kind, k=5))

    from sklearn.metrics import confusion_matrix
    result = confusion_matrix(predicts, test_y.astype(int))
    print(result)

    # dist, ind = kd_tree1.query(test_x, k=5)
    # pre = []
    # for i in ind:
    #     # pre.append(train_y[i].astype(int))
    #     if np.sum(train_y[i].astype(int)) >= 3:
    #         pre.append(1)
    #     else:
    #         pre.append(0)
    #
    # for p1, p2 in zip(predicts, pre):
    #     print(p1, p2)
    # from sklearn.metrics import confusion_matrix
    #
    # result = confusion_matrix(pre, test_y.astype(int))
    # print(result)
