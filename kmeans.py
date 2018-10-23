import numpy as np


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]#样本个数
        k = self.cluster_number#类别数
        k=2
        box_area = boxes[:, 0] * boxes[:, 1]#[样本的面积]
        box_area = box_area.repeat(k)#如果k=2,box_area=[2,12,30],[2,2,12,12,30,30]# axis=None，时候就会flatten当前矩阵，实际上就是变成了一个行向量
        box_area = np.reshape(box_area, (n, k))
        #计算聚类中心的面积shape=（样本数，聚类数目），每行是9个聚类的面积
        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])#[2,12,30,2,12,30]
        cluster_area = np.reshape(cluster_area, (n, k))
        #找出样本和聚类中心宽方向上最小值
        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))#[[1,1],[3,3],[5,5]]
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))#[[1,3],[1,3],[1,3]]
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)#样本数，9

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)#所有样本与各个聚类中心的iou

        result = inter_area / (box_area + cluster_area - inter_area)
        return result#样本数，9

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(#从box_number中随机选取k个作为聚类中心
            box_number, k, replace=False)]  # init k clusters
        while True:
            boxes = np.arange(1,7).reshape(3,2)
            clusters = np.array([[1,2],[3,4]])
            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)#找出所样本中每个样本对应的最大的iou对应的类别
            if (last_nearest == current_nearest).all():#如果连续两次所有样本的类别不变则停止
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)#更新聚类中心

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)#将结果写入文件
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "2007_train.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    kmeans.txt2clusters()
