import numpy as np
import matplotlib.pyplot as plt
import sys

k = int(sys.argv[1])
print(k)

total_k = []
def calculate_distance(x, c):
    diff = c-x
    diff_square = np.square(diff)
    sum_square = np.sum(diff_square, axis=1)
    square_root = np.sqrt(sum_square)
    return np.argmin(square_root)


x_matrix = np.loadtxt(fname='iris.data.txt', delimiter=",",usecols=[0, 1, 2, 3])
y_matrix = np.loadtxt(fname='iris.data.txt', delimiter=",", usecols=[4], dtype=str)
centriods = np.zeros((k, 4))
for i in range(0, 4):
    x_min = x_matrix[:, i].min()
    x_max = x_matrix[:, i].max()
    print("-------")
    print(x_min)
    centriods[:,i] = np.random.uniform(low=x_min, high=x_max, size=(1,k))
    centriods[:, i] = np.random.uniform(low=x_min, high=x_max, size=(1, k))
print("Inital centriods")
print(centriods)

nearest_centriod = []
k_sum = 0
for k1 in range(0, 300):
    nearest_centriod = np.apply_along_axis(calculate_distance, 1, x_matrix, centriods)
    for i in range(0, 3):
        index = np.where(nearest_centriod == i)
        under_centriod = x_matrix[index]
        new_centriod = []
        if (np.shape(under_centriod)[0] != 0):
            new_centriod = np.sum(under_centriod, axis=0) / np.shape(under_centriod)[0]
            centriods[i] = new_centriod


print("Final centriods")
print(centriods)



error = []
for i in range(0, 3):
    j = i+1
    error1 = list(nearest_centriod[i*50:j*50])
    error1_clusters = list(set(error1))
    temp = []
    for i in error1_clusters:
        temp.append(error1.count(i))
    cluster_number = temp.index(max(temp))
    cluster_number1 = error1_clusters[cluster_number]
    error.append((len(error1) - error1.count(cluster_number1)))

sum = 0

for i in error:
    sum =+ i

print("accuracy is :")
print(100 - sum)



c = ["green"]
plt.scatter(x_matrix[:,0], x_matrix[:, 1], c=c)
plt.scatter(centriods[:,0], centriods[:,1], c = "red")
plt.xlabel("spepal length")
plt.ylabel("sepal width")
plt.title("sepal plot")
plt.show()





