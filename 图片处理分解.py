import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

lena = mpimg.imread(r'C:\Users\user\Pictures\Saved Pictures\螳螂.jpg') # 读取
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理

plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()


# 显示图片的第一个通道
lena_1 = lena[:, :, 0]
plt.imshow(lena_1)
plt.show()
# 此时会发现显示的是热量图，不是我们预想的灰度图，可以添加 cmap 参数，有如下几种添加方法：
plt.imshow(lena_1, cmap='Greys_r')
plt.show()

img = plt.imshow(lena_1)
img.set_cmap('gray') # 'hot' 是热量图
plt.show()


from PIL import Image
i = Image.open(r'C:\Users\user\Pictures\Saved Pictures\螳螂.jpg').convert('L')

from numpy import linalg
u, s, v = linalg.svd(i)

k = 20
i1 = np.dot(u[:, :k], (np.dot(np.diag(s[:k]), v[:k, :])))
i1 = i1.astype('uint8')

# 数组转化为图片
img = Image.fromarray(i1)
img.show()



def rebuild_img(u, sigma, v, p): #p表示奇异值的百分比
    print(p)
    m = len(u)
    n = len(v)
    a = np.zeros((m, n))

    count = (int)(sum(sigma))
    curSum = 0
    k = 0
    while curSum <= count * p:
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
        curSum += sigma[k]
        k += 1
    print('k:',k)
    a[a < 0] = 0
    a[a > 255] = 255
    #按照最近距离取整数，并设置参数类型为uint8
    return np.rint(a).astype("uint8")

if __name__ == '__main__':
    img = Image.open('test.jpg', 'r')
    a = np.array(img)

for p in np.arange(0.1, 1, 0.1):
    u, sigma, v = np.linalg.svd(a[:, :, 0])
    R = rebuild_img(u, sigma, v, p)

    u, sigma, v = np.linalg.svd(a[:, :, 1])
    G = rebuild_img(u, sigma, v, p)

    u, sigma, v = np.linalg.svd(a[:, :, 2])
    B = rebuild_img(u, sigma, v, p)

I = np.stack((R, G, B), 2)
#保存图片在img文件夹下
Image.fromarray(I).save("img\\svd_" + str(p * 100) + ".jpg")





from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier()
dc.feature_importances_