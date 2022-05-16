##
#引用的库
import os.path
import numpy as np
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from scipy import io
import cv2 as cv
##输出数据集（即苹果糖度值）的制备
#糖度值的预处理函数
def normalization(data):#归一化
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):#标准化
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

apple_label= io.loadmat('apple_label_nonblack.mat')['apple_label_nonblack']#读取糖度值数据
#这边习惯把数据保存到matlab分析所以是mat文件，mat是matlab输出文件，也可以换种读取方式读取excel
apple_label_len = np.size(apple_label)#获取糖度值长度
apple_label = np.reshape(apple_label,(apple_label_len,1))#转化下矩阵成列矩阵
apple_label = np.vstack((apple_label[-1],apple_label))#这里在数据后面拼接一个糖度值是为了后面读取苹果图片数据集的方便
#apple_label即为输出数据集
print(apple_label.shape)

#因为我是用归一化的预处理，所以最后网络预测的也是归一化后的结果，所以需要回推
apple_label_max = np.max(apple_label)
apple_label_min = np.min(apple_label)
# print(apple_label_max)
# print(apple_label_min)
apple_label_rang =  apple_label_max-apple_label_min

apple_label = normalization(apple_label)
##
#批量读取文件夹图片改变图片大小
filebox = 'E:/helloworld/ps_test/'
files = os.listdir(filebox)
for filename in files:
    add = filebox+filename
    a = plt.imread(add)
    a = cv.resize(a,(256,256))
    plt.imsave(str(add),a)
##输入数据集（即苹果图片）的制备
data_dir = './ps_test'#苹果所在文件夹路径
data_root = pathlib.Path(data_dir)#读取路径
data_src = data_root.iterdir()
for src in data_root.iterdir():#输出路径
    print(src)
#读取数据集最后一张图，为了后面for系统拼接方便
img1 = plt.imread(src)#item是你数据集中的最后一张图的路径，与前面标签后插入的糖度值对应
img1 = img1.astype(np.float32)
img1 = tf.expand_dims(img1,0)
print(img1.shape)
#批量读取图片并拼接保存为Tensor类型的img1
for item in data_root.iterdir():
    img2 = plt.imread(item)
    img2 = img2.astype(np.float32)
    img2 = tf.expand_dims(img2,0)
    img1 = tf.concat([img1,img2],0)
#img1即为输入数据集
print(img1.shape)#输入数据集大小，需和输出数据集大小匹配
##模型定义
#用预训练模型（不止MobileNetV2这一个）还是自定义模型取决你自己
# 预训练模型
# covn_base = tf.keras.applications.MobileNetV2(weights='imagenet',
#                                         include_top=False,
#                                         input_shape=(256, 256, 3))
#model = tf.keras.Sequential()
# model.add(covn_base)
# model.add(tf.keras.layers.GlobalAveragePooling2D())
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(160, activation='relu'))
# model.add(tf.keras.layers.Dense(80, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#自定义模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(256, 256, 3)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy'
              # loss='mse'
)
history = model.fit(img1 ,#输入，即苹果图片
                    apple_label,#输出，即苹果糖度
                    epochs=15,#训练次数
                    batch_size=16)#更新梯度所需批次，一般越大越好
                    # validation_data=(output_test, input_test))#测试集检验误差，读取方式类似于训练集，上面不再阐释
model.save('apple_test0731.h5')

##loss画图
loss = history.history['loss']
plt.plot(loss,label='Training Loss')
plt.xlabel('epoches')
plt.ylabel('loss')
plt.legend()
plt.show()
##预测
model = tf.keras.models.load_model('apple_test0731.h5')#读取模型
#预测

# img_test = plt.imread('./apple_new/apple093.jpg')
img_test = plt.imread('E:/helloworld/ps_test/apple073 拷贝.jpg')
img_test  =  cv.resize(img_test,(256,256))
img_test = np.reshape(img_test,(256,256,3))
img_test = tf.expand_dims(img_test,0)

result = model.predict(img_test)#预测
print(result)
#按归一化的参数回调回正确糖度值
result_new = result*apple_label_rang+apple_label_min
print(result_new)