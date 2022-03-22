import os
from dataProcess import testGenerator, saveResult, color_dict

#  训练模型保存地址
from seg_unet import unet

model_path = r"Model\unet_model3.hdf5"
#  测试数据路径
test_iamge_path = r"E:\data\YAMATO\test"
#  结果保存路径
save_path = r"E:\data\YAMATO\result"
#  测试数据数目
test_num = len(os.listdir(test_iamge_path))
#  类的数目(包括背景)
classNum = 6
#  模型输入图像大小
input_size = (512, 512, 3)
#  生成图像大小
output_size = (512, 512, 3)
#  训练数据标签路径
train_label_path = r"E:\data\YAMATO\label"
#  标签的颜色字典
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)

model = unet(model_path)

testGene = testGenerator(test_iamge_path, input_size)

#  预测值的Numpy数组
results = model.predict(testGene,
                        test_num,
                        verbose=1)

#  保存结果
saveResult(test_iamge_path, save_path, results, colorDict_RGB, output_size)

    