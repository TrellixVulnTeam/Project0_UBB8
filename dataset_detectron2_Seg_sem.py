import os
from detectron2.data import MetadataCatalog, DatasetCatalog

def greenhouse_seg_dataset_function(datasetdir: str, txtname: str):
    # txt的绝对路径
    txtpath = os.path.join(datasetdir, txtname)
    # 读取txt的数据
    f = open(txtpath)
    lines = f.readlines()
    f.close()
    # 最终存储的数据，语义分割需要 file_name, height, width, image_id, sem_seg_file_name
    data = []
    # 逐条写入data
    image_id = 0
    for line in lines:
        imgname, annoname = line.strip().split()
        file_name = os.path.join(datasetdir, imgname)
        # 可以使用opencv获取行列数，这里是固定的大小
        height = 1024
        width = 1024
        sem_seg_file_name = os.path.join(datasetdir, annoname)
        item = {"file_name": file_name,
                "height": height,
                "width": width,
                "image_id": image_id,
                "sem_seg_file_name": sem_seg_file_name}
        data.append(item)
        image_id += 1
    # 返回
    return data


datasetdir = "/home/wang/dataset"
trainname = "train_list.txt"
valname = "val_list.txt"

DatasetCatalog.register("greenhouse_seg_train",
                        lambda x=datasetdir, y=trainname: greenhouse_seg_dataset_function(x, y))
DatasetCatalog.register("greenhouse_seg_val",
                        lambda x=datasetdir, y=valname: greenhouse_seg_dataset_function(x, y))

MetadataCatalog.get("greenhouse_seg_train").set(thing_classes=["none_greenhouse", "greenhouse"])
MetadataCatalog.get("greenhouse_seg_val").set(thing_classes=["none_greenhouse", "greenhouse"])
