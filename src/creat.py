from glob import glob
import os
import numpy as np
import shutil
'''
#按照标签分类
path='\Python\PPL\data'
files=glob(os.path.join(path,'*.jpg'))
print(f'Total no of images {len(files)}')
no_of_images = len(files)
shuffle=np.random.permutation(no_of_images)
os.mkdir(os.path.join(path,'valid'))
os.mkdir(os.path.join(path,'train'))
for t in ['train','valid']:
    for folder in ['fire/','non/']:
        os.mkdir(os.path.join(path,t,folder))
for i in shuffle[:688]:
    folder = files[i].split('/')[-1].split('.')[0]
    if 'fire' in folder:
        shutil.copy(files[i], '/Python/PPL/data/valid/fire')
    else:
        shutil.copy(files[i], '/Python/PPL/data/valid/non')
    image = files[i].split('/')[-1]
    os.rename(files[i], os.path.join(path, 'valid', folder, image))
for i in shuffle[688:]:
    folder = files[i].split('/')[-1].split('.')[0]
    if 'fire' in folder:
        shutil.copy(files[i], '/Python/PPL/data/train/fire')
    else:
        shutil.copy(files[i], '/Python/PPL/data/train/non')
        image = files[i].split('/')[-1]
        os.rename(files[i], os.path.join(path, 'train', folder, image))
'''
'''
#按文件夹分类
path='/home/ppl/下载/data/'
path1='/home/ppl/下载/data/gray/'
path2='/home/ppl/下载/data/figure_ground/'
files1=glob(os.path.join(path1,'*.jpg'))
files2=glob(os.path.join(path2,'*.jpg'))
print(f'Total no of images in fire {len(files1)}')
print(f'Total no of images in non {len(files2)}')
fireimages = len(files1)
nonimages=len(files2)
shuffle1=np.random.permutation(fireimages)
shuffle2=np.random.permutation(nonimages)
os.mkdir(os.path.join(path,'train'))
os.mkdir(os.path.join(path,'valid'))
for t in ['train','valid']:
    for folder in ['image/','label/']:
        os.mkdir(os.path.join(path,t,folder))

for i in shuffle1[:262]:
    folder = files1[i].split('/')[-1].split('.')[0]
    shutil.copy(files1[i], '/home/ppl/下载/data/train/image')
    folder = files2[i].split('/')[-1].split('.')[0]
    shutil.copy(files2[i], '/home/ppl/下载/data/train/label')
for i in shuffle1[262:]:
    folder = files1[i].split('/')[-1].split('.')[0]
    shutil.copy(files1[i], '/home/ppl/下载/data/valid/image')
    folder = files2[i].split('/')[-1].split('.')[0]
    shutil.copy(files2[i], '/home/ppl/下载/data/valid/label')
'''
'''
for i in shuffle2[:262]:
    folder = files2[i].split('/')[-1].split('.')[0]
    shutil.copy(files2[i], '/home/ppl/下载/data/train/label')
for i in shuffle2[262:]:
    folder = files2[i].split('/')[-1].split('.')[0]
    shutil.copy(files2[i], '/home/ppl/下载/data/valid/label')
'''


#按多文件夹分类
path0 = '/home/ppl/PycharmProjects/da/dataset-resized/'
PATH = '/home/ppl/PycharmProjects/datap/xin1/A1/'
train = os.path.join(PATH, 'train1')
valid = os.path.join(PATH, 'valid1')
os.mkdir(valid)
os.mkdir(train)
path = os.listdir(path0)
path_num = len(path)
for i in range(path_num):
    os.mkdir(os.path.join(train, path[i]))
    os.mkdir(os.path.join(valid, path[i]))

for i in range(path_num):
    path1 = os.path.join(path0, path[i])
    ipath = os.listdir(path1)
    path1_num = len(ipath)
    num = path1_num // 8
    shuffle = np.random.permutation(path1_num)
    for j in shuffle[:num]:
        shutil.copy(os.path.join(path1, ipath[j]), os.path.join(valid, path[i]))
    for j in shuffle[num:]:
        shutil.copy(os.path.join(path1, ipath[j]), os.path.join(train, path[i]))
