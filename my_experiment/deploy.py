import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

gpu = None
pth_path = "../alexnet_cirf10_checkpoint.pth.tar"
arch = "alexnet"

print("=> creating model '{}'".format(arch))
model = models.__dict__[arch](weights=models.AlexNet_Weights.DEFAULT)
# print("=> loading checkpoint '{}'".format(pth_path))
# if gpu is None:
#     checkpoint = torch.load(pth_path)
# elif torch.cuda.is_available():
#     # Map model to be loaded to specified single gpu.
#     loc = 'cuda:{}'.format(gpu)
#     checkpoint = torch.load(pth_path, map_location=loc)


# checkpoint = torch.load("./checkpoints/cnn_best.pth")  
# model.load_state_dict(checkpoint['state_dict'])
model.eval()

def stream_processing(test_cb):
    import time
    # fcap = cv2.VideoCapture('demo.mp4')
    fcap = cv2.VideoCapture(0)

    w = fcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = fcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = fcap.get(cv2.CAP_PROP_FPS)
    fcount = fcap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # 获取VideoWriter类实例
    # writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), int(fps), (int(w), int(h)))

    last_time=time.time()
    
    while fcap.isOpened():
        success, frame = fcap.read()
        while success:
            label, y = test_cb(frame)
            interval = time.time()-last_time
            last_time = time.time()
            cv2.putText(frame, 
                    "fps:%.2f label:%s, (%.3f)"%(1./interval, label, y),
                    (0, 50),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, 
                    (0, 255, 255), 1, cv2.LINE_4)
            cv2.imshow("demo", frame)  ## 显示画面
            # 获取帧画面
            success,frame = fcap.read()

            # 保存帧数据
            # writer.write(frame)

            if (cv2.waitKey(20) & 0xff) == ord('q'):  ## 等待20ms并判断是按“q”退出，相当于帧率是50hz，注意waitKey只能传入整数，
                break
        fcap.release()
    # writer.release()
    cv2.destroyAllWindows()  ## 销毁所有opencv显示窗口

def test(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = cv2.resize(image1, (224, 224))
    # cv2.imshow('imshow',image)
    # cv2.imshow('imshow1',image2)
    img = transforms.ToTensor()(image2)
    img=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0)  # 增加一个维度
    # img = Variable(img)
    y_pred = model(img)
    smax = nn.Softmax(1)
    smax_out = smax(y_pred)
    y = smax_out.cpu().data.numpy()
    pred_label = np.argmax(y)
    # print("pred_label ",pred_label, y[0][pred_label])
    # print("类别为1的概率  ",smax_out.cpu().data.numpy()[0][1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return pred_label, y[0][pred_label]

stream_processing(test)

# img_path="n01443537"
# for i in os.listdir(img_path):
#     if os.path.isfile(os.path.join(img_path,i)):
#         image=cv2.imread(os.path.join(img_path,i), flags=1)
#         label, y = test(image)
#         print(f"{i}: {label} - {y}")
