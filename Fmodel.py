import cv2
import unicom
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from torchvision.datasets import DTD
from tqdm import tqdm
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import json
torch.manual_seed(2048)

class groundingbox(object):
    def __init__(self, box:list, sim: float, img_id: int, query: str) -> None:
        self.box = box
        self.sim = sim
        self.img_id = img_id
        self.query = query
    def __str__(self) -> str:
        return "this similarity of box {} to query {} is {:.4f}, which id is {}".format(self.box, self.query, self.sim, self.img_id)

def cropped_image(img_path: str, box: list):
    try:
        if img is not None:
            pass
        else:
            print("Failed to read the image. Check the file path.")
            raise TypeError
    except Exception as e:
        print(f"An error occurred: {e}")
    x_min, y_min, x_max, y_max = box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    cropped = img[y_min:y_max, x_min:x_max]
    cv2.imwrite("./cropped_image.jpg", cropped)
    return cropped

# model, preprocess = unicom.load("ViT-B/16")
if __name__ == "__main__":
    # img_id = "2425411995"
    # boxes = [[8, 241, 389, 499], [5, 221, 389, 499], [93, 164, 131, 205]]
    # flickr_root = "your_root"
    # img = cv2.imread(flickr_root + img_id + ".jpg")
    # box = boxes[0]
    # x_min, y_min, x_max, y_max = box
    # print(type(img))
    # cropped = img[y_min:y_max, x_min:x_max]
    # cv2.imwrite("./cropped_image.jpg", cropped)
    # print(preprocess)

    with open('./match.json', 'r', encoding='utf8') as j:
        data = json.load(j)[0]

    phrases = list(data.keys())[0]
    # print(phrases)
    data_detailed = data[phrases]
    # shape[batch_num, top10]
    boxes_list = data_detailed['boxes_list']
    sim_list = data_detailed['sim_list']
    img_id_list = data_detailed['img_id']
    
    s = set()
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[0])):
            box = boxes_list[i][j]
            sim = sim_list[i][j]
            img_id = img_id_list[i][j]
            s.add(tuple([img_id, tuple(box), sim]))
    groundingbox_list = []
    for tu in s:
        img_id, box, sim = tu
        box = list(box)
        gb = groundingbox(box,sim,img_id,phrases)
        groundingbox_list.append(gb)

    flickr_root = "your_root"
    groundingbox_list = sorted(groundingbox_list, key=lambda x: x.sim, reverse=True)

    t1 = time.time()
    model, preprocess = unicom.load("ViT-B/16")
    t2 = time.time()
    model.cuda()
    
    tensor_stack = []
    #crop img
    img_read_list = []
    #TODO stack img_tensor and model(x) -> features
    for i in groundingbox_list:
        box = i.box
        img_id = i.img_id
        sim = i.sim
        img_path = flickr_root + str(img_id) + ".jpg"
        x = cropped_image(img_path, box)
        # add origin tensor
        img_read_list.append(x)
        x = Image.fromarray(x)
        img_tensor = preprocess(x)
        tensor_stack.append(img_tensor)
        # input_imges =torch.stack([img_tensor, img_tensor])
        # f1 = model(input_imges.cuda())
    input_images = torch.stack(tensor_stack).cuda()
    features = model(input_images)
    t3 = time.time()
    print("loading time {}".format(t2-t1))
    print("forwading time {}".format(t3-t2))
    print(features.shape)
    sim_matrix = features @ features.T
    print(sim_matrix)

    total_num = len(img_read_list)
    Width = 6
    import math
    height = math.ceil(total_num / Width)
    fig, axs = plt.subplots(height, Width, dpi=300)
    for i in range(height):
        for j in range(Width):
            axs[i][j].axis('off')
    for idx, img in enumerate(img_read_list):
        axs[idx // Width, idx % Width].imshow(img)
        axs[idx // Width, idx % Width].set_title("sim:{:.2f}".format(groundingbox_list[idx].sim))
        # axs[idx // Width, idx % Width].axis('off')
    plt.tight_layout()
    plt.savefig("./matrix.jpg")



           
    
