import os
import shutil
import json
import numpy as np

zip_path = "/home/zhouli/work/data/garbage_classify.zip"
out_path = os.path.dirname(zip_path)

train_path = os.path.join(out_path, "garbage_classify", 'train')
valid_path = os.path.join(out_path, "garbage_classify", 'valid')
valid_proportion = 0.1
category_num = 43

id2class = {
0: "Other_garbage-disposable_snack_box",  # "0": "其他垃圾/一次性快餐盒",
1: "Other_garbage-soiled_plastic",  # "1": "其他垃圾/污损塑料",
2: "Other_garbage-cigarette_butts",  # "2": "其他垃圾/烟蒂",
3: "Other_garbage-toothpicks",  # "3": "其他垃圾/牙签",
4: "Other_garbage-broken_flower_pots_and_bowls",  # "4": "其他垃圾/破碎花盆及碟碗",
5: "Other_garbage-bamboo_chopsticks",  # "5": "其他垃圾/竹筷",
6: "Kitchen_waste-leftovers",  # "6": "厨余垃圾/剩饭剩菜",
7: "Kitchen_waste-big_bones",  # "7": "厨余垃圾/大骨头",
8: "Kitchen_waste-fruit_peel",  # "8": "厨余垃圾/水果果皮",
9: "Kitchen_waste-fruit_pulp",  # "9": "厨余垃圾/水果果肉",
10: "Kitchen_waste-tea_residue",  # "10": "厨余垃圾/茶叶渣",
11: "Kitchen_waste-vegetable_leaf_and_root",  # "11": "厨余垃圾/菜叶菜根",
12: "Kitchen_waste-eggshell",  # "12": "厨余垃圾/蛋壳",
13: "Kitchen_waste-fish_bones",  # "13": "厨余垃圾/鱼骨",
14: "Recyclables-Power_Bank",  # "14": "可回收物/充电宝",
15: "Recyclables-Bags",  # "15": "可回收物/包",
16: "Recyclable-cosmetic_bottle",  # "16": "可回收物/化妆品瓶",
17: "Recyclables-plastic_toys",  # "17": "可回收物/塑料玩具",
18: "Recyclable-plastic_bowl",  # "18": "可回收物/塑料碗盆",
19: "Recyclable-plastic_hanger",  # "19": "可回收物/塑料衣架",
20: "Recyclable_materials-express_paper_bags",  # "20": "可回收物/快递纸袋",
21: "Recyclable-Plug_Wire",  # "21": "可回收物/插头电线",
22: "Recyclables-used_clothing",  # "22": "可回收物/旧衣服",
23: "Recyclables-cans",  # "23": "可回收物/易拉罐",
24: "Recyclables-Pillows",  # "24": "可回收物/枕头",
25: "Recyclables-plush_toys",  # "25": "可回收物/毛绒玩具",
26: "Recyclable-shampoo_bottle",  # "26": "可回收物/洗发水瓶",
27: "Recyclable-glass",  # "27": "可回收物/玻璃杯",
28: "recyclables-leather_shoes",  # "28": "可回收物/皮鞋",
29: "Recyclables-Chopping_boards",  # "29": "可回收物/砧板",
30: "Recyclables-Cartons",  # "30": "可回收物/纸板箱",
31: "Recycle-condiment_bottle",  # "31": "可回收物/调料瓶",
32: "Recyclables-Bottles",  # "32": "可回收物/酒瓶",
33: "Recyclable-metal_food_cans",  # "33": "可回收物/金属食品罐",
34: "Recyclables-Pots",  # "34": "可回收物/锅",
35: "Recyclable-edible_oil_barrel",  # "35": "可回收物/食用油桶",
36: "Recyclables-Beverage_Bottles",  # "36": "可回收物/饮料瓶",
37: "Hazardous_waste-dry_battery",  # "37": "有害垃圾/干电池",
38: "Hazardous_waste-ointment",  # "38": "有害垃圾/软膏",
39: "Hazardous_waste-expired_drugs",  # "39": "有害垃圾/过期药物",
40: "Recyclables-Towels",  # "40": "可回收物/毛巾",
41: "Recyclable-Drink_Box",  # "41": "可回收物/饮料盒",
42: "Recyclables-paper_bags"  # "42": "可回收物/纸袋"
}


# shutil.unpack_archive(filename=zip_path, extract_dir=out_path)


def main():

    try:
        rule = json.loads(os.path.join(out_path, "garbage_classify",
                                       "garbage_classify_rule.json"),
                          encoding='utf-8')
        print(rule)
    except Exception as e:
        print(e)

    mkdir(valid_path)
    mkdir(train_path)
    for i in range(category_num):
        mkdir(os.path.join(train_path, "%s" % id2class[i]))
        mkdir(os.path.join(valid_path, "%s" % id2class[i]))

    img_path = os.path.join(out_path, "garbage_classify", "train_data")
    for i in os.listdir(img_path):
        file = os.path.join(img_path, i)
        name, ext = os.path.splitext(file)
        if ext == ".txt":
            with open(file, 'r', encoding='utf-8') as f:
                data = f.readline()
                data = data.split(", ")
                # print(data)
                id = data[1]
                img_name = data[0]
                shutil.copyfile(os.path.join(img_path, img_name),
                                os.path.join(train_path, id2class[int(id)], img_name))

    #extract some training set as validation set
    train_class = os.listdir(train_path)
    for i in train_class:
        p = os.path.join(train_path, i)
        img_list = os.listdir(p)
        img_num = len(img_list)
        shuffle = np.random.permutation(img_num)
        for j in shuffle[:int(img_num * valid_proportion)]:
            shutil.move(os.path.join(p, img_list[j]),
                        os.path.join(valid_path, i, img_list[j]))


def mkdir(path):
    if os.path.exists(path):
        print('%s:存在' % path)
    else:
        os.mkdir(path)
        print('新建文件夹：%s' % path)


if __name__ == "__main__":
    main()