import os
import json
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='BDD100K to MOT format')
    parser.add_argument(
          "-i", "--img_dir",
          default="/path/to/bdd/image/",
          help="root directory of BDD image files",
    )
    parser.add_argument(
          "-l", "--label_dir",
          default="/path/to/bdd/label/",
          help="root directory of BDD label Json files",
    )
    parser.add_argument(
          "-s", "--save_path",
          default="/save/path",
          help="path to save MOT formatted label file",
    )
    parser.add_argument(
          "-H", "--height",
          default=720,
          help="height of an image",
    )
    parser.add_argument(
          "-W", "--width",
          default=1280,
          help="width of an image",
    )
    return parser.parse_args()


def bdd2mot_detection(img_dir, label_dir, save_img_dir, save_label_dir):
    with open(label_dir) as f:
        labels_json = json.load(f)
        for label_json in tqdm(labels_json):
            img_name = label_json['name']
            labels = label_json['labels']
            txt_string = ""
            for label in labels:
                category = label['category']
                if category not in attr_id_dict:
                    continue
                else:
                    x1 = label['box2d']['x1']
                    x2 = label['box2d']['x2']
                    y1 = label['box2d']['y1']
                    y2 = label['box2d']['y2']
                    width = x2 - x1
                    height = y2 - y1
                    x_center = (x1+x2)/2./args.width
                    y_center = (y1+y2)/2./args.height
                    width /= args.width
                    height /= args.height
                    identity = 0 ## no identity
                    # [class] [identity] [x_center] [y_center] [width] [height]
                    txt_string += "{} {} {} {} {} {}\n".format(attr_id_dict[category], identity, x_center, y_center, width, height)
                
                fn_label = os.path.join(save_label_dir, img_name[:-4]+'.txt')
                source_img = os.path.join(img_dir, img_name)
                target_img = os.path.join(save_img_dir, img_name)
                # print(fn_label, target_img, txt_string)
                with open(fn_label, 'w') as f:
                    f.write(txt_string)
                os.system('cp {} {}'.format(source_img, target_img))
            # break


if __name__ == '__main__':

    args = parse_arguments()

    ### for bdd detection dataset
    attr_dict = dict()
    attr_dict["categories"] = [
        {"supercategory": "none", "id": 0, "name": "person"},
        {"supercategory": "none", "id": 1, "name": "rider"},
        {"supercategory": "none", "id": 2, "name": "car"},
        {"supercategory": "none", "id": 3, "name": "truck"},
        {"supercategory": "none", "id": 4, "name": "bus"},
        {"supercategory": "none", "id": 5, "name": "train"},
        {"supercategory": "none", "id": 6, "name": "motor"},
        {"supercategory": "none", "id": 7, "name": "bike"},
        # {"supercategory": "none", "id": 8, "name": "other person"},
        # {"supercategory": "none", "id": 9, "name": "trailer"},
        # {"supercategory": "none", "id": 10, "name": "other vehicle"}
    ]

    attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}


    print('Loading and converting validation set...')
    # create BDD validation set tracking in MOT format
    val_img_dir = os.path.join(args.img_dir, 'val')
    val_label_dir = os.path.join(args.label_dir, 'bdd100k_labels_images_val.json')
    save_img_dir = os.path.join(args.save_path, 'images', 'val')#os.path.join(args.save_path, 'val')
    save_label_dir = os.path.join(args.save_path, 'labels_with_ids', 'val')
    if not os.path.exists(save_img_dir): os.makedirs(save_img_dir)
    if not os.path.exists(save_label_dir): os.makedirs(save_label_dir)
    bdd2mot_detection(val_img_dir, val_label_dir, save_img_dir, save_label_dir)
    
    print('Loading and converting training set...')
    # create BDD training set tracking in MOT format
    train_img_dir = os.path.join(args.img_dir, 'train')
    train_label_dir = os.path.join(args.label_dir, 'bdd100k_labels_images_train.json')
    save_img_dir = os.path.join(args.save_path, 'images', 'train')
    save_label_dir = os.path.join(args.save_path, 'labels_with_ids', 'train')
    if not os.path.exists(save_img_dir): os.makedirs(save_img_dir)
    if not os.path.exists(save_label_dir): os.makedirs(save_label_dir)
    bdd2mot_detection(train_img_dir, train_label_dir, save_img_dir, save_label_dir)