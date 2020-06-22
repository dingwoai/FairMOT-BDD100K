import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='generate path')
    parser.add_argument(
          "-it", "--img_train_dir",
          default="/path/to/MOT/images/train",
          help="root directory of BDD image files",
    )
    parser.add_argument(
          "-iv", "--img_val_dir",
          default="/path/to/MOT/images/val",
          help="root directory of BDD image files",
    )
    parser.add_argument(
          "-s", "--save_dir",
          default="/save/path",
          help="path to save generated file",
    )
    parser.add_argument(
          "-n", "--save_name",
          default="bdd100k",
          help="name.train or name.val",
    )
    return parser.parse_args()


def gen_path_file(img_dir, save_dir, fname):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    img_files = os.listdir(img_dir)
    with open(os.path.join(save_dir, fname), 'w') as f:
        for txt_string in img_files: 
            f.write(os.path.join(img_dir, txt_string)+'\n')


if __name__ == '__main__':
    args = parse_arguments()
    img_train_dir = args.img_train_dir
    img_val_dir = args.img_val_dir
    save_dir = args.save_dir
    save_name = args.save_name

    gen_path_file(img_train_dir, save_dir, save_name+'.train')
    gen_path_file(img_val_dir, save_dir, save_name+'.val')