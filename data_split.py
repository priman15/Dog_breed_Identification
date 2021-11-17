import os
from shutil import copy2 
import random

def create_train_test_list(target_dir):
    random.seed(42)
    image_list = [os.path.join(target_dir, img_path) for img_path in os.listdir(target_dir)]
    train_split = int(0.8 * len(image_list))
    train_image_list = random.sample(image_list, train_split)
    test_image_list = list(set(image_list).difference(set(train_image_list)))
    return train_image_list, test_image_list




def copy_images_to_file(img_path_list, target_dir, train=True):
        if train:
            split_dir = "train"
        else:
            split_dir = "test"

        # Copy images 
        for image_path in img_path_list:
            image_file_name = os.path.split(image_path)[-1]
            dest_path = os.path.join(target_dir, split_dir, image_dir, image_file_name)
            print(f"Copying: {image_path} to {dest_path}")
            copy2(image_path, dest_path)

if __name__=="__main__":            
    cmd="ls model_test_images/"
    os.system(cmd)
    
    
    data_dir = "model_test_images"
    target_dir = "model_test_images_split"
    cmd="mkdir model_test_images_split/train/NotDog"
    os.system(cmd)
    cmd="mkdir model_test_images_split/test/NotDog"
    os.system(cmd)
    for image_dir in os.listdir(data_dir):
        for split_dir in ["train", "test"]:
            os.makedirs(os.path.join(target_dir, split_dir, image_dir), exist_ok=True)
    
        # Make training and test lists of target images
        train_image_list, test_image_list = create_train_test_list(os.path.join(data_dir, image_dir))
    
        # Copy training images
        copy_images_to_file(img_path_list=train_image_list, 
                            target_dir=target_dir, 
                            train=True)
    
        # Copy testing images
        copy_images_to_file(img_path_list=test_image_list, 
                            target_dir=target_dir, 
                            train=False)
    