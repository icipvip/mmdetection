import os
import shutil
import argparse


def move_files(src_folder_list, dest_folder):

    for src_folder in src_folder_list:
        if os.path.basename(src_folder) == 'fisheye-day-30072020':
            imagefolder = os.path.join(src_folder, 'images', 'train')
            labelfolder = os.path.join(src_folder, 'labels', 'train')
        else:
            imagefolder = os.path.join(src_folder, 'images')
            labelfolder = os.path.join(src_folder, 'labels')

        image_file_list = os.listdir(imagefolder)
        label_file_list = os.listdir(labelfolder)

        # move images
        dest_imagefolder = os.path.join(dest_folder, 'images')
        os.makedirs(dest_imagefolder, exist_ok=True)
        for image_file in image_file_list:
            image_path = os.path.join(imagefolder, image_file)
            shutil.move(image_path, dest_imagefolder)

        
        # move labels
        dest_labelfolder = os.path.join(dest_folder, 'labels')
        os.makedirs(dest_labelfolder, exist_ok=True)
        for label_file in label_file_list:
            label_path = os.path.join(labelfolder, label_file)
            shutil.move(label_path, dest_labelfolder)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="organize the data in train and test set")

    parser.add_argument("originpath", help="path to the original data folder provided by the organizers")
    parser.add_argument("finalpath", help="path to the final organized data folder")
    args = parser.parse_args()

    # make train set
    src_trainfolder_list  = [
        os.path.join(args.originpath, "fisheye-day-30072020"),
        os.path.join(args.originpath, "fisheye-night-30072020")
    ]
    dest_trainfolder = os.path.join(args.finalpath, "train")
    move_files(src_trainfolder_list, dest_trainfolder)

    # make test set
    src_testfolder_list = [
        os.path.join(args.originpath, "fisheye-day-test-30072020"),
        os.path.join(args.originpath, "fisheye-night-test-30072020")
    ]
    dest_testfolder = os.path.join(args.finalpath, "test")
    move_files(src_testfolder_list, dest_testfolder)
