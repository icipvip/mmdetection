## Installation

The code was tested with Python 3.6, CUDA 10.1, PyTorch 1.5.0 and torchvision 0.6.0.

1. Install filterpy, lap, pandas
~~~
pip install filterpy==1.4.5
pip install lap==0.4.0
pip install pandas
~~~
2. Install pycocotools.
~~~
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
~~~
3. Install pytorch 1.5.0, torchvision 0.6.0 compatible with cuda 10.1.
~~~
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
~~~
4. Install mmcv-full 1.1.1, with compatibility for pytorch and cuda version.
~~~
pip install mmcv-full==1.1.1+torch1.5.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
~~~
5. cd into the `teamzodiac` directory and Install mmdetection.
~~~
cd teamzodiac
pip install -e .
~~~

If you face any installation issues, please follow the [mmdetection installation page](https://mmdetection.readthedocs.io/en/latest/install.html)


## Test on an imagefolder
Suppose you want to run prediction on a folder of images, and store the results in submission format. Then cd into the `teamzodiac` folder of this codebase and run the following command:
~~~
cd teamzodiac
python vipcup/predict_imagefolder.py vipcup/finalmodel/epoch_2.pth  ${path_to_imagefolder}$  ${path_to_resultfolder}$
~~~
- replace  `${path_to_imagefolder}$`  with the complete path to the folder containing all the images.
- replace  `${path_to_resultfolder}$`  with the desired path to the folder where the results will be stored. The folder will be created if it doesn't exist.

This script makes prediction with a batch size of 1 and stores the result in submission format. It also prints the speed after every 50 prediction. 

After running the script, an example output should look like this:
~~~
Done image [50 / 393], speed: 19.3 img/s
Done image [100/ 393], speed: 19.5 img/s
Done image [150/ 393], speed: 19.6 img/s
Done image [200/ 393], speed: 19.6 img/s
Done image [250/ 393], speed: 19.6 img/s
Done image [300/ 393], speed: 19.6 img/s
Done image [350/ 393], speed: 19.6 img/s

Done image [393/ 393]
Overall speed: 19.6 img/s
~~~

## Reproduce

If you want to reproduce our training setup, you will need to organize the dataset.

### ICIP 2020 Dataset
We assume that the privoded dataset is formatted as follows:
~~~
ICIP2020-fisheye-dataset-30072020
    |--- fisheye-day-30072020
        |--- images
            |--- train
        |--- labels
            |--- train
    |--- fisheye-night-30072020
        |--- images
        |--- labels
    |--- fisheye-day-test-30072020
        |--- images
        |--- labels
    |--- fisheye-night-test-30072020
        |--- images
        |--- labels
~~~
To compactly organize the data in train and test set we use the script `organize_data.py`. Please cd into the `teamzodiac` directory of this codebase and run the follwoing command:

~~~
cd teamzodiac
python organize_data.py  ${path_to_icip_dataset}$  ./data
~~~
replace `${path_to_icip_dataset}$` with the complete path to where the `ICIP2020-fisheye-dataset-30072020` folder is located.

running this script should create a `data` directory inside the `teamzodiac` folder. It will also merge the training data for night and day under `data/train`. Similarly, it will also merge the test data for night and day under `data/test`. After running the script, the folder structure should be as below:
~~~
teamzodiac
|--- ...
|--- configs
|--- data
    |--- train
        |--- images
        |--- labels
    |--- test
        |--- images
        |--- labels
| ...
|--- tests
|-- tools
|-- vipcup
    |--- custom_gfocal.py
    |--- yolo_to_coco.py
    |--- ...
| ...
~~~

The ground truths are provided in yolo format. But our codebase uses coco style json for training. To convert yolo format labels to coco formatted json, we have defined a script `yolo_to_coco.py`. It takes three command line arguments as follows:
~~~
python vipcup/yolo_to_coco.py  ${path_to_imagefolder}$  ${path_to_labelfolder}$  ${path_to_final_json}$
~~~

Assuming you are in the `teamzodaic` directory, run the following command to create the json for train set:
~~~
python vipcup/yolo_to_coco.py data/train/images data/train/labels data/train/labels_cocoformat.json
~~~

Similarly, create the json for test set:
~~~
python vipcup/yolo_to_coco.py data/test/images data/test/labels data/test/labels_cocoformat.json
~~~
Notice that, creating the json files may take some time as it requires to actually read all the images to find their height and width.

After running the above two commands the folder structure should be like this:
~~~
teamzodiac
|--- ...
|--- configs
|--- data
    |--- train
        |--- images
        |--- labels
        |--- labels_cocoformat.json
    |--- test
        |--- images
        |--- labels
        |--- labels_cocoformat.json
| ...
|--- tests
|-- tools
|-- vipcup
    |--- custom_gfocal.py
    |--- yolo_to_coco.py
    |--- ...
| ...
~~~

### Training
At this point the data should be properly organized. Assuming you are in the `teamzodiac` directory, run the following command to reproduce our training results:
~~~
python tools/train.py vipcup/custom_gfocal.py --seed 1 --deterministic
~~~
This will train the network for 5 epochs. After each epoch, it runs evaluation on the test set and saves the checkpoint in `vipcup/logdir` folder. We found that the best result is achieved after epoch 2. This is the model we use for our final prediction and we save the weights in `vipcup/finalmodel` folder.

## Acknowledgement
We biuld on top of MMDetection codebase. Thanks MMDetection team for the wonderful open source project!

## Reference
- Generalized Focal Loss. https://arxiv.org/abs/2006.04388
- Simple Online and Realtime Tracking. https://arxiv.org/abs/1602.00763
