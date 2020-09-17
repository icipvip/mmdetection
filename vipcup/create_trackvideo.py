import os
import glob
import argparse
import cv2

from mmdet.apis import inference_detector, init_detector
from sort import Sort

def make_trackvideo(configpath, weightpath, imagefolder, videofolder):
    os.makedirs(videofolder, exist_ok=True)

    imagenames = sorted(os.listdir(imagefolder))

    # find video names
    videonames = set()
    for img_nm in imagenames:
        video = img_nm.rsplit('_', 1)[0]
        videonames.add(video)
    print(f"video files: {videonames}\n")

    detector = init_detector(configpath, weightpath)
    # make video files
    for vdo_id, vdo_nm in enumerate(videonames):
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        fps = 12
        shape = (800, 800)
        vdo_pth = os.path.join(videofolder, f"{vdo_nm}.avi")
        result_video = cv2.VideoWriter(vdo_pth, fourcc, fps, shape)
        
        image_paths = glob.glob(os.path.join(imagefolder, f"{vdo_nm}*"))
        image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(x)[0].rsplit('_', 1)[1]))

        tracker = Sort()
        for img_id, img_pth in enumerate(image_paths):
            image = cv2.imread(img_pth)
            result, inf_time = inference_detector(detector, image)
            detected_boxes = result[0] 
            tracked_boxes = tracker.update(detected_boxes)

            # draw the result on image
            bboxes, track_ids = tracked_boxes[:, :-1], tracked_boxes[:, -1]
            
            for box, trk_id in zip(bboxes, track_ids):
                x1, y1, x2, y2 = box.astype("int")
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                cx, cy = (x1+x2)//2, (y1+y2)//2
                text = f"{trk_id.astype(int)}"
                cv2.putText(image, text, (cx-10, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
            
            image = cv2.resize(image, shape) 
            result_video.write(image)
            print(f"video {vdo_id} frame {img_id} drawing done.")
        
        result_video.release()
        print(f"\n{vdo_nm}.avi created!!\n")
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="make tracking video for a folder of images")

    parser.add_argument("configpath", help="path to the detector config file")
    parser.add_argument("weightpath", help="path to the detection network weights")
    parser.add_argument("imagefolder", help="path to the folder containing images")
    parser.add_argument("videofolder", help="path to the folder that will contain tracking videos")
    args = parser.parse_args()

    make_trackvideo(args.configpath, args.weightpath, args.imagefolder, args.videofolder)
