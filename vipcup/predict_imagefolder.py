import os
import argparse
import pandas as pd
from mmdet.apis import inference_detector, init_detector


def make_prediction(configpath, weightpath, imagefolder, resultfolder):
    os.makedirs(resultfolder, exist_ok=True)

    detector = init_detector(configpath, weightpath)
    imagenames = sorted(os.listdir(imagefolder))
    folderlength = len(imagenames)

    total_inf_time = 0
    for i, img_nm in enumerate(imagenames):
        img_pth = os.path.join(imagefolder, img_nm)
        result, inf_time = inference_detector(detector, img_pth)
        total_inf_time += inf_time

        basename = os.path.splitext(img_nm)[0]
        resulttxt = f"{basename}.txt"
        resultpath = os.path.join(resultfolder, resulttxt)

        result_df = pd.DataFrame(
            result[0],
            columns=["x1", "y1", "x2", "y2", "confidence"]
        )

        result_df["bbwidth"] = result_df["x2"] - result_df["x1"]
        result_df["bbheight"] = result_df["y2"] - result_df["y1"]
        result_df["category_name"] = "vehicle"
        result_df = result_df.astype({
            "category_name": str,
            "confidence": float,
            "x1": int,
            "y1": int,
            "bbwidth": int,
            "bbheight": int
        })

        result_df.to_csv(
            resultpath,
            columns=["category_name", "confidence", "x1", "y1", "bbwidth", "bbheight"],
            sep=" ",
            float_format="%.2f",
            header=False,
            index=False
        )

        if (i + 1) % 50 == 0:
            speed = (i + 1) / total_inf_time
            print(f"Done image [{i + 1:<3}/ {folderlength}], speed: {speed:.1f} img/s")

    print("\n")
    print(f"Done image [{i + 1:<3}/ {folderlength}]")
    print(f"Overall speed: {(i + 1) / total_inf_time:.1f} img/s")
    
    return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="predict the result for a folder of images in txt format")

    parser.add_argument("configpath", help="path to the config file")
    parser.add_argument("weightpath", help="path to the network weights")
    parser.add_argument("imagefolder", help="path to the folder containing test images")
    parser.add_argument("resultfolder", help="path to the folder that will contain predicted results")
    args = parser.parse_args()

    make_prediction(args.configpath, args.weightpath, args.imagefolder, args.resultfolder)
