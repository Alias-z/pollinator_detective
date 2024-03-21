"""Module providing functions preparing images for training"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, relative-beyond-top-level, wildcard-import
import os  # interact with the operating system
import shutil  # for copying files
import json  # manipulate json files
import numpy as np  # Numpy
from PIL import Image  # Pillow image library
from tqdm import tqdm  # progress bar
from .core import image_types, resize_isat, data_split, to_coco  # core functions


class Data4Training:
    """Prapre training data for pollinator detection"""
    def __init__(self,
                 input_dir: str,
                 new_width: int = 1920,
                 new_height: int = 1080,
                 crop_size: tuple = (640, 640),
                 r_train: float = 0.8):
        self.input_dir = input_dir  # input directory
        self.new_width = new_width  # new width after resizing
        self.new_height = new_height  # new height after resizing
        self.r_train = r_train  # ratio of training data
        self.crop_size = crop_size  # cropping size around objects for images and json files

    def ensemble_files(self, output_dir: str) -> None:
        """Ensemble all images and json files"""
        os.makedirs(output_dir, exist_ok=True)  # create output directory
        subfolders = os.listdir(self.input_dir)  # list all subfolders
        for subfolder in subfolders:
            file_names = sorted(os.listdir(os.path.join(self.input_dir, subfolder)), key=str.casefold)  # sort file names
            file_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in image_types + ['.json'])]  # image and json files only
            for file_name in file_names:
                file_path = os.path.join(self.input_dir, subfolder, file_name)  # file path
                output_path = os.path.join(output_dir, file_name)  # output path
                shutil.copy2(file_path, output_path)  # copy file for ensembling
        return None

    def batch_crop(self, folder_path: str) -> None:
        """Batch crop images and json files to around the objects"""

        def load_image_isat(image_path: str, json_path: str) -> tuple:
            """Load images and their ISAT jsons"""
            image = Image.open(image_path)  # load the image
            with open(json_path, encoding='utf-8') as file:
                annotations = json.load(file)  # load and parse the JSON file
            return np.array(image), annotations

        def crop_image_isat(image_np: np.ndarray, annotations: dict, object_index: int) -> tuple:
            """Crop images and their jsons from the center object bboxes"""
            obj = annotations['objects'][object_index]  # use obj as object to differentiate from the built-in one
            bbox = obj['bbox']  # get the bbox
            center_x = (bbox[0] + bbox[2]) / 2  # calculate the center x
            center_y = (bbox[1] + bbox[3]) / 2  # same for the center y
            crop_x1 = max(center_x - self.crop_size[0] / 2, 0)  # x1
            crop_x2 = min(center_x + self.crop_size[0] / 2, image_np.shape[1])  # x2
            crop_y1 = max(center_y - self.crop_size[1] / 2, 0)  # y1
            crop_y2 = min(center_y + self.crop_size[1] / 2, image_np.shape[0])  # y2
            cropped_image = image_np[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]  # crop image based on padded bbox
            adjusted_annotations = {'info': annotations['info'], 'objects': []}  # remove objects outside the cropped region
            for obj in annotations['objects']:
                adjusted_bbox = [
                    max(obj['bbox'][0] - crop_x1, 0),
                    max(obj['bbox'][1] - crop_y1, 0),
                    min(obj['bbox'][2] - crop_x1, crop_x2 - crop_x1),
                    min(obj['bbox'][3] - crop_y1, crop_y2 - crop_y1)
                ]
                adjusted_bbox[0] = max(0, min(adjusted_bbox[0], cropped_image.shape[1]))
                adjusted_bbox[1] = max(0, min(adjusted_bbox[1], cropped_image.shape[0]))
                adjusted_bbox[2] = max(0, min(adjusted_bbox[2], cropped_image.shape[1]))
                adjusted_bbox[3] = max(0, min(adjusted_bbox[3], cropped_image.shape[0]))
                if adjusted_bbox[2] > adjusted_bbox[0] and adjusted_bbox[3] > adjusted_bbox[1]:
                    adjusted_segmentation = []
                    for point in obj['segmentation']:
                        adjusted_point = [max(point[0] - crop_x1, 0), max(point[1] - crop_y1, 0)]
                        if 0 <= adjusted_point[0] <= self.crop_size[0] and 0 <= adjusted_point[1] <= self.crop_size[1]:
                            adjusted_segmentation.append(adjusted_point)
                    adjusted_obj = obj.copy()
                    adjusted_obj['bbox'] = adjusted_bbox
                    adjusted_obj['segmentation'] = [adjusted_segmentation]
                    adjusted_annotations['objects'].append(adjusted_obj)
            return cropped_image, adjusted_annotations

        def save_images_isat(image_np: np.ndarray, annotations: dict, object_index: int, base_filename: str, output_folder: str) -> tuple:
            """Save the cropped images and ISAT json files"""
            os.makedirs(output_folder, exist_ok=True)  # make the output directory
            image_filename = f'{base_filename[0]}_object_{object_index}{base_filename[1]}'  # get image filename and extension
            annotations_filename = f"{base_filename[0]}_object_{object_index}.json"  # rename the new json file
            Image.fromarray(image_np).save(os.path.join(output_folder, image_filename))  # save the cropped image
            with open(os.path.join(output_folder, annotations_filename), 'w', encoding='utf-8') as file:
                annotations['info']['name'] = image_filename  # change the cropped image name
                json.dump(annotations, file)  # save the new json file
            return image_filename, annotations_filename

        def crop_save(image_path: str, json_path: str, output_folder: str) -> None:
            """Batch crop images based their ISAT json files"""
            image_np, annotations = load_image_isat(image_path, json_path)  # load images and json files
            base_filename = os.path.splitext(os.path.basename(image_path))  # get the image name and extension
            for object_index in range(len(annotations['objects'])):
                cropped_image, adjusted_annotations = crop_image_isat(image_np, annotations, object_index)  # get the cropped image and ISAT json
                save_images_isat(cropped_image, adjusted_annotations, object_index, base_filename, output_folder)
            os.remove(image_path); os.remove(json_path)  # noqa: remove the original image and json files
            return None

        file_names = sorted(os.listdir(folder_path), key=str.casefold)  # sort file names
        image_files = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in image_types)]  # image files only
        json_files = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in ['json'])]  # json files only
        for idx, image_file in tqdm(enumerate(image_files), total=len(image_files)):
            image_path = os.path.join(folder_path, image_file)
            json_path = os.path.join(folder_path, json_files[idx])
            crop_save(image_path, json_path, folder_path)
        print('note that after cropping, the number of images increase if multiple objects in one image')
        return None

    def ensemle_crop(self, if_resize_isat: bool = False) -> None:
        """Ensemle files and crop images and annoations around the detection objects"""
        output_dir = os.path.join(os.path.dirname(self.input_dir), 'Bug_data')  # output directory
        self.ensemble_files(output_dir)  # ensemble files to the output folder
        if if_resize_isat:
            resize_isat(output_dir, new_width=self.new_width, new_height=self.new_height)  # resize images and annotations
        self.batch_crop(output_dir)  # crop all images in the ensembled directory
        return None

    def data4training(self, sampled_image_paths: list) -> None:
        """Generate training data after sampled from each cluster"""
        sampled_image_paths = [os.path.normpath(path) for path in sampled_image_paths]  # normalize image paths
        sampled_json_paths = [os.path.splitext(path)[0] + '.json' for path in sampled_image_paths]  # normalized json paths
        output_dir = os.path.join(os.path.dirname(self.input_dir), 'Bug_data')  # output directory
        for file_name in set(os.listdir(output_dir)):
            file_path = os.path.normpath(os.path.join(output_dir, file_name))  # get the file path
            if file_path not in set(sampled_image_paths) and file_path not in set(sampled_json_paths):
                os.remove(file_path)  # only keep the selected files
        train_dir = os.path.join(output_dir, 'train')  # COCO json train dir
        val_dir = os.path.join(output_dir, 'val')  # COCO json val dir
        data_split(output_dir, output_dir, r_train=self.r_train)  # split train and val
        to_coco(train_dir, output_dir=os.path.join(train_dir, 'COCO.json'))  # convert train ISAT json files to COCO
        to_coco(val_dir, output_dir=os.path.join(val_dir, 'COCO.json'))  # same for val
        _ = [os.remove(path) for path in sampled_image_paths + sampled_json_paths]  # remove the copied files
        return None
