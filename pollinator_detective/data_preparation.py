"""Module providing functions preparing images for training"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, relative-beyond-top-level, wildcard-import
import os  # interact with the operating system
import shutil  # for copying files
import json  # manipulate json files
from PIL import Image  # Pillow image library
from tqdm import tqdm  # progress bar
from sahi.slicing import slice_coco  # slice COCO fortmat json files  https://github.com/obss/sahi/blob/main/sahi/slicing.py
from sahi.utils.file import load_json  # load COCO format json files
from .core import image_types, resize_isat, data_split, to_coco  # core functions


class Data4Training:
    """Prapre training data for stomata"""
    def __init__(self,
                 input_dir: str,
                 new_width: int = 1920,
                 new_height: int = 1080,
                 crop_ratio: float = 0.8,
                 r_train: float = 0.8,
                 sahi_slices: int = 5,
                 sahi_overlap_ratio: float = 0.7):
        self.input_dir = input_dir  # input directory
        self.new_width = new_width  # new width after resizing
        self.new_height = new_height  # new height after resizing
        self.crop_ratio = crop_ratio  # center crop ratio
        self.r_train = r_train  # ratio of training data
        self.sahi_slices = sahi_slices  # numbder of slice to crop from each direction, total patch will be the sqaured
        self.sahi_overlap_ratio = sahi_overlap_ratio  # fractional overlap in width of eachslice

    def ensemble_files(self, output_dir):
        """Ensemble all images and json files"""
        os.makedirs(output_dir)  # create output directory
        subfolders = os.listdir(self.input_dir)  # list all subfolders
        for subfolder in subfolders:
            file_names = sorted(os.listdir(os.path.join(self.input_dir, subfolder)), key=str.casefold)  # sort file names
            file_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in image_types + ['.json'])]  # image and json files only
            for file_name in file_names:
                file_path = os.path.join(self.input_dir, subfolder, file_name)  # file path
                output_path = os.path.join(output_dir, file_name)  # output path
                shutil.copy2(file_path, output_path)  # copy file for ensembling
        return None

    def center_crop(self, image_path, json_path):
        """Center crop the image and json file"""
        def adjust_bbox(bbox, crop_x1, crop_y1, crop_x2, crop_y2):
            """recompute bbox after center croppping"""
            x_min, y_min, x_max, y_max = bbox
            if x_max < crop_x1 or x_min > crop_x2 or y_max < crop_y1 or y_min > crop_y2:
                return None
            new_x_min = max(x_min, crop_x1) - crop_x1
            new_y_min = max(y_min, crop_y1) - crop_y1
            new_x_max = min(x_max, crop_x2) - crop_x1
            new_y_max = min(y_max, crop_y2) - crop_y1
            return [new_x_min, new_y_min, new_x_max, new_y_max]

        def adjust_segmentation(segmentation, crop_x1, crop_y1, crop_x2, crop_y2):
            """Recompute segmentation after center cropping"""
            adjusted_segmentation = []
            for point_x, point_y in segmentation:
                if crop_x1 <= point_x <= crop_x2 and crop_y1 <= point_y <= crop_y2:
                    adjusted_x = point_x - crop_x1
                    adjusted_y = point_y - crop_y1
                    adjusted_segmentation.append([adjusted_x, adjusted_y])
            return adjusted_segmentation if adjusted_segmentation else None

        image = Image.open(image_path)
        image_width, image_height = image.size

        crop_width = int(image_width * self.crop_ratio)
        crop_height = int(image_height * self.crop_ratio)
        crop_x1 = (image_width - crop_width) // 2
        crop_y1 = (image_height - crop_height) // 2
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height
        center_cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        center_cropped_image.save(image_path)

        with open(json_path, 'r', encoding='utf-8') as file:
            annotation_data = json.load(file)
        adjusted_objects = []
        for obj in annotation_data['objects']:
            bbox = obj['bbox']
            adjusted_bbox = adjust_bbox(bbox, crop_x1, crop_y1, crop_x2, crop_y2)
            segmentation = obj.get('segmentation', [])
            adjusted_segmentation = adjust_segmentation(segmentation, crop_x1, crop_y1, crop_x2, crop_y2)
            if adjusted_bbox and adjusted_segmentation:
                obj['bbox'] = adjusted_bbox
                obj['segmentation'] = adjusted_segmentation
                adjusted_objects.append(obj)
        annotation_data['objects'] = adjusted_objects
        if 'width' in annotation_data.get('info', {}):
            annotation_data['info']['width'] = crop_width
        if 'height' in annotation_data.get('info', {}):
            annotation_data['info']['height'] = crop_height
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(annotation_data, file)

    def data4training(self, sahi_mode=True):
        """generate data for training stomata instance segmentation from ISAT json files"""
        input_copy_dir = self.input_dir + ' - Copy'  # folder copy dir
        self.ensemble_files(input_copy_dir)  # create a copy
        resize_isat(input_copy_dir, new_width=self.new_width, new_height=self.new_height)  # resize images and annotations
        file_names = sorted(os.listdir(input_copy_dir), key=str.casefold)  # sort file names
        image_file_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in image_types)]  # image files only
        json_file_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in ['.json'])]  # json files only
        for idx, image_file_name in tqdm(enumerate(image_file_names), total=len(image_file_names)):
            image_path = os.path.join(input_copy_dir, image_file_name)
            json_path = os.path.join(input_copy_dir, json_file_names[idx])
            self.center_crop(image_path, json_path)
        output_name = 'Bug_mmdet'  # for object detection
        output_dir = os.path.join(os.path.split(input_copy_dir)[0], output_name)  # COCO json output dir
        train_dir = os.path.join(output_dir, 'train')  # COCO json train dir
        val_dir = os.path.join(output_dir, 'val')  # COCO json val dir
        data_split(input_copy_dir, output_dir, r_train=self.r_train)  # split train and val
        to_coco(train_dir, output_dir=os.path.join(train_dir, 'COCO.json'))  # convert train ISAT json files to COCO
        to_coco(val_dir, output_dir=os.path.join(val_dir, 'COCO.json'))  # same for val
        if sahi_mode:
            for folder in ['train', 'val']:
                slice_coco(
                    coco_annotation_file_path=train_dir.replace(folder, folder) + '//COCO.json',
                    image_dir=train_dir.replace(folder, folder),
                    output_coco_annotation_file_name="sliced",
                    ignore_negative_samples=True,
                    output_dir=train_dir.replace(folder, folder).replace(folder, folder + '_sliced'),
                    slice_height=int(self.new_height / self.sahi_slices / self.sahi_overlap_ratio),
                    slice_width=int(self.new_width / self.sahi_slices / self.sahi_overlap_ratio),
                    overlap_height_ratio=1 - self.sahi_overlap_ratio,
                    overlap_width_ratio=1 - self.sahi_overlap_ratio,
                    min_area_ratio=0.1,
                    verbose=False)
                coco_dict = load_json(train_dir.replace(folder, folder) + '//train_sliced.json')
                good_path_names = [coco_dict["images"][idx]['file_name'] for idx, _ in  enumerate(coco_dict["images"])]
                images_dir = train_dir.replace(folder, folder).replace(folder, folder + '_sliced')
                all_file_names = [name for name in os.listdir(images_dir) if any(name.lower().endswith(file_type) for file_type in image_types)]
                for name in all_file_names:
                    if name not in good_path_names:
                        os.remove(os.path.join(images_dir, name))
        return None
