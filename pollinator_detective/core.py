"""Module providing core functions"""
# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import os  # interact with the operating system
import shutil  # for copy files
import glob  # Unix style pathname pattern expansion
import json  # manipulate json files
import random  # generate random numbers
import cv2  # OpenCV
import torch  # PyTorch
import numpy as np  # NumPy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # use GPU if available
image_types = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.ico', '.jfif', '.webp']  # supported image types
video_types = ['.avi', '.mp4']  # supported video types


def imread_rgb(image_dir: str):
    """cv2.imread + BRG2RGB """
    image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
    return image


def get_json_paths(json_dir):
    """under a given folder, get all json files paths"""
    return glob.glob(os.path.join(json_dir, '*.json'))


def resize_isat(input_dir, new_width=1280, new_height=960):
    """resize ISAT json files and images in a given folder"""
    json_paths = get_json_paths(input_dir)
    for json_path in json_paths:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # load the json data
        old_width = data['info']['width']
        old_height = data['info']['height']
        width_ratio = new_width / old_width
        height_ratio = new_height / old_height
        data['info']['width'] = new_width  # assign resized width
        data['info']['height'] = new_height  # resized height
        for obj in data['objects']:
            for point in obj['segmentation']:
                point[0] = int(point[0] * width_ratio)  # update width
                point[1] = int(point[1] * height_ratio)  # update height
            obj['bbox'][0] = int(obj['bbox'][0] * width_ratio)  # update bbox x1
            obj['bbox'][1] = int(obj['bbox'][1] * height_ratio)  # bbox y1
            obj['bbox'][2] = int(obj['bbox'][2] * width_ratio)  # bbox x2
            obj['bbox'][3] = int(obj['bbox'][3] * height_ratio)  # bbox y2
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)  # save the sorted json data
    image_names = [name for name in os.listdir(input_dir) if any(name.endswith(file_type) for file_type in image_types)]  # get images names
    image_dirs = [os.path.join(input_dir, file_name) for file_name in image_names]  # get image paths
    for image_dir in image_dirs:
        image = imread_rgb(image_dir)  # load images
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)  # resize images
        cv2.imwrite(image_dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # nolint: replace image files
    return None


def data_split(images_dir, output_dir, r_train=0.8):
    """split dataset into train and val with defined ratio"""
    random.seed(42); np.random.seed(42)  # noqa: set seed
    file_names = sorted(os.listdir(images_dir), key=str.casefold)
    file_names = [name for name in file_names if any(name.endswith(file_type) for file_type in image_types)]  # image files only
    train_size = int(len(file_names) * r_train)  # training size
    validation_size = int(len(file_names) * (1 - r_train))  # validation size
    file_names_shuffle = file_names.copy()  # prevent changing in place
    random.shuffle(file_names_shuffle)  # random shuffle file names
    train_names = file_names_shuffle[:train_size]  # file names for training
    val_names = file_names_shuffle[train_size:train_size + validation_size]
    print(f'train size={train_size}, validation size={validation_size}')
    destination_train = os.path.join(output_dir, 'train'); os.makedirs(destination_train, exist_ok=True)  # noqa
    destination_val = os.path.join(output_dir, 'val'); os.makedirs(destination_val, exist_ok=True)  # noqa
    for name in train_names:
        source = os.path.join(images_dir, name)
        destination = os.path.join(destination_train, name)
        shutil.copy2(source, destination)  # paste train images
        name_json = os.path.splitext(name)[0] + '.json'
        source = os.path.join(images_dir, name_json)
        destination = os.path.join(destination_train, name_json)
        shutil.copy2(source, destination)  # paste train jsons
    for name in val_names:
        source = os.path.join(images_dir, name)
        destination = os.path.join(destination_val, name)
        shutil.copy2(source, destination)  # paste validation images
        name_json = os.path.splitext(name)[0] + '.json'
        source = os.path.join(images_dir, name_json)
        destination = os.path.join(destination_val, name_json)
        shutil.copy2(source, destination)  # paste validation jsons
    return None


def to_coco(input_dir, output_dir):
    """convert ISAT format to MSCOCO format (modified from TOCOCO.py)"""
    coco_anno = {}
    coco_anno['info'] = {}
    coco_anno['info']['description'] = 'Nectar seekers'
    coco_anno['info']['year'] = None
    coco_anno['info']['contributor'] = 'Ursina Baselgia'
    coco_anno['images'] = []
    coco_anno['annotations'] = []
    coco_anno['categories'] = []
    categories_dict = {}
    json_paths = get_json_paths(input_dir)
    for file_index, json_path in enumerate(json_paths):
        with open(json_path, encoding='utf-8') as file:
            dataset = json.load(file)
            info = dataset.get('info', {})
            description = info.get('description', '')
            if not description.startswith('ISAT'):
                continue
            img_name = info.get('name', '')
            width = info.get('width', None)
            height = info.get('height', None)
            objects = dataset.get('objects', [])
            coco_image_info = {}
            coco_image_info['license'] = None
            coco_image_info['url'] = None
            coco_image_info['file_name'] = img_name
            coco_image_info['height'] = height
            coco_image_info['width'] = width
            coco_image_info['date_captured'] = None
            coco_image_info['id'] = file_index
            coco_anno['images'].append(coco_image_info)
            objects_groups = [obj.get('group', 0) for obj in objects]
            objects_groups.sort()
            objects_groups = set(objects_groups)
            for group in objects_groups:
                objs_with_group = [obj for obj in objects if obj.get('group', 0) == group]
                cats = [obj.get('category', 'unknow') for obj in objs_with_group]
                cats = set(cats)
                for cat in cats:
                    if cat not in categories_dict:
                        categories_dict[cat] = len(categories_dict)
                    category_index = categories_dict.get(cat)
                    objs_with_cat = [obj for obj in objs_with_group if obj.get('category', 0) == cat]
                    crowds = [obj.get('iscrowd', 'unknow') for obj in objs_with_group]  # noqa
                    crowds = set(crowds)
                    for crowd in crowds:
                        objs_with_crowd = [obj for obj in objs_with_cat if obj.get('iscrowd', 0) == crowd]
                        coco_anno_info = {}
                        coco_anno_info['iscrowd'] = crowd
                        coco_anno_info['image_id'] = file_index
                        coco_anno_info['image_name'] = img_name
                        coco_anno_info['category_id'] = category_index
                        coco_anno_info['id'] = len(coco_anno['annotations'])
                        coco_anno_info['segmentation'] = []
                        coco_anno_info['area'] = 0.
                        coco_anno_info['bbox'] = []
                        for obj in objs_with_crowd:
                            segmentation = obj.get('segmentation', [])
                            area = obj.get('area', 0)
                            bbox = obj.get('bbox', [])
                            if bbox is None:
                                segmentation_nd = np.array(segmentation)
                                bbox = [min(segmentation_nd[:, 0]), min(segmentation_nd[:, 1]),
                                        max(segmentation_nd[:, 0]), max(segmentation_nd[:, 1])]
                                del segmentation_nd
                            segmentation = [e for p in segmentation for e in p]
                            if bbox != []:
                                if coco_anno_info['bbox'] == []:
                                    coco_anno_info['bbox'] = bbox
                                else:
                                    bbox_tmp = coco_anno_info['bbox']
                                    bbox_tmp = [min(bbox_tmp[0], bbox[0]), min(bbox_tmp[1], bbox[1]),
                                                max(bbox_tmp[2], bbox[2]), max(bbox_tmp[3], bbox[3])]
                                    coco_anno_info['bbox'] = bbox_tmp
                            coco_anno_info['segmentation'].append(segmentation)
                            if area is not None:
                                coco_anno_info['area'] += float(area)
                        # (xmin, ymin, xmax, ymax) 2 (xmin, ymin, w, h)
                        bbox_tmp = coco_anno_info['bbox']
                        coco_anno_info['bbox'] = [bbox_tmp[0], bbox_tmp[1],
                                                  bbox_tmp[2] - bbox_tmp[0], bbox_tmp[3] - bbox_tmp[1]]
                        coco_anno['annotations'].append(coco_anno_info)
        os.remove(json_path)
    categories_dict = sorted(categories_dict.items(), key=lambda x: x[0])  # sort categories by keys
    new_category_ids = {category: index for index, (category, old_id) in enumerate(categories_dict)}  # create a new dictionary mapping class names to new ids
    for annotation in coco_anno['annotations']:
        old_category_id = annotation['category_id']
        for category, new_id in new_category_ids.items():
            if old_category_id == categories_dict[new_category_ids[category]][1]:  # match old id
                annotation['category_id'] = new_id
    coco_anno['categories'] = [{"name": name, "id": id, "supercategory": None} for name, id in new_category_ids.items()]
    with open(output_dir, 'w', encoding='utf-8') as file:
        json.dump(coco_anno, file)
    return None


def ocr_regions(image):
    """crop the image to the OCR region"""
    cropped_region = image[1050:1080, 700:1200]
    return cropped_region
