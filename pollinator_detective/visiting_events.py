"""Module providing functions for counting pollinator visiting events"""
# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import os  # interact with the operating system
import shutil  # for copying and pasting files
import cv2  # OpenCV
import torch  # PyTorch
import numpy as np  # NumPy
import pandas as pd  # for Excel sheet
from matplotlib import pyplot as plt
from tqdm import tqdm
from mmdet.utils import register_all_modules as mmdet_utils_register_all_modules  # register mmdet modules
from mmdet.apis import init_detector as mmdet_apis_init_detector  # initialize mmdet model
from mmdet.apis import inference_detector as mmdet_apis_inference_detector  # mmdet inference detector
from mmpretrain import ImageClassificationInferencer as classifier  # classify bugs
from .core import device, image_types, video_types, imread_rgb


class SegColors:
    """Define the segmentation class names, colors and their one-hot encoding"""
    def __init__(self, class_name, mask_rgb, class_encoding):
        self.class_name = class_name
        self.mask_rgb = mask_rgb
        self.class_encoding = class_encoding


Bug = [SegColors('Bumblebees', [100, 0, 0], 0),
       SegColors('Flies', [100, 67, 0], 1),
       SegColors('Honeybees', [100, 100, 0], 2),
       SegColors('Hoverfly_A', [33, 0, 50], 3),
       SegColors('Hoverfly_B', [100, 0, 50], 4),
       SegColors('Wildbees', [33, 100, 100], 5)]


class VisitingEvents():
    """Counting pollinator visiting events"""
    def __init__(self,
                 input_dir: str,
                 output_name: str = 'Results',
                 batch_size: int = 100,
                 detector_config_path: str = None,
                 detector_weight_path: str = None,
                 bumblebees_threshold: float = 0.9,
                 flies_threshold: float = 0.9,
                 honeybees_threshold: float = 0.9,
                 hoverfly_a_threshold: float = 0.9,
                 hoverfly_b_threshold: float = 0.9,
                 wildbees_threshold: float = 0.9,
                 ):
        self.input_dir = os.path.normpath(input_dir)  # input directory
        self.output_name = output_name  # output folder name
        self.batch_size = batch_size  # inference batch size
        self.detector_config_path = detector_config_path  # object detection config path
        self.detector_weight_path = detector_weight_path  # object detection weight path
        self.bumblebees_threshold = bumblebees_threshold  # object detection threshold for bumblebees
        self.flies_threshold = flies_threshold  # object detection threshold for flies
        self.honeybees_threshold = honeybees_threshold  # object detection threshold for honeybees
        self.hoverfly_a_threshold = hoverfly_a_threshold  # object detection threshold for hoverfly_a
        self.hoverfly_b_threshold = hoverfly_b_threshold  # object detection threshold for hoverfly_b
        self.wildbees_threshold = wildbees_threshold  # object detection threshold for wildbees

    def get_video_paths(self):
        """Get the paths of videos under a given folder"""
        file_names = sorted(os.listdir(self.input_dir), key=str.casefold)
        file_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in video_types)]
        file_paths = [os.path.join(self.input_dir, file_name) for file_name in file_names]
        return file_paths

    def extract_frames(self, video_path, frame_rate=30):
        """Extract all frames from a video"""
        video_name, _ = os.path.splitext(video_path)
        os.makedirs(video_name, exist_ok=True)
        video = cv2.VideoCapture(video_path)
        filename = os.path.splitext(os.path.basename(video_path))[0]
        frame_duration = 1000 // frame_rate  # The time between frames in milliseconds
        milliseconds, frame, frame_paths = 0, 1, []
        while True:
            frame_number = int((video.get(cv2.CAP_PROP_FPS) / 1000) * milliseconds)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = video.read()
            if success:
                frame_path = os.path.join(video_name, f"{filename} frame{frame:04d}.png")
                frame_paths.append(frame_path)
                cv2.imwrite(frame_path, image)
                milliseconds += frame_duration
                frame += 1
            else:
                break
        return video_name

    def batch_extractor(self):
        """Extract frames from all videos"""
        video_paths = self.get_video_paths()
        video_names = [self.extract_frames(video_path, frame_rate=30) for video_path in video_paths]
        return video_names

    def bug_detector(self, folder_path):
        """Detect pollinators for all frames under a given folder"""
        output_dir = os.path.join(os.path.split(folder_path)[0], f'{os.path.split(folder_path)[1]} predictions')
        os.makedirs(output_dir, exist_ok=True)
        frame_names = [name for name in os.listdir(folder_path) if any(name.lower().endswith(file_type) for file_type in image_types)]
        mmdet_utils_register_all_modules(init_default_scope=False)  # initialize mmdet scope
        detector = mmdet_apis_init_detector(self.detector_config_path, self.detector_weight_path, device=device)  # initialize a detector from config file
        n_bugs = []
        for idx, frame_name in tqdm(enumerate(frame_names), total=len(frame_names)):
            image = imread_rgb(os.path.join(folder_path, frame_name))
            prediction = mmdet_apis_inference_detector(detector, image)
            n_bumblebees, n_flies, n_honeybees, n_hoverfly_a, n_hoverfly_b, n_wildbees = 0, 0, 0, 0, 0, 0
            for bug_idx, bug in enumerate(Bug):
                threshold = getattr(self, f"{bug.class_name.lower()}_threshold")
                indices = torch.where((prediction.pred_instances.labels == bug_idx) & (prediction.pred_instances.scores > threshold))[0]
                if len(indices) > 0:
                    if bug.class_name == 'Bumblebees':
                        n_bumblebees += len(indices)
                    elif bug.class_name == 'Flies':
                        n_flies += len(indices)
                    elif bug.class_name == 'Honeybees':
                        n_honeybees += len(indices)
                    elif bug.class_name == 'Hoverfly_A':
                        n_hoverfly_a += len(indices)
                    elif bug.class_name == 'Hoverfly_B':
                        n_hoverfly_b += len(indices)
                    elif bug.class_name == 'Wildbees':
                        n_wildbees += len(indices)
                    bboxes = [prediction.pred_instances.bboxes[ind].cpu().numpy() for ind in indices]
                    bboxes = [np.array(bbox, dtype=np.int32) for bbox in bboxes]
                    locations = [(x_1, y_1, x_2, y_2) for x_1, y_1, x_2, y_2 in bboxes]
                    scores = [prediction.pred_instances.scores[ind].cpu().numpy().item() for ind in indices]
                    for idx, location in enumerate(locations):
                        x_1, y_1, x_2, y_2 = location
                        cv2.rectangle(image, (x_1, y_1), (x_2, y_2), bug.mask_rgb, 4)
                        cv2.putText(image, f'{bug.class_name}, p={round(scores[idx], 2)}', (x_1, y_1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bug.mask_rgb, 2)
            cv2.imwrite(os.path.join(output_dir, frame_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # export the image
            n_bugs.append([n_bumblebees, n_flies, n_honeybees, n_hoverfly_a, n_hoverfly_b, n_wildbees])
        return np.array(n_bugs).T.tolist(), frame_names

    def count_visits(self, frames, absence_window=2):
        """
        Analyze a given pollinator visits and their durations through video frames.

        Rules:
        1. Initialization: Initially, there is no event, n = 0.
        2. Event Start: An event starts when n > 0.
        3. Absence Window: Once an event starts, an absence window of a fixed number of frames (default 2 frames) is set to check if the event ends.
        4. Event Continuation: An event continues if any frame within the absence window has n that is the same or greater than the starting n value.
           If this happens, the absence window extends starting from that frame.
        5. Event End: An event ends at the last frame during or before the absence window where n is the same as the starting n. If all n values in the absence window are smaller, the event ends.
        6. Overlapping Events: During any event, there could be overlapping events, which occur when n is higher than the starting n. Each overlapping event has its own absence window.
        7. Event End & Overlapping: When one event ends, the n value for overlapping events reduces by 1.
        8. Multiple Concurrent Events: When n > 1, multiple events occur concurrently. Each concurrent event follows the same lifecycle rules as a single event.

        Parameters:
        - frames (list): A list of integers representing the number of pollinators in each frame.
        - absence_window (int): The size of the absence window to check for event ending.

        Returns:
        - List of tuples: Each tuple contains the starting and ending frame of each event.
        """
        ongoing_events, completed_events = [], []  # t store ongoing events and completed events
        for frame_num, n_bug in enumerate(frames):
            # check for event continuation and ending
            new_ongoing_events = []
            for start_n, start_frame, end_frame in ongoing_events:
                if frame_num <= end_frame:
                    # check if the event should continue
                    if n_bug >= start_n:
                        # extend the absence window
                        new_ongoing_events.append([start_n, start_frame, frame_num + absence_window])
                        n_bug -= start_n  # reduce n for overlapping events
                    else:
                        completed_events.append((start_frame, end_frame - absence_window))
                else:
                    completed_events.append((start_frame, end_frame - absence_window))
            ongoing_events = new_ongoing_events
            # check for new events
            while n_bug > 0:
                # start a new event
                ongoing_events.append([n_bug, frame_num, frame_num + absence_window])
                n_bug -= n_bug
        # check for events that are still ongoing at the end of the frame list
        for start_n, start_frame, end_frame in ongoing_events:
            if end_frame >= len(frames):
                completed_events.append((start_frame, "ongoing"))
        return completed_events

    def counts2excel(self, folder_path, absence_window_range):
        """Count visiting events for all pollinator types and output to an Excel file"""
        if absence_window_range is None:
            absence_window_range = [2]
        counts, frame_names = self.bug_detector(folder_path)
        frame_numbers = [number + 1 for number, _ in enumerate(frame_names)]
        for absence_window in absence_window_range:
            all_events = []  # to collect all events
            for count in counts:
                events = [""] * len(count)
                visits = self.count_visits(count, absence_window=absence_window)
                for index_i, (start, end) in enumerate(visits, 1):  # index_i starts from 1
                    for index_j in range(start, end + 1):  # +1 to include the end index
                        events[index_j] = f"Visits {index_i}"
                all_events.append(events)
            result = {'frame_names': frame_names,
                      'frame_numbers': frame_numbers,
                      f'{Bug[0].class_name} counts': counts[0],
                      f'{Bug[0].class_name} visits': all_events[0],
                      f'{Bug[1].class_name} counts': counts[1],
                      f'{Bug[1].class_name} visits': all_events[1],
                      f'{Bug[2].class_name} counts': counts[2],
                      f'{Bug[2].class_name} visits': all_events[2],
                      f'{Bug[3].class_name} counts': counts[3],
                      f'{Bug[3].class_name} visits': all_events[3],
                      f'{Bug[4].class_name} counts': counts[4],
                      f'{Bug[4].class_name} visits': all_events[4],
                      f'{Bug[5].class_name} counts': counts[5],
                      f'{Bug[5].class_name} visits': all_events[5],
                      }
            result = pd.DataFrame(data=result)  # collect results in a pd dataframe for exporting to an Excel sheet
            excel_filename = f"{folder_path} predictions//Absence Window {absence_window}.xlsx"
            result.to_excel(excel_filename, index=False)
        return None

    def batch_predict(self, absence_window_range=None):
        """Batch predit for all video frames (subfolders) under a given parent folder"""
        if absence_window_range is None:
            absence_window_range = [2]
        folder_names = [name for name in os.listdir(self.input_dir) if not any(name.lower().endswith(file_type) for file_type in video_types + ['predictions'])]
        folder_dirs = [os.path.join(self.input_dir, folder_name) for folder_name in folder_names]
        for folder_dir in tqdm(folder_dirs, total=len(folder_dirs)):
            self.counts2excel(folder_dir, absence_window_range=absence_window_range)
        return None




################################################################ legacy #################################################################


def bug_classifier(video_name, config='Models//EVA///mmpretrain_bugs.py', weights='Models//EVA//best_accuracy_top1_epoch_105.pth', sort2train=True):
    """classify bugs for each video frame under a given folder"""
    model = classifier(model=config, pretrained=weights)
    image_paths = [os.path.join(video_name, path) for path in os.listdir(video_name) if any(path.endswith(file_type) for file_type in image_types)]
    if sort2train:
        results = model(image_paths, batch_size=1)
        scores = [result['pred_score'] for result in results]
        classes = [result['pred_class'] for result in results]
        class_dirs = []
        for class_name in classes:
            os.makedirs(os.path.join(video_name, class_name), exist_ok=True)
            class_dirs.append(os.path.join(video_name, class_name))
        for idx, image_path in enumerate(image_paths):
            source = image_path
            if scores[idx] > 0.8:
                destination = os.path.join(class_dirs[idx], os.path.basename(image_path))
            else:
                os.makedirs(os.path.join(video_name, 'No_insects'), exist_ok=True)
                destination = os.path.join(video_name, 'No_insects', os.path.basename(image_path))
            shutil.copyfile(source, destination) 
            os.remove(source)
    else:
        results = model(image_paths, batch_size=1, show_dir=video_name + ' Results')
        scores = [result['pred_score'] for result in results]
        classes = [result['pred_class'] for result in results]
    return scores, classes


def find_visiting_events(frames, absence_window=2):
    """find all visiting events
    1. Initially, we are not in an event.
    2. Whenever we encounter a frame that is not "No_insects", we start an event.
    3. Whenever we encounter a frame that is "No_insects", we look ahead in the next "absence_window" frames:
           If we find any frame that is not "No_insects", we continue with the current event. If not, the event ends.
    4. We will use "nothing" to denote frames that are not part of an event, and "Visiting X" to denote frames that are part of visiting event X.
    """
    events = [""] * len(frames)
    current_event, in_event = 0, False
    for i in range(len(frames)):
        # If we are not in an event and we encounter insects
        if not in_event and frames[i] != 'No_insects':
            in_event = True
            current_event += 1
            events[i] = f"Visiting {current_event}"
        elif in_event and frames[i] == 'No_insects':
            # Look ahead in the next "absence_window" frames
            if any(frame != 'No_insects' for frame in frames[i:min(i+absence_window, len(frames))]):
                # If we find insects in the next "absence_window" frames, continue with the current event
                events[i] = f"Visiting {current_event}"
            else:
                # If we don't find insects in the next "absence_window" frames, the event ends
                in_event = False
        elif in_event:
            events[i] = f"Visiting {current_event}"
    return events