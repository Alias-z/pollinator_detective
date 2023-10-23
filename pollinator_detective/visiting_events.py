"""Module providing functions for counting pollinator visiting events"""
# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import os  # interact with the operating system
import cv2  # OpenCV
import numpy as np  # NumPy
import pandas as pd  # for Excel sheet
from tqdm import tqdm
import pytesseract; pytesseract.pytesseract.tesseract_cmd = 'C://Programs//Tesseract-OCR//tesseract.exe'   # noqa: for OCR
from mmdet.utils import register_all_modules as mmdet_utils_register_all_modules  # register mmdet modules
from sahi import AutoDetectionModel  # loads a DetectionModel from given path
from sahi.predict import get_sliced_prediction  # detect bugs from frame slices
from .core import device, image_types, video_types, imread_rgb  # import core functions


class SegColors:
    """Define the segmentation class names, colors and their one-hot encoding"""
    def __init__(self, class_name, mask_rgb, class_encoding):
        self.class_name = class_name
        self.mask_rgb = mask_rgb
        self.class_encoding = class_encoding


Bug = [SegColors('Bumblebees', [255, 0, 0], 0),
       SegColors('Flies', [0, 0, 255], 1),
       SegColors('Honeybees', [255, 165, 0], 2),
       SegColors('Hoverfly_A', [165, 42, 42], 3),
       SegColors('Hoverfly_B', [64, 224, 208], 4),
       SegColors('Wildbees', [255, 0, 255], 5),
       SegColors('Others', [255, 223, 0], 6)]

bug_color = {bug.class_name: bug.mask_rgb for bug in Bug}


class VisitingEvents():
    """Counting pollinator visiting events"""
    def __init__(self,
                 input_dir: str,
                 save_frames: bool = False,
                 detector_config_path: str = None,
                 detector_weight_path: str = None,
                 new_width: int = 1920,
                 new_height: int = 1080,
                 sahi_slices: int = 5,
                 sahi_overlap_ratio: float = 0.7,
                 bumblebees_threshold: float = 0.9,
                 flies_threshold: float = 0.9,
                 honeybees_threshold: float = 0.9,
                 hoverfly_a_threshold: float = 0.9,
                 hoverfly_b_threshold: float = 0.9,
                 wildbees_threshold: float = 0.9,
                 others_threshold: float = 0.9,
                 ):
        self.input_dir = os.path.normpath(input_dir)  # input directory
        self.save_frames = save_frames  # wether to save video frame in a separate folder
        self.detector_config_path = detector_config_path  # object detection config path
        self.detector_weight_path = detector_weight_path  # object detection weight path
        self.new_width = new_width  # new width after resizing
        self.new_height = new_height  # new height after resizing
        self.sahi_slices = sahi_slices  # numbder of slice to crop from each direction, total patch will be the sqaured
        self.sahi_overlap_ratio = sahi_overlap_ratio  # fractional overlap in width of eachslice
        self.bumblebees_threshold = bumblebees_threshold  # object detection threshold for bumblebees
        self.flies_threshold = flies_threshold  # object detection threshold for flies
        self.honeybees_threshold = honeybees_threshold  # object detection threshold for honeybees
        self.hoverfly_a_threshold = hoverfly_a_threshold  # object detection threshold for hoverfly_a
        self.hoverfly_b_threshold = hoverfly_b_threshold  # object detection threshold for hoverfly_b
        self.wildbees_threshold = wildbees_threshold  # object detection threshold for wildbees
        self.others_threshold = others_threshold  # object detection threshold for other bugs

    def get_video_paths(self):
        """Get the paths of videos under a given folder"""
        file_names = sorted(os.listdir(self.input_dir), key=str.casefold)  # list of all files under the input directory
        file_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in video_types)]  # select only video files
        file_paths = [os.path.join(self.input_dir, file_name) for file_name in file_names]  # get the paths of these video files
        return file_paths

    def extract_frames(self, video_path):
        """Extract all frames from a video and initiate CSRT tracker"""
        output_dir, _ = os.path.splitext(video_path)  # get the video path excluding the file extension
        if self.save_frames:
            os.makedirs(output_dir, exist_ok=True)  # make output folder for store video frames
        filename = os.path.splitext(os.path.basename(video_path))[0]  # get the video name exlcuding the file extension
        video = cv2.VideoCapture(video_path)  # load the video
        params = cv2.TrackerCSRT_Params()
        params.padding = 5.0  # increase padding to consider more context
        params.gsl_sigma = 1.5  # increase GSL sigma for a smoother spatial reliability map
        params.hog_orientations = 9  # number of HOG orientations
        params.num_hog_channels_used = 18  # use more HOG channels
        params.hog_clip = 2.5  # clip HOG values to prevent large gradients from dominating
        params.filter_lr = 0.005  # decrease the learning rate for the filter for more stability
        params.weights_lr = 0.005  # decrease the learning rate for the weights for more stability
        params.background_ratio = 2  # background ratio
        params.psr_threshold = 0.035  # PSR (Peak-to-Sidelobe Ratio) threshold to detect lost object
        params.histogram_bins = 16  # the number of bins in the histogram
        params.histogram_lr = 0.04  # learning rate for the histogram
        tracker, tracker_initialized = cv2.TrackerCSRT_create(), False  # create a CSRT tracker
        roi_locations, n_frame, n_frames, timing = [], 1, [], []  # roi in the form of x, y, w, h
        while True:
            success, frame = video.read()  # read each video frame
            if not success:
                break
            ocr_region = frame[980:1080, 700:1200]  # crop to OCR region
            text = pytesseract.image_to_string(cv2.cvtColor(ocr_region, cv2.COLOR_BGR2GRAY))  # get the OCR result
            lines = text.split('\n')  # separate text by spaces
            text = next((line for line in lines if 'TLC' in line), 'No match found')  # extract text starting with TLC
            timing.append(text)  # append timestamp
            if self.save_frames:
                frame_path = os.path.join(output_dir, f"{filename} {text.replace('/', '.').replace(':', '.')} frame{n_frame:04d}.png")  # define the frame path
                cv2.imwrite(frame_path, frame)  # save the frame as an image
            if not tracker_initialized:
                bbox = cv2.selectROI('Select the flower of interest and hit ENTER', frame, fromCenter=False, showCrosshair=True)  # select ROI
                bbox_success = bbox  # if any tracking fails, the bbox will be the previous successful one (here initiating)
                cv2.destroyAllWindows()
                tracker.init(frame, bbox); tracker_initialized = True  # noqa: initialize the tracker
            else:
                success, bbox = tracker.update(frame)  # update tracker
            if success:
                bbox_success = bbox  # if any tracking fails, the bbox will be the previous successful one
            roi_locations.append(bbox_success)  # add the roi location for each video frame
            (point_x, point_y, width, height) = [int(varible) for varible in bbox_success]  # bbox coordinates
            cv2.rectangle(frame, (point_x, point_y), (point_x + width, point_y + height), (255, 255, 255), 3)  # draw the bbox
            cv2.imshow('Tracking ROI', frame)
            n_frames.append(n_frame); n_frame += 1  # noqa: frame name increment by 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release(); cv2.destroyAllWindows()  # noqa: quite video viewing
        np.save(os.path.join(self.input_dir, f"{filename}.npy"), np.array(roi_locations))  # save ROI list os np.array
        result = {'frame_names': n_frames,
                  'time': timing}
        result = pd.DataFrame(data=result)  # collect results in a pd dataframe for exporting to an Excel sheet
        excel_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}.xlsx"
        result.to_excel(os.path.join(output_dir, excel_filename), index=False)
        return output_dir

    def visualize_tracking(self, video_path):
        """Load a video and tracker npy file for visualization"""
        video = cv2.VideoCapture(video_path)  # load the video
        roi_locations = np.load(f'{os.path.splitext(video_path)[0]}.npy').tolist()   # load the ROI locations
        frames, paused, current_frame = [], True, 0  # to load all frames, pause at a given frame or move around
        print('loading video and ROI locations')
        while True:
            success, frame = video.read()  # load video frames
            if not success:
                break
            frames.append(frame)  # buffer the frames
        try:
            while True:
                frame = frames[current_frame].copy()  # create a copy of the frame to draw the bbox
                bbox = roi_locations[current_frame]  # the ROI bbox
                (point_x, point_y, width, height) = [int(variable) for variable in bbox]  # bbox coordinates
                cv2.rectangle(frame, (point_x, point_y), (point_x + width, point_y + height), (255, 255, 255), 3)  # draw the bbox
                cv2.imshow('Tracking ROI', frame)
                if paused:
                    key = cv2.waitKey(0) & 0xFF
                else:
                    key = cv2.waitKey(25) & 0xFF
                    current_frame = (current_frame + 1) % len(frames)
                if key == 27:  # ESC key to quit
                    break
                elif key == 32:  # SPACE key to pause or continue
                    paused = not paused
                elif key == ord('d') and paused:  # D key to the next frame
                    current_frame = (current_frame + 1) % len(frames)
                    paused = True
                elif key == ord('a') and paused:  # A key to the previous frame
                    current_frame = (current_frame - 1) % len(frames)
                    paused = True
        finally:
            video.release()
            cv2.destroyAllWindows()

    def batch4videos(self, mode='E'):
        """Extract frames from all videos or visualize the tracking"""
        video_paths = self.get_video_paths()
        if mode == 'E':  # extraction mode
            output_dirs = [self.extract_frames(video_path) for video_path in video_paths]
            return output_dirs
        elif mode == 'V':  # visualization mode
            _ = [self.visualize_tracking(video_path) for video_path in video_paths]
            return None

    def bug_detector(self, folder_path):
        """Detect pollinators for all frames under a given folder"""
        output_dir = os.path.join(os.path.split(folder_path)[0],
                                  f'{os.path.split(folder_path)[1]} predictions bb{self.bumblebees_threshold} fl{self.flies_threshold} hb{self.honeybees_threshold} fa{self.hoverfly_a_threshold} fb{self.hoverfly_b_threshold} wb{self.wildbees_threshold} o{self.others_threshold}')
        os.makedirs(output_dir, exist_ok=True)
        frame_names = [name for name in os.listdir(folder_path) if any(name.lower().endswith(file_type) for file_type in image_types)]
        mmdet_utils_register_all_modules(init_default_scope=True)  # initialize mmdet scope
        detector = AutoDetectionModel.from_pretrained(model_type='mmdet',
                                                      model_path=self.detector_weight_path,
                                                      config_path=self.detector_config_path,
                                                      confidence_threshold=0.1,
                                                      image_size=self.new_width,
                                                      device=device)

        def detect_bug(image_path):
            """Return sliced detection results"""
            result = get_sliced_prediction(
                image_path,
                detector,
                slice_height=int(self.new_height / self.sahi_slices / self.sahi_overlap_ratio),
                slice_width=int(self.new_width / self.sahi_slices / self.sahi_overlap_ratio),
                overlap_height_ratio=1 - self.sahi_overlap_ratio,
                overlap_width_ratio=1 - self.sahi_overlap_ratio,
                perform_standard_pred=False,
                postprocess_type='GREEDYNMM',  # 'NMM', 'GRREDYNMM' or 'NMS'. Default is 'GRREDYNMM'.
                postprocess_match_metric='IOS',  # 'IOU' for intersection over union, 'IOS' for intersection over smaller area.
                postprocess_match_threshold=0.6,
                verbose=0)
            return result

        def bboxes_postprocess(bboxes, labels, scores):
            """Remove smaller bounding boxes that overlap with a larger one for MSCOCO format
            based on the overlap criteria."""
            bboxes = np.array(bboxes)  # convert list to np.array
            scores = np.array(scores)
            labels = np.array(labels)
            to_remove = set()  # initialize the set of bboxes to be removed
            areas = bboxes[:, 2] * bboxes[:, 3]  # calculate area for all bounding boxes
            new_scores = scores.copy()
            new_labels = labels.copy()
            for idx, _ in enumerate(bboxes):
                for idx_j in range(idx + 1, len(bboxes)):
                    if idx in to_remove or idx_j in to_remove:
                        continue
                    bbox1 = bboxes[idx]
                    bbox2 = bboxes[idx_j]
                    x_overlap_start = max(bbox1[0], bbox2[0])
                    x_overlap_end = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
                    y_overlap_start = max(bbox1[1], bbox2[1])
                    y_overlap_end = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
                    overlap_width = x_overlap_end - x_overlap_start
                    overlap_height = y_overlap_end - y_overlap_start
                    intersection = max(0, overlap_width) * max(0, overlap_height)
                    area1, area2 = areas[idx], areas[idx_j]  # get areas of both bounding boxes
                    smaller_area = min(area1, area2)
                    if intersection > 0.5 * smaller_area:
                        smaller_bbox_idx = idx if area1 < area2 else idx_j
                        to_remove.add(smaller_bbox_idx)
            final_bboxes = np.delete(bboxes, list(to_remove), axis=0).tolist()
            final_scores = np.delete(new_scores, list(to_remove)).tolist()
            final_labels = np.delete(new_labels, list(to_remove)).tolist()
            return final_bboxes, final_labels, final_scores

        n_bugs = []
        for idx, frame_name in tqdm(enumerate(frame_names), total=len(frame_names)):
            # print(f'{idx + 1} out of {len(frame_names) + 1}')
            image = imread_rgb(os.path.join(folder_path, frame_name))
            result = detect_bug(image)
            prediction_labels = np.array([result.to_coco_annotations()[idx]['category_name'] for idx, _ in enumerate(result.to_coco_annotations())])
            prediction_scores = np.array([result.to_coco_annotations()[idx]['score'] for idx, _ in enumerate(result.to_coco_annotations())])
            prediction_bboxes = np.array([result.to_coco_annotations()[idx]['bbox'] for idx, _ in enumerate(result.to_coco_annotations())])
            valid_labels, valid_bboxes, valid_scores = [], [], []
            n_bumblebees, n_flies, n_honeybees, n_hoverfly_a, n_hoverfly_b, n_wildbees = 0, 0, 0, 0, 0, 0
            for bug in Bug:
                threshold = getattr(self, f"{bug.class_name.lower()}_threshold")
                indices = np.where((prediction_labels == bug.class_name) & (prediction_scores > threshold))[0]  # filter bbox based class-specific threshold
                if len(indices) > 0:
                    valid_labels.append([prediction_labels[ind] for ind in indices][0])
                    valid_bboxes.append([prediction_bboxes[ind] for ind in indices][0])   # all bboxes (>threshold) for a given bug class
                    valid_scores.append([prediction_scores[ind] for ind in indices][0])
            if len(valid_bboxes) >= 2:
                valid_bboxes, valid_labels, valid_scores = bboxes_postprocess(valid_bboxes, valid_labels, valid_scores)
            valid_bboxes = [np.array(bbox, dtype=np.int32) for bbox in valid_bboxes]
            locations = [(x_1, y_1, x_1 + width, y_1 + height) for x_1, y_1, width, height in valid_bboxes]
            for idx, label in enumerate(valid_labels):
                if label == 'Bumblebees':
                    n_bumblebees += 1
                elif label == 'Flies':
                    n_flies += 1
                elif label == 'Honeybees':
                    n_honeybees += 1
                elif label == 'Hoverfly_A':
                    n_hoverfly_a += 1
                elif label == 'Hoverfly_B':
                    n_hoverfly_b += 1
                elif label == 'Wildbees':
                    n_wildbees += 1
                x_1, y_1, x_2, y_2 = locations[idx]
                cv2.rectangle(image, (x_1, y_1), (x_2, y_2), bug_color.get(label), 4)
                cv2.putText(image, f'{label}, p={round(valid_scores[idx], 2)}', (x_1, y_1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bug_color.get(label), 2)
            n_bugs.append([n_bumblebees, n_flies, n_honeybees, n_hoverfly_a, n_hoverfly_b, n_wildbees])
            cv2.imwrite(os.path.join(output_dir, frame_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # export the image
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
                completed_events.append((start_frame, -1))
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
            excel_filename = f"{folder_path} predictions bb{self.bumblebees_threshold} fl{self.flies_threshold} hb{self.honeybees_threshold} fa{self.hoverfly_a_threshold} fb{self.hoverfly_b_threshold} wb{self.wildbees_threshold} o{self.others_threshold}//Absence Window {absence_window}.xlsx"
            result.to_excel(excel_filename, index=False)
        return None

    def batch_predict(self, absence_window_range=None):
        """Batch predit for all video frames (subfolders) under a given parent folder"""
        if absence_window_range is None:
            absence_window_range = [2, 3, 4]
        folder_names = [name for name in os.listdir(self.input_dir) if not any(file_type in name.lower() for file_type in video_types + ['predictions', 'xlsx'])]
        folder_dirs = [os.path.join(self.input_dir, folder_name) for folder_name in folder_names]
        for folder_dir in tqdm(folder_dirs, total=len(folder_dirs)):
            self.counts2excel(folder_dir, absence_window_range=absence_window_range)
        return None
