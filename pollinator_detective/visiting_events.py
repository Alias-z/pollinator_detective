"""Module providing functions for counting pollinator visiting events"""
# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, no-member, no-name-in-module, relative-beyond-top-level, wildcard-import
import os  # interact with the operating system
import cv2  # OpenCV
import numpy as np  # NumPy
import pandas as pd  # for Excel sheet
from tqdm import tqdm
import pytesseract; pytesseract.pytesseract.tesseract_cmd = 'C://Programs//Tesseract-OCR//tesseract.exe'   # noqa: for OCR
from PIL import Image  # Pillow image processing
from mmdet.utils import register_all_modules as mmdet_utils_register_all_modules  # register mmdet modules
from mmdet.apis import init_detector as mmdet_apis_init_detector  # initialize mmdet model
from mmdet.apis import inference_detector as mmdet_apis_inference_detector  # mmdet inference detector
from .core import device, video_types  # import core functions


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


class VisitingEvents():
    """Counting pollinator visiting events"""
    def __init__(self,
                 input_dir: str,
                 save_frames: bool = False,
                 detector_config_path: str = None,
                 detector_weight_path: str = None,
                 new_width: int = 1920,
                 new_height: int = 1080,
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
        file_names = [name for name in file_names if any(name.lower().endswith(file_type) for file_type in video_types) and 'predictions' not in name.lower()]  # select only source video files
        file_paths = [os.path.join(self.input_dir, file_name) for file_name in file_names]  # get the paths of these video files
        return file_paths

    def select_roi(self, video_path):
        """Select the ROI for a video tracker"""
        video = cv2.VideoCapture(video_path)  # load the video
        success, frame = video.read()  # read each video frame
        if success:
            bbox = cv2.selectROI('Select the flower of interest and hit ENTER', frame, fromCenter=False, showCrosshair=True)  # select ROI
        video.release(); cv2.destroyAllWindows()  # noqa: quite video viewing
        return bbox

    def extract_frames(self, video_path, roi_bbox):
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
                bbox = roi_bbox  # load selected ROI
                bbox_success = bbox  # if any tracking fails, the bbox will be the previous successful one (here initiating)
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
        excel_filename = f"{os.path.splitext(video_path)[0]}.xlsx"
        result.to_excel(excel_filename, index=False)
        return output_dir

    def video_buffer(self, video_path):
        """Preload all video frames into memory"""
        video = cv2.VideoCapture(video_path)  # load the video
        frames = []  # to store all frames
        while True:
            success, frame = video.read()  # load video frames
            if not success:
                break
            frames.append(frame)  # buffer the frames
        video.release()
        return frames

    def visualize_tracking(self, video_path):
        """Load a video and tracker npy file for visualization"""
        frames = self.video_buffer(video_path)  # load the video frames
        roi_locations = np.load(f'{os.path.splitext(video_path)[0]}.npy').tolist()   # load the ROI locations
        paused, current_frame = True, 0  # to pause at a given frame or move around
        print('loading video and ROI locations')

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
            cv2.destroyAllWindows()

    def batch4videos(self, mode='E'):
        """Extract frames from all videos or visualize the tracking"""
        video_paths = self.get_video_paths()  # get all video paths
        if mode == 'E':  # extraction mode
            roi_bboxes = [self.select_roi(video_path) for video_path in video_paths]
            for idx, video_path in enumerate(video_paths):
                self.extract_frames(video_path, roi_bboxes[idx])
        elif mode == 'V':  # visualization mode
            _ = [self.visualize_tracking(video_path) for video_path in video_paths]
        return None

    def frames_to_video(self, frames, output_path, fps=30):
        """Convert a list of frames (NumPy arrays) to a video file"""
        height, width, _ = frames[0].shape  # get the dimensions of the first frame
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'DIVX' for AVI
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # write each frame to the video file
        video.release()  # Release the video writer object
        return None

    def bug_detector(self, video_path):
        """Detect pollinators for all frames under a given folder"""
        output_dir = os.path.join(os.path.splitext(video_path)[0],
                                  f'predictions bb{self.bumblebees_threshold} fl{self.flies_threshold} hb{self.honeybees_threshold} fa{self.hoverfly_a_threshold} fb{self.hoverfly_b_threshold} wb{self.wildbees_threshold} o{self.others_threshold}')
        os.makedirs(output_dir, exist_ok=True)  # create directory for frames

        def bboxes_postprocess(roi_area, bboxes, labels, scores):
            """Remove smaller bounding boxes that overlap with a larger one based on the overlap criteria (default 0.5)"""
            bboxes = np.array(bboxes)  # convert list to np.array
            scores = np.array(scores)
            labels = np.array(labels)
            to_remove = set()  # initialize the set of bboxes to be removed
            widths = bboxes[:, 2] - bboxes[:, 0]  # calculate the width
            heights = bboxes[:, 3] - bboxes[:, 1]  # calculate the height
            areas = widths * heights  # calculate area for all bounding boxes
            new_scores = scores.copy()  # prevent changing in position
            new_labels = labels.copy()
            for idx, _ in enumerate(bboxes):
                for idx_j in range(idx + 1, len(bboxes)):
                    if idx in to_remove or idx_j in to_remove:
                        continue
                    bbox1 = bboxes[idx]
                    bbox2 = bboxes[idx_j]
                    x_overlap_start = max(bbox1[0], bbox2[0])
                    x_overlap_end = min(bbox1[2], bbox2[2])
                    y_overlap_start = max(bbox1[1], bbox2[1])
                    y_overlap_end = min(bbox1[3], bbox2[3])
                    overlap_width = x_overlap_end - x_overlap_start
                    overlap_height = y_overlap_end - y_overlap_start
                    intersection = max(0, overlap_width) * max(0, overlap_height)
                    area1, area2 = areas[idx], areas[idx_j]  # get areas of both bounding boxes
                    smaller_area = min(area1, area2)
                    if intersection > 0.5 * smaller_area:
                        smaller_bbox_idx = idx if area1 < area2 else idx_j
                        to_remove.add(smaller_bbox_idx)
                    if smaller_area < 0.001 * roi_area:
                        to_remove.add(smaller_bbox_idx)  # remove every small object
            final_bboxes = np.delete(bboxes, list(to_remove), axis=0).tolist()
            final_scores = np.delete(new_scores, list(to_remove)).tolist()
            final_labels = np.delete(new_labels, list(to_remove)).tolist()
            return final_bboxes, final_labels, final_scores

        frames = self.video_buffer(video_path)  # load the video frames
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]  # BRG to RGB
        roi_locations = np.load(f'{os.path.splitext(video_path)[0]}.npy').tolist()   # load the ROI locations
        roi_patches = []  # list of patches and locations
        for idx, frame in enumerate(frames):
            frame = Image.fromarray(frame)  # convert to PIL image
            point_x, point_y, width, height = roi_locations[idx]  # get the coordinates
            roi_patch = frame.crop((point_x, point_y, point_x + width, point_y + height))  # crop to ROI
            roi_patches.append(np.array(roi_patch))  # append np.array format ROI
        mmdet_utils_register_all_modules(init_default_scope=True)  # initialize mmdet scope
        detector = mmdet_apis_init_detector(self.detector_config_path, self.detector_weight_path, device=device)  # initialize a object detector from config file
        results = [mmdet_apis_inference_detector(detector, roi_patch) for roi_patch in roi_patches]  # inference image(s) with the object detector
        scores = [result.pred_instances.scores.cpu().numpy() for result in results]  # get the predicted scores
        labels = [result.pred_instances.labels.cpu().numpy() for result in results]  # get the predicted labels
        labels = [np.array(label, dtype=np.int32) for label in labels]  # convert labels to integers
        bboxes = [result.pred_instances.bboxes.cpu().numpy() for result in results]  # get the predicted bboxes
        n_bugs, new_frames = [], []
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            frame = frame.copy()  # prevent chang in position
            valid_labels, valid_bboxes, valid_scores = [], [], []
            n_bumblebees, n_flies, n_honeybees, n_hoverfly_a, n_hoverfly_b, n_wildbees = 0, 0, 0, 0, 0, 0
            for bug in Bug:
                threshold = getattr(self, f"{bug.class_name.lower()}_threshold")  # get the threshold for the given bug type
                indices = np.where((labels[idx] == bug.class_encoding) & (scores[idx] > threshold))[0]  # filter bbox based class-specific threshold
                if len(indices) > 0:
                    valid_labels.append([labels[idx][ind] for ind in indices][0])
                    valid_bboxes.append([bboxes[idx][ind] for ind in indices][0])   # all bboxes (>threshold) for a given bug class
                    valid_scores.append([scores[idx][ind] for ind in indices][0])
                if len(valid_bboxes) >= 2:
                    roi_area = roi_locations[idx].shape[0] * roi_locations[idx].shape[1]  # get the ROI area
                    valid_bboxes, valid_labels, valid_scores = bboxes_postprocess(roi_area, valid_bboxes, valid_labels, valid_scores)
            valid_bboxes = [np.array(bbox, dtype=np.int32) for bbox in valid_bboxes]
            point_x, point_y, width, height = roi_locations[idx]  # ROI location in MSCOCO format
            ox_1, oy_1, ox_2, oy_2 = point_x, point_y, point_x + width, point_y + height  # convet MSCOCO format to mmdet format
            cv2.rectangle(frame, (ox_1, oy_1), (ox_2, oy_2), (255, 255, 255), 3)  # draw the ROI in white
            for idx_2, label in enumerate(valid_labels):
                bug_type = Bug[label].class_name
                if bug_type == 'Bumblebees':
                    n_bumblebees += 1
                elif bug_type == 'Flies':
                    n_flies += 1
                elif bug_type == 'Honeybees':
                    n_honeybees += 1
                elif bug_type == 'Hoverfly_A':
                    n_hoverfly_a += 1
                elif bug_type == 'Hoverfly_B':
                    n_hoverfly_b += 1
                elif bug_type == 'Wildbees':
                    n_wildbees += 1
                x_1, y_1, x_2, y_2 = valid_bboxes[idx_2]  # bug bbox in mmdet format
                x_1 += ox_1; y_1 += oy_1; x_2 += ox_1; y_2 += oy_1  # noqa: map bboxes back to the image
                cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), Bug[label].mask_rgb, 3)  # draw the bug bboxes
                cv2.putText(frame, f'{bug_type}, p={round(float(valid_scores[idx_2]), 2)}', (x_1, y_1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Bug[label].mask_rgb, 2)  # draw the bug types and scores
            if self.save_frames:
                cv2.imwrite(os.path.join(output_dir, f'frame {idx+1}.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # export the frames as images
            new_frames.append(frame)  # collect frams with bboxes
            n_bugs.append([n_bumblebees, n_flies, n_honeybees, n_hoverfly_a, n_hoverfly_b, n_wildbees])
        self.frames_to_video(new_frames, f'{output_dir}.AVI')
        return np.array(n_bugs).T.tolist()

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

    def counts2excel(self, video_path, absence_window_range):
        """Count visiting events for all pollinator types and output to an Excel file"""
        if absence_window_range is None:
            absence_window_range = [2]
        counts = self.bug_detector(video_path)
        for absence_window in absence_window_range:
            all_events = []  # to collect all events
            for count in counts:
                events = [""] * len(count)
                visits = self.count_visits(count, absence_window=absence_window)
                for index_i, (start, end) in enumerate(visits, 1):  # index_i starts from 1
                    for index_j in range(start, end + 1):  # +1 to include the end index
                        events[index_j] = f"Visits {index_i}"
                all_events.append(events)
            result = {f'{Bug[0].class_name} counts': counts[0],
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
            result_ocr = pd.read_excel(f'{os.path.splitext(video_path)[0]}.xlsx')
            result = pd.concat([result_ocr, result], axis=1)
            excel_filename = os.path.join(os.path.splitext(video_path)[0], f'predictions bb{self.bumblebees_threshold} fl{self.flies_threshold} hb{self.honeybees_threshold} fa{self.hoverfly_a_threshold} fb{self.hoverfly_b_threshold} wb{self.wildbees_threshold} o{self.others_threshold} Absence Window {absence_window}.xlsx')
            result.to_excel(excel_filename, index=False)
        return None

    def batch_predict(self, absence_window_range=None):
        """Batch predit for all video frames (subfolders) under a given parent folder"""
        if absence_window_range is None:
            absence_window_range = [2]
        video_paths = self.get_video_paths()
        for video_path in tqdm(video_paths, total=len(video_paths)):
            self.counts2excel(video_path, absence_window_range=absence_window_range)
        return None
