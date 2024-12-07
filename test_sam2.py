import os
from pathlib import Path
import subprocess

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator, SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from _utils import ModelManager, RunFfmpeg


class MaskShower(object):

    def __init__(self, image, masks=None, points=None, labels=None, boxes=None, **kwargs):
        self._image = image
        self._masks = masks
        self._points = points
        self._labels = labels
        self._boxes = boxes
        self._figure = None
        self._ax = None
        self._config = {
            'color': np.array([30 / 255, 144 / 255, 255 / 255, 0.6]),
            'random_color': True,
            'marker_size': 375,
            'dpi': 96
        }
        self._process_kwargs(kwargs)

    def _process_kwargs(self, kwargs):
        for k, v in kwargs.items():
            if k in self._config.keys():
                self._config[k] = v

    def _show_mask(self, mask):
        color = self._config['color']
        if self._config['random_color']:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=1)
        self._ax.imshow(mask_image)

    def _show_points(self, points, labels):
        pos_points = points[labels == 1]
        neg_points = points[labels == 0]
        self._ax.scatter(pos_points[:, 0], pos_points[:, 1],
                   color='green', marker='*', s=self._config['marker_size'],
                   edgecolor='white', linewidth=1.25)
        self._ax.scatter(neg_points[:, 0], neg_points[:, 1],
                   color='red', marker='*', s=self._config['marker_size'],
                   edgecolor='white', linewidth=1.25)

    def _show_box(self, box):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        self._ax.add_patch(plt.Rectangle((x0, y0), w, h,
                                         edgecolor='green',
                                         facecolor=(0, 0, 0, 0),
                                         lw=2))

    def _render(self):
        # setup figure
        width, height = self._image.size
        dpi = self._config['dpi']
        self._figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=self._config['dpi'])
        self._ax = plt.gca()

        # show elements
        # image
        self._ax.imshow(self._image)
        plt.imshow(self._image)
        # points
        if self._points is not None:
            self._show_points(self._points, self._labels)
        # box
        if self._boxes is not None:
            if len(np.array(self._boxes).shape) == 1:
                self._boxes = [self._boxes]
            for box in self._boxes:
                self._show_box(box)
        # masks
        if self._masks is not None:
            if len(np.array(self._masks).shape) == 4:  # For multiple box
                self._masks = np.squeeze(self._masks, axis=1)
            for mask in self._masks:
                self._show_mask(np.array(mask))

        plt.axis('off')
        # remove white spaces
        self._figure.subplots_adjust(bottom=0, top=1, left=0, right=1, hspace=0, wspace=0)

    def show(self):
        self._render()
        plt.show()
        return self

    def save(self, filepath):
        self._render()
        plt.savefig(filepath, dpi=self._config['dpi'])
        plt.close(self._figure)


class SAM2forImage(object):

    def __init__(self):
        self._model_name = "facebook/sam2-hiera-large"
        self._device = 'cuda:0'
        self._model = self._load_model()
        self._predictor = SAM2ImagePredictor(self._model)

    def _load_model(self):
        # use absolute path
        model_directory = ModelManager.get(self._model_name).resolve()
        filename = self._model_name.split("/")[1].replace('-', '_')
        checkpoint = str(model_directory / (filename + ".pt"))
        model_config = str(model_directory / filename)
        return build_sam2(model_config, checkpoint, device=self._device)

    def predict(self, image_path, points=None, labels=None, boxes=None):
        image = Image.open(image_path).convert('RGB')
        self._predictor.set_image(image)
        masks, scores, _ = self._predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=boxes,
            multimask_output=False,
        )
        ms = MaskShower(image, masks, points, labels, boxes)
        ms.show().save("outputs/sam2_out_image_predict.png")

    def segment(self, image_path):
        mask_generator = SAM2AutomaticMaskGenerator(
            self._model,
            # 显存不够就调小
            points_per_side = 8,  # 默认值 32, 采样的点数 = points_per_side * points_per_side
            points_per_batch = 8,  # 默认值 64，并行跑的点数
        )
        image = Image.open(image_path).convert('RGB')
        result = mask_generator.generate(np.array(image))
        masks = np.array([item['segmentation'] for item in result])
        ms = MaskShower(image, masks)
        ms.show().save("outputs/sam2_out_image_segment.png")


class SAM2forVideo(object):

    def __init__(self, video_path):
        self._model_name = "facebook/sam2-hiera-tiny"
        self._device = 'cuda:0'
        self._predictor = self._load_model()
        self._video_path= Path(video_path)
        self._frame_directory = Path('outputs') / f"video_frames_{self._video_path.stem}"
        self._frame_names = None
        self._prompt = None
        self._prompt_frame = 0
        self._obj_id = 0
        self._inference_state = None

    def _load_model(self):
        # use absolute path
        model_directory = ModelManager.get(self._model_name).resolve()
        filename = self._model_name.split("/")[1].replace('-', '_')
        checkpoint = str(model_directory / (filename + ".pt"))
        model_config = str(model_directory / filename)
        return build_sam2_video_predictor(model_config, checkpoint,
                                              device=self._device,
                                          apply_postprocessing=False)

    def _video_to_frames(self):
        RunFfmpeg.video_to_frames(self._video_path, self._frame_directory)
        self._frame_names = self._get_frame_names()

    def _frames_out_to_video(self, save_to):
        directory = self._frame_directory / 'out'
        filename_format = "out_%05d.jpg"
        if not any(directory.iterdir()):
            return
        fps = RunFfmpeg.get_video_fps(self._video_path)
        RunFfmpeg.frames_to_video(directory, filename_format, fps, save_to)

    def _get_frame_names(self):
        frame_names = [
            p.name for p in self._frame_directory.iterdir()
            if p.is_file() and p.suffixes[0] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(Path(p).stem))
        return frame_names

    def _get_segments(self, start_frame, end_frame):
        # run propagation throughout the video and collect the results in a dict
        frame_number = end_frame - start_frame + 1
        segments = {}
        for values in self._predictor.propagate_in_video(self._inference_state,
                                                       start_frame_idx=start_frame,
                                                       max_frame_num_to_track=frame_number-2):
            frame, object_id, logits = values
            segments[frame] = {
                object_id: (logits[i] > 0.0).cpu().numpy()
                for i, object_id in enumerate(object_id)
            }
        return segments

    def _predict_and_save(self, batch_size):
        frame_count = len(self._frame_names)
        iterations = frame_count // batch_size
        if frame_count % batch_size > 0:
            iterations += 1

        result = {}
        for i in range(iterations):
            segments = self._get_segments(i * batch_size, (i + 1) * batch_size)
            result.update(segments)
            self._render_and_save(segments)
            print(f"\n[Predicted and saved] >> start_frame = {i * batch_size}, end_frame = {(i + 1) * batch_size}\n")

    def _render_and_save(self, segments):
        directory = self._frame_directory / 'out'
        if not directory.exists():
            directory.mkdir()
        for frame, values in segments.items():
            image = Image.open(self._frame_directory / self._frame_names[frame])
            masks = []
            filepath = directory / f"out_{self._frame_names[frame]}"
            for _, mask in values.items():
                masks.append(mask)
            MaskShower(image, np.array(masks), random_color=False).save(filepath)

    def _set_prompt(self, to_frame, prompt):
        """
        prompt = {
            'points': points,
            'labels': labels,
            'boxe': box
        }
        """
        self._prompt_frame = to_frame
        if isinstance(prompt, list):
            self._prompt = prompt
        else:
            self._prompt = [prompt]

    def set_prompt(self, to_frame, prompt):
        self._set_prompt(to_frame, prompt)
        return self

    def _add_prompt(self, prompt):
        frame_idx, obj_ids, video_res_masks = self._predictor.add_new_points_or_box(
            inference_state=self._inference_state,
            frame_idx=self._prompt_frame,
            obj_id=self._obj_id,
            points=prompt.get('points'),
            labels=prompt.get('labels'),
            box=prompt.get('box'),
        )
        self._obj_id += 1
        return frame_idx, obj_ids, video_res_masks

    def _unzip_prompt(self):
        points_combine = []
        labels_combine = []
        boxes = []
        for prompt in self._prompt:
            points = prompt.get('points')
            if points is not None:
                points_combine += list(points)
            labels = prompt.get('labels')
            if labels is not None:
                labels_combine += list(labels)
            box = prompt.get('box')
            if box is not None:
                boxes.append(box)

        points_combine = np.array(points_combine) if len(points_combine) > 0 else None
        labels_combine = np.array(labels_combine) if len(labels_combine) > 0 else None
        boxes = np.array(boxes) if len(boxes) > 0 else None

        return points_combine, labels_combine, boxes

    def test_prompt(self, predict=False):
        if not self._frame_directory.exists():
            self._video_to_frames()
        frame_names = self._get_frame_names()
        image = Image.open(self._frame_directory / frame_names[self._prompt_frame])

        masks = None
        if predict:
            self._inference_state = self._predictor.init_state(video_path=str(self._frame_directory),
                                                         offload_video_to_cpu=True)
            if self._prompt is None:
                raise ValueError("Run [set_prompt] before [predict]!")
            # add prompt
            for prompt in self._prompt:
                _, _, logits = self._add_prompt(prompt)
                # logits to masks
                masks = (logits > 0.0).cpu().numpy()

        points, labels, boxes = self._unzip_prompt()
        MaskShower(image, masks, points, labels, boxes).show()

    def predict(self):
        self._video_to_frames()
        print(f"[Video frames created] >> frames = {len(self._frame_names)}")
        self._inference_state = self._predictor.init_state(
            video_path=str(self._frame_directory),
            offload_video_to_cpu=True)
        # add prompt
        if self._prompt is None:
            raise ValueError("Run [set_prompt] before [predict]!")
        for prompt in self._prompt:
            self._add_prompt(prompt)
        self._predict_and_save(batch_size=50)
        # frames to video
        filepath = "outputs/sam2_out_video_predict.mp4"
        self._frames_out_to_video(filepath)


class RunTest:

    @staticmethod
    def image_identify_object_by_points():
        image = 'data/images/p1.jpg'
        points = np.array([[390, 260], [390, 600]])
        labels = np.array([1, 1])  # as selected
        SAM2forImage().predict(image, points, labels)

    @staticmethod
    def image_identify_object_by_box():
        image = 'data/images/p2.jpg'
        box = np.array([920, 100, 1200, 800])
        # Remove bicycle parts
        points = np.array([[1030, 650]])
        labels = np.array([0])  # as de-selected
        SAM2forImage().predict(image, points, labels, boxes=box)

    @staticmethod
    def image_identify_multiple_objects():
        image = 'data/images/p2.jpg'
        boxes = np.array(
            [
                [920, 100, 1200, 800],
                [1180, 530, 1330, 820],
                [0, 160, 270, 370],
            ]
        )
        SAM2forImage().predict(image, boxes=boxes)

    @staticmethod
    def image_segment():
        SAM2forImage().segment('data/images/p2.jpg')

    @staticmethod
    def track_object_by_points():
        prompt = {
            'points': np.array([[210, 350], [250, 220]], dtype=np.float32),
            'labels': np.array([1, 1], dtype=np.float32)
        }
        sm = SAM2forVideo("data/videos/v2.mp4")
        sm.set_prompt(to_frame=0, prompt=prompt)
        # sm.test_prompt(predict=True)
        sm.predict()

    @staticmethod
    def track_object_by_box():
        sm = SAM2forVideo("data/videos/v1.mp4")
        prompt = {'box': np.array([440, 370, 710, 560])}
        sm.set_prompt(to_frame=0, prompt=prompt)
        # sm.test_prompt(predict=True)
        sm.predict()

    @staticmethod
    def track_multiple_objects():
        sm = SAM2forVideo("data/videos/v2.mp4")
        prompt1 = {
            'points': np.array([[220, 270]]), 'labels': np.array([1]),
            'box': np.array([170, 200, 270, 340])
        }
        prompt2 = {
            'points': np.array([[400, 150]]), 'labels': np.array([1]),
            'box': np.array([350, 106, 440, 190])
        }
        sm.set_prompt(0, [prompt1, prompt2])
        # sm.test_prompt(predict=True)
        sm.predict()


if __name__ == '__main__':
    RunTest.image_identify_object_by_points()
    # RunTest.image_identify_object_by_box()
    # RunTest.image_identify_multiple_objects()
    # RunTest.image_segment()
    # RunTest.track_object_by_points()
    # RunTest.track_object_by_box()
    # RunTest.track_multiple_objects()
