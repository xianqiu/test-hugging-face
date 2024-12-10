from pathlib import Path
import os
import shutil
import subprocess
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

from huggingface_hub import snapshot_download


class ModelManager:

    cache_directory = "models"

    @classmethod
    def _to_folder_name(cls, model_name):
        return "models--" + model_name.replace('/', '--')

    @classmethod
    def get(cls, model_name):
        model_folder_name = cls._to_folder_name(model_name)
        snapshots = Path(cls.cache_directory) / model_folder_name / 'snapshots'
        if snapshots.exists():
            folder_names = [folder.name for folder in snapshots.iterdir()]
            assert len(folder_names) == 1, ValueError("Model directory error!")
            return snapshots / folder_names[0]
        elif snapshots.parent.exists():
            return snapshots.parent
        else:
            print(">> Cache not found!")
            return model_name

    @classmethod
    def download(cls, model_name, token=None):
        print(">> Downloading ...")
        download_path = snapshot_download(model_name, token=token)
        # 根据 download_path 解析模型地址
        model_directory = None
        # 按模型名称解析
        model_folder_name = cls._to_folder_name(model_name)
        index = download_path.find(model_folder_name)
        if index != -1:
            model_directory = Path(download_path[:index + len(model_folder_name)])

        if model_directory is not None:
            print(">> Moving to local cache ...")
            destination = (Path(cls.cache_directory) / model_folder_name).resolve()
            print(f">> [FROM]: {model_directory} \n  [TO]: {destination}")
            cls._move(model_directory, destination.resolve())
        print(">> Done.")

    @staticmethod
    def _move(source, destination):
        """ 把文件夹 source 下所有的文件和文件夹移动到 destination 下。
        执行如下步骤：
        1、把 source 下所有的文件和文件夹移动到 destination；
        2、如果移动成功，则把 source 下所有的文件和文件夹删除。
        """
        try:
            # 确保目标文件夹存在，如果不存在则创建
            os.makedirs(destination, exist_ok=True)

            # 移动文件和文件夹
            for item in os.listdir(source):
                s = os.path.join(source, item)
                d = os.path.join(destination, item)
                shutil.move(s, d)
            # 如果所有文件都移动成功，清空源文件夹
            for item in os.listdir(source):
                s = os.path.join(source, item)
                if os.path.isfile(s) or os.path.islink(s):
                    os.remove(s)
                else:
                    shutil.rmtree(s)
        except Exception as e:
            print(f"Error occurred: {e}")


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
        if self._config['random_color']:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = self._config['color']
        self._ax.add_patch(plt.Rectangle((x0, y0), w, h,
                                         edgecolor=color,
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


class RunFfmpeg:

    os.chdir(Path('.'))

    @staticmethod
    def _run_command(command):
        # 执行命令并捕获输出
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # 解析进度信息
            while True:
                output = process.stderr.readline()  # 读取标准错误输出
                if output == '' and process.poll() is not None:
                    break
                if output and 'Metadata' not in output:
                    print(output.strip())
            # 等待进程完成
            process.wait()
        except Exception as e:
            print(e)

    @classmethod
    def video_to_frames(cls, video_path, save_to):
        save_to= Path(save_to)
        if not save_to.exists():
            save_to.mkdir()

        command = [
            "ffmpeg",
            '-y',  # force output, overwrite
            "-i", str(video_path),
            "-q:v", "2",
            "-start_number", "0",
            str(save_to) + "\\%05d.jpg"
        ]
        cls._run_command(command)

    @staticmethod
    def get_video_fps(video_path):
        command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            fps_str = result.stdout.strip()
            # 处理帧率字符串
            if '/' in fps_str:
                numerator, denominator = map(int, fps_str.split('/'))
                fps = numerator / denominator  # 计算实际的帧率
            else:
                fps = int(fps_str)  # 如果没有分数，直接转换
            return fps
        except subprocess.CalledProcessError as e:
            print(f"{e}")
            return None

    @classmethod
    def frames_to_video(cls, frames_directory, filename_format, fps, save_to):
        if not any(Path(frames_directory).iterdir()):
            return
        command = [
            'ffmpeg',
            '-y',  # force output, overwrite
            '-framerate', str(fps),
            '-i', str(frames_directory / filename_format),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(save_to)
        ]
        cls._run_command(command)


class CudaMemory:

    # For Debugging.

    @staticmethod
    def get(is_print=True):
        memory = torch.cuda.memory_allocated()
        memory = memory / (1024 ** 2)  # MB
        if is_print:
            print(f"Cuda memory used: {memory:.2f} MB")
        return memory

    @staticmethod
    def get_tensor(tensor, is_print=True):
        memory = tensor.element_size() * tensor.numel()
        memory = memory / (1024 ** 2)  # MB
        if is_print:
            print(f"Cuda memory [{str(tensor)}] used: {memory:.2f} MB")
        return memory

