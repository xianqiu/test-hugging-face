from pathlib import Path
import shutil
from ultralytics import YOLO
import cv2

from _utils import ModelManager, RunFfmpeg


class UltralyticsYolo(object):

    def __init__(self, task=None):
        """
        :param task: str,
            value in {'detect', 'pose', 'segment', 'obb', 'classify'}
            task=None implies 'detect'
        """
        self._model_name = "yolo11n"
        self._model = YOLO(self._get_model_path(task))

    def _get_model_path(self, task):
        if task is None or task == "detect":
            filename = "yolo11n.pt"
        elif task == "pose":
            filename = "yolo11n-pose.pt"
        elif task == "segment":
            filename = "yolo11n-seg.pt"
        elif task == "obb":
            filename = "yolo11n-obb.pt"
        elif task == "classify":
            filename = "yolo11n-cls.pt"
        else:
            raise ValueError("[task] value error! Possible values are: [detect, pose, segment, obb, classify]")

        directory = ModelManager.get(self._model_name)

        return directory / filename

    def predict_image(self, source, save_to=None):
        results = self._model(source)
        # Process results list
        for result in results:
            result.show()
            if save_to:
                result.save(filename=save_to)

    def predict_video(self, source, save_to):
        cache_directory = Path("outputs") / f"yolo_video_frames_{Path(source).stem}"
        if not cache_directory.exists():
            cache_directory.mkdir()
        results = self._model(source, stream=True)
        # save results (images) to cache directory
        i = 0
        for result in results:
            result.save(cache_directory / f"out_{i:05d}.jpg")
            i += 1
        # make videos
        fps = RunFfmpeg.get_video_fps(source)
        RunFfmpeg.frames_to_video(cache_directory,
                                  "out_%05d.jpg",
                                  fps, save_to)
        # remove cache
        shutil.rmtree(cache_directory)

    def track_video(self, source):
        cap = cv2.VideoCapture(source)
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                # Run YOLO11 tracking on the frame, persisting tracks between frames
                results = self._model.track(frame, persist=True)
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                # Display the annotated frame
                cv2.imshow("YOLO11 Tracking", annotated_frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break
        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

    def get_model(self):
        return self._model


class TestDetect:

    # 目标检测

    yl = UltralyticsYolo(task='detect')

    @classmethod
    def predict_image(cls):
        source = "data/images/p3.jpg"
        cls.yl.predict_image(source)

    @classmethod
    def predict_video(cls):
        source = "data/videos/v2.mp4"
        save_to = "outputs/yolo_out_v2.mp4"
        cls.yl.predict_video(source, save_to)

    @classmethod
    def track_video(cls):
        source = "data/videos/v1.mp4"
        cls.yl.track_video(source)

    @classmethod
    def track_video_youtube(cls):
        from pytubefix import YouTube
        source = "https://www.youtube.com/watch?v=wj3tP65-iYA"
        yt = YouTube(source)
        # 获取最高分辨率的视频流 URL
        stream = yt.streams.filter(file_extension='mp4').get_highest_resolution()
        cls.yl.track_video(stream.url)

    @staticmethod
    def _get_bilibili_video_stream(watch_url):
        import subprocess
        import json
        # 获取 bilibili 下载链接，保存到 stream 中
        command = ["you-get", "--json", watch_url]
        streams = None
        try:
            result = subprocess.run(command, capture_output=True)
            result = json.loads(result.stdout)
            # 音频和视频流的链接
            streams = result['streams']['dash-flv480-HEVC']['src']
            # 视频流的链接
            streams = streams[0]
        except Exception as e:
            print(e)
        return streams

    @classmethod
    def track_video_bilibili(cls):
        source = "https://www.bilibili.com/video/BV1bVzbYNESB"
        streams = cls._get_bilibili_video_stream(source)
        if streams:
            # 处理第一个视频流
            cls.yl.track_video(streams[0])


class TestPose:

    # 姿态估计

    yl = UltralyticsYolo(task='pose')

    @classmethod
    def predict_image(cls):
        source = "data/images/p3.jpg"
        cls.yl.predict_image(source)

    @classmethod
    def track_video(cls):
        source = "data/videos/v3.mp4"
        cls.yl.track_video(source)


class TestObb:

    # 定向物体检测

    yl = UltralyticsYolo(task='obb')

    @classmethod
    def predict_image(cls):
        source = "data/images/p4.jpg"
        cls.yl.predict_image(source)


class TestClassify:

    # 分类

    yl = UltralyticsYolo(task='classify')

    @classmethod
    def predict_image(cls):
        source = "data/images/p2.jpg"
        # returns top5 classes
        cls.yl.predict_image(source)


class TestSegment:

    # 分割

    yl = UltralyticsYolo(task='segment')

    @classmethod
    def predict_image(cls):
        source = "data/images/p2.jpg"
        cls.yl.predict_image(source)


if __name__ == '__main__':
    TestDetect.predict_image()
    # TestDetect.predict_video()
    # TestDetect.track_video()
    # TestDetect.track_video_youtube()
    # TestDetect.track_video_bilibili()
    # TestPose.predict_image()
    # TestPose.track_video()
    # TestObb.predict_image()
    # TestClassify.predict_image()
    # TestSegment.predict_image()