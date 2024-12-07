from pathlib import Path
import os
import shutil
import subprocess

from huggingface_hub import snapshot_download


class ModelManager:

    cache_directory = "models"
    token = None

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
    def download(cls, model_name):
        print(">> Downloading ...")
        download_path = snapshot_download(model_name, token=cls.token)
        # 根据 download_path 解析模型地址
        model_directory = None
        # 按模型名称解析
        model_folder_name = cls._to_folder_name(model_name)
        index = download_path.find(model_folder_name)
        if index != -1:
            model_directory = Path(download_path[:index + len(model_folder_name)])

        if model_directory is not None:
            print(">> Moving to local cache ...")
            destination = Path(cls.cache_directory) / model_folder_name
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

            print(f"All items moved from {source} to {destination}")

        except Exception as e:
            print(f"Error occurred: {e}")


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

