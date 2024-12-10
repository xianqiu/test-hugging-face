import numpy as np
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

from _utils import ModelManager, MaskShower


class Detr(object):

    def __init__(self):
        self._model_name = ModelManager.get("facebook/detr-resnet-50")
        self._device = "cpu:0"
        self._model = DetrForObjectDetection.from_pretrained(self._model_name, revision="no_timm")
        self._processor = DetrImageProcessor.from_pretrained(self._model_name, revision="no_timm")

    def identify(self, image_path):
        image = Image.open(image_path)
        inputs = self._processor(images=image, return_tensors="pt")
        outputs = self._model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self._processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        boxes = []
        print(">> Results:")
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            boxes.append(box)
            print(
                f"+ Location = {box} --> [{self._model.config.id2label[label.item()]}] / confidence = {round(score.item(), 3)}"
            )
        MaskShower(image, boxes=np.array(boxes)).show()


if __name__ == '__main__':
    Detr().identify("data/images/p2.jpg")

