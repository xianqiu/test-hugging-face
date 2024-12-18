from diffusers import StableDiffusion3Pipeline

from _utils import ModelManager


class StableDiffusion(object):

    def __init__(self):
        self._model_name = "stabilityai/stable-diffusion-3.5-medium"
        self._model = StableDiffusion3Pipeline.from_pretrained(
            ModelManager.get(self._model_name),
            text_encoder_3=None,
            tokenizer_3=None,
        )
        self._model.to(device='cpu')

    def generate(self, prompt, negative_prompt=None):
        image = self._model(prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=512,
                            height=512,
                            num_inference_steps=50,
                            guidance_scale=4).images[0]
        image.save("outputs/sd_out.png")


def test_text_to_image():
    prompt = """
        A sunflower, personified, with a smiling mouth, living on the beach. 
        The background features beach, waves, sunshine, and clouds in the sky. 
        The picture style is watercolor, anime, with minimal details, abstract, 
        very low saturation.
        """
    negative_prompt = "lines, sketches"

    StableDiffusion().generate(prompt=prompt,
                               negative_prompt=negative_prompt)


if __name__ == '__main__':

    test_text_to_image()

