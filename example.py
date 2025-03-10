from diffusers.utils import load_image
from simple_leftrefill_inpainting import LeftRefillGuidance

model = LeftRefillGuidance()

result = model.predict(
    load_image("./assets/inpainted.png"),
    load_image("./assets/mask.png"),
    load_image("./assets/reference.png"),
)

result.save("./assets/result.png")