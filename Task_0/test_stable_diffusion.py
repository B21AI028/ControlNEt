import torch
from diffusers import StableDiffusionPipeline
import torch.nn as nn

class ZeroConv2d(nn.Module):
    """
    A 2D Convolution layer that initializes all weights and biases to zero.
    This is used in ControlNet to initialize the control parameters.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Initialize weights and biases to zero
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

def test_stable_diffusion():
    # Model ID for Stable Diffusion v1.4
    model_id = "CompVis/stable-diffusion-v1-4"
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    
    # Generate image
    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]
    
    # Save the generated image
    image.save("astronaut_rides_horse.png")
    print("Image generated successfully!")

if __name__ == "__main__":
    test_stable_diffusion()