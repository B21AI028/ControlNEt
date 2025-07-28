import json
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from pathlib import Path
import copy

class ZeroConv2d(nn.Module):
    """Zero-initialized convolution layer"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # Initialize weights and bias to zero
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.conv(x)

class ControlNet(nn.Module):
    def __init__(self, unet):
        super().__init__()
        # Copy UNet architecture
        self.encoder = copy.deepcopy(unet.encoder)
        self.middle_block = copy.deepcopy(unet.middle_block)
        
        # Initialize zero convolutions for control
        self.zero_convs = nn.ModuleList([
            ZeroConv2d(block.out_channels, block.out_channels)
            for block in self.encoder.blocks
        ])
        self.zero_convs.append(
            ZeroConv2d(self.middle_block.out_channels, self.middle_block.out_channels)
        )

    def forward(self, x, timestep, context, control_outputs=None):
        hidden_states = x
        
        # Down blocks
        for i, block in enumerate(self.encoder.blocks):
            hidden_states = block(hidden_states, timestep, context)
            if control_outputs is not None:
                # Add control output to hidden states
                hidden_states = hidden_states + control_outputs[i]
        
        # Middle block
        hidden_states = self.middle_block(hidden_states, timestep, context)
        if control_outputs is not None:
            hidden_states = hidden_states + control_outputs[-1]
        
        return hidden_states

def generate_test_images():
    # Load test prompts
    with open('./data/test_prompts.json', 'r') as f:
        test_prompts = json.load(f)
    
    # Initialize pipeline
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    
    # Create output directory
    output_dir = Path("./generated_images")
    output_dir.mkdir(exist_ok=True)
    
    # Generate images for each prompt
    for idx, prompt in enumerate(test_prompts):
        image = pipe(prompt).images[0]
        image.save(output_dir / f"generated_image_{idx}.png")
        print(f"Generated image {idx} for prompt: {prompt}")

if __name__ == "__main__":
    generate_test_images()