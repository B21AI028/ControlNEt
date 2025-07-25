import torch
from diffusers import UNet2DConditionModel
from diffusion.controlnet import ControlNet
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

def test_controlnet():
    # Load pretrained UNet
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet"
    )
    
    # Initialize ControlNet
    controlnet = ControlNet(unet)
    
    # Create dummy inputs with correct shapes and types
    batch_size = 1
    channels = 4
    height = 64
    width = 64
    
    # Input tensor
    x = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
    
    # Timesteps tensor and embedding
    timesteps = torch.tensor([999], dtype=torch.long)
    time_embedding_dim = unet.config.block_out_channels[0] * 4
    
    # Create time embedding layers (matching UNet's time embedding)
    time_proj = Timesteps(num_channels=320, flip_sin_to_cos=True, downscale_freq_shift=0)
    timestep_embedding = TimestepEmbedding(
        in_channels=320,
        time_embed_dim=time_embedding_dim,  # Changed from out_dim to time_embed_dim
        act_fn="silu"
    )
    
    # Process timesteps
    t_emb = time_proj(timesteps)
    emb = timestep_embedding(t_emb)
    
    # Context tensor (text embeddings)
    context = torch.randn(batch_size, 77, 768, dtype=torch.float32)
    
    # Control input tensor
    control_input = torch.randn(batch_size, 3, height, width, dtype=torch.float32)
    
    # Test forward pass with properly embedded timesteps
    outputs = controlnet(x, emb, context, control_input)
    print("ControlNet test passed!")
    
    # Print output shapes for verification
    print("\nOutput shapes:")
    for i, output in enumerate(outputs[:-1]):
        print(f"Down block {i}: {output.shape}")
    print(f"Mid block: {outputs[-1].shape}")

if __name__ == "__main__":
    test_controlnet()