import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Loop through each GPU and print its details
    for i in range(num_gpus):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(
            f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
        )
        print(
            f"  CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
        )
else:
    print("No GPUs available. Using CPU.")
