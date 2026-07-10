import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

class StableDiffusionV1_5:

    def __init__(self, model_name: str, target_device: str, generation_conf: dict):
        """
        Initializes the StableDiffusionV1_5 pipeline.

        Args:
            model_name (str): The name or path of the pre-trained model.
                              For Hugging Face models, use the model ID (e.g., "runwayml/stable-diffusion-v1-5").
                              If it's a local path, ensure the model is correctly downloaded there.
            target_device (str): The device to load the model onto (e.g., "cuda" for GPU, "cpu" for CPU).
            generation_conf (dict): A dictionary containing configuration for image generation.
                                    Expected keys include:
                                    - 'num_inference_steps' (int): Number of denoising steps.
                                    - 'guidance_scale' (float): Classifier-free guidance scale.
                                    - 'height' (int): Height of the generated image.
                                    - 'width' (int): Width of the generated image.
                                    - 'seed' (int, optional): Random seed for reproducibility.
                                    - 'safety_checker' (bool, optional): If False, the safety checker
                                                                         will be disabled. Defaults to True.
        """
        self.model_name = model_name
        self.device = target_device
        self.generation_conf = generation_conf

        # Determine if the safety checker should be disabled based on the configuration.
        # If 'safety_checker' is explicitly False in generation_conf, disable it.
        # Otherwise, the default behavior of the pipeline will be used (safety checker enabled if present).
        disable_safety_checker = not self.generation_conf.get('safety_checker', True)

        # Prepare keyword arguments for the from_pretrained method
        pipeline_kwargs = {
            "torch_dtype": torch.float16,
        }

        # If disabling the safety checker is requested, set the appropriate flags
        if disable_safety_checker:
            pipeline_kwargs["safety_checker"] = None
            pipeline_kwargs["requires_safety_checker"] = False

        # Initialize the DiffusionPipeline with the specified model and configurations
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            **pipeline_kwargs
        )

        # Apply a faster scheduler (DPMSolverMultistepScheduler) as suggested in the original code
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        
        # Move the pipeline to the target device (e.g., GPU or CPU)
        self.pipeline = self.pipeline.to(self.device)

    def generate_image(self, prompt: str, **kwargs) -> Image.Image:
        """
        Generates a single image from a text prompt using the configured pipeline.

        This method allows overriding default generation parameters specified
        during initialization via `kwargs`.

        Args:
            prompt (str): The text prompt for image generation.
            **kwargs: Optional keyword arguments to override parameters from `generation_conf`
                      for this specific image generation. These can include:
                      - `num_inference_steps` (int)
                      - `guidance_scale` (float)
                      - `height` (int)
                      - `width` (int)
                      - `seed` (int)

        Returns:
            PIL.Image.Image: The generated image.
        """
        # Create a mutable copy of the generation configuration and apply any overrides
        current_conf = self.generation_conf.copy()
        current_conf.update(kwargs)

        # Extract generation parameters, using defaults if not specified in config or kwargs
        num_inference_steps = current_conf.get('num_inference_steps', 50)
        guidance_scale = current_conf.get('guidance_scale', 7.5)
        height = current_conf.get('height', 512)
        width = current_conf.get('width', 512)
        seed = current_conf.get('seed', None)

        # Set up a torch generator for reproducible results if a seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate the image without gradient calculation (for inference efficiency)
        with torch.no_grad():
            image = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator
            ).images[0]
        return image

    def generate_batch(self, prompts: list[str], **kwargs) -> list[Image.Image]:
        """
        Generates multiple images from a list of text prompts.

        Each prompt in the batch will use the same generation parameters.

        Args:
            prompts (list[str]): A list of text prompts for image generation.
            **kwargs: Optional keyword arguments to override parameters from `generation_conf`
                      for this entire batch generation.

        Returns:
            list[PIL.Image.Image]: A list of generated images.
        """
        images = []
        for prompt in prompts:
            # Generate each image in the batch using the same (or overridden) parameters
            image = self.generate_image(prompt, **kwargs)
            images.append(image)
        return images

# if __name__ == "__main__":
#     # This dictionary represents your YAML configuration
#     # Note: "stable-diffusion-v1-5/stable-diffusion-v1-5" might be a local path.
#     # For a standard Hugging Face model, use "runwayml/stable-diffusion-v1-5".
#     config = {
#         "model_name": "runwayml/stable-diffusion-v1-5", # Or your local path if applicable
#         "target_device": "cuda",  # Use "cpu" if you don't have a GPU
#         "generation_conf": {
#             "num_inference_steps": 50,
#             "guidance_scale": 7.5,
#             "height": 512,
#             "width": 512,
#             "seed": 1,
#             "safety_checker": False  # Set to False to disable the safety checker
#         }
#     }

#     try:
#         # Instantiate the StableDiffusionV1_5 pipeline with the provided configuration
#         print(f"Initializing Stable Diffusion pipeline for model: {config['model_name']} on device: {config['target_device']}")
#         sd_pipeline = StableDiffusionV1_5(
#             model_name=config["model_name"],
#             target_device=config["target_device"],
#             generation_conf=config["generation_conf"]
#         )
#         print("Pipeline initialized successfully.")

#         # Example of generating a single image with a prompt that might be flagged by a safety checker
#         prompt_nsfw = "a photo of a nude figure in an artistic pose, renaissance style, highly detailed, oil painting"
#         print(f"\nGenerating image with prompt: '{prompt_nsfw}'")
#         generated_image_nsfw = sd_pipeline.generate_image(prompt_nsfw)
#         generated_image_nsfw.save("generated_nsfw_image.png")
#         print("NSFW image generated and saved as 'generated_nsfw_image.png'.")

#         # Example of generating a batch of images
#         prompts_batch = [
#             "a futuristic cityscape at sunset, neon lights, high detail",
#             "a serene forest with a hidden waterfall, misty atmosphere, digital art"
#         ]
#         print(f"\nGenerating a batch of {len(prompts_batch)} images...")
#         generated_batch_images = sd_pipeline.generate_batch(prompts_batch)
#         for i, img in enumerate(generated_batch_images):
#             img.save(f"batch_image_{i}.png")
#             print(f"Batch image {i+1} saved as 'batch_image_{i}.png'.")
#         print("Batch image generation complete.")

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         print("Please ensure you have the 'diffusers' library installed (`pip install diffusers transformers accelerate torch`),")
#         print("and that your 'model_name' is correct and accessible (e.g., 'runwayml/stable-diffusion-v1-5' for Hugging Face models).")
#         print("Also, verify your 'target_device' ('cuda' or 'cpu') is available.")

