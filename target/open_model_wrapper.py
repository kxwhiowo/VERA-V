
import transformers
import torch
from .base_target_lm import BaseTarget
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class OpenModelWrapper(BaseTarget):
    def __init__(self, model_name, generation_conf, target_device):
        super().__init__()
        self.model_id = model_name
        self.target_device = target_device
        self.generation_conf = generation_conf


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=target_device  # Use device_map for efficient loading
        )

        # --- 2. Configure tokenizer settings ---
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # Important for batched generation

        # --- 3. (CRITICAL) Ensure chat template exists ---
        # This is the modern replacement for your entire if/elif block.
        # It relies on the template built into the model's files.
        if self.tokenizer.chat_template is None:
            # As a fallback, we manually set the Llama 3 template.
            # This makes your code robust even if the local model files are incomplete.
            print("Warning: Chat template not found in tokenizer. Manually setting Llama 3 template.")
            self.tokenizer.chat_template = (
                "{{ bos_token }}"
                "{% for message in messages %}"
                    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
            )

    def batch_respond(self, queries):
        # --- 4. (CRITICAL) Use the modern chat template method ---
        # This replaces the old `prompt.format()` logic.
        batch_messages = [[{"role": "user", "content": q}] for q in queries]
        prompts = [self.tokenizer.apply_chat_template(
            p,
            tokenize=False,
            add_generation_prompt=True
        ) for p in batch_messages]

        # --- Tokenize and Generate ---
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.target_device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.generation_conf)

        # --- Decode the response correctly ---
        gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        responses = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return [r.strip() for r in responses]

    # The rest of your methods can remain as they are.
    def get_generation_pipeline(self):
        return pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.model.device if not hasattr(self.model, 'device_map') else 0,
            **self.generation_conf
        )

    def clean_response(self, prompt, response):
        return response[len(prompt):]





        