# engine/model_runner.py

import torch
from transformers import AutoModelForCausalLM

from config import (
    MODEL_NAME,
    DEVICE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
)

from engine.tokenizer import Tokenizer

class ModelRunner:

    def __init__(self):

        # ----------------------------------
        # Device
        # ----------------------------------

        self.device = torch.device(DEVICE)

        # ----------------------------------
        # Tokenizer
        # ----------------------------------

        self.tokenizer = Tokenizer()

        # ----------------------------------
        # Model
        # ----------------------------------

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )

        # ----------------------------------
        # Move model to GPU
        # ----------------------------------

        self.model.to(self.device)

        # ----------------------------------
        # Inference mode
        # ----------------------------------

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):

        # ----------------------------
        # Tokenize
        # ----------------------------

        inputs = self.tokenizer.batch_encode([prompt])

        # ----------------------------
        # Move tensors to GPU
        # ----------------------------

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # ----------------------------
        # Generate
        # ----------------------------

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

        # ----------------------------
        # Decode
        # ----------------------------

        generated_text = self.tokenizer.batch_decode(outputs)[0]

        return generated_text

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device