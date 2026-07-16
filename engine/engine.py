# engine/engine.py

from engine.model_runner import ModelRunner
from metrics.metrics import Metrics


class InferenceEngine:

    def __init__(self):

        self.runner = ModelRunner()

    def generate(
        self,
        prompt,
        max_new_tokens,
        temperature,
    ):

        metrics = Metrics()

        metrics.set_prompt_tokens(
            self.runner
            .get_tokenizer()
            .count_tokens(prompt)
        )

        metrics.start()

        generated_text = self.runner.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        metrics.stop()

        generated_tokens = self.runner.get_tokenizer().count_tokens(
            generated_text
        )

        metrics.set_generated_tokens(
            generated_tokens
        )

        return {
            "generated_text": generated_text,
            "metrics": metrics.report(),
        }