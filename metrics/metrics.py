# metrics/metrics.py

import time


class Metrics:

    def __init__(self):

        self.start_time = None
        self.end_time = None

        self.prompt_tokens = 0
        self.generated_tokens = 0

    def start(self):

        self.start_time = time.perf_counter()

    def stop(self):

        self.end_time = time.perf_counter()

    def set_prompt_tokens(self, count):

        self.prompt_tokens = count

    def set_generated_tokens(self, count):

        self.generated_tokens = count

    def latency(self):

        return self.end_time - self.start_time

    def tokens_per_second(self):

        latency = self.latency()

        if latency == 0:
            return 0

        return self.generated_tokens / latency

    def report(self):

        return {
            "latency_seconds": round(self.latency(), 3),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "tokens_per_second": round(
                self.tokens_per_second(),
                2
            ),
        }