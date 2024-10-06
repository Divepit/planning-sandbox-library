import time
import logging

class Benchmark:
    def __init__(self, description, start_now=False, silent=False):
        self.silent = silent
        self.description = description
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        if start_now:
            self.start()

    def start(self):
        if not self.silent:
            logging.debug(f"Starting benchmark: {self.description}")
        if self.start_time is not None:
            raise Exception("Benchmark already started")
        self.start_time = time.perf_counter()

    def stop(self, silent=False):
        if self.start_time is None:
            raise Exception("Benchmark not started")
        elif self.end_time is not None:
            raise Exception("Benchmark already stopped")
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        if not self.silent and not silent:
            logging.debug(f"{self.description} took {self.elapsed_time:.2f} seconds")
        return self.elapsed_time
        