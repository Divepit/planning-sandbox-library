import time

class Benchmark:
    def __init__(self, description, start_now=False):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        if start_now:
            self.start()

    def start(self):
        print(f"Starting benchmark: {self.description}")
        if self.start_time is not None:
            raise Exception("Benchmark already started")
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise Exception("Benchmark not started")
        elif self.end_time is not None:
            raise Exception("Benchmark already stopped")
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.description} took {self.elapsed_time:.2f} seconds")
        