import time

class RateLimiter:
    def __init__(self, max_requests=5, interval=60):
        self.max_requests = max_requests
        self.interval = interval
        self.first_request_timestamp = None
        self.request_count = 0

    def wait_if_needed(self):
        current_time = time.time()

        if self.first_request_timestamp is None:
            self.first_request_timestamp = current_time

        elapsed = current_time - self.first_request_timestamp

        if elapsed > self.interval:
            # Reset window
            self.first_request_timestamp = current_time
            self.request_count = 0

        if self.request_count >= self.max_requests:
            wait_time = self.interval - elapsed
            print(f"Rate limit hit. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            # Reset after waiting
            self.first_request_timestamp = time.time()
            self.request_count = 0

        self.request_count += 1
