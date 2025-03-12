import time
import random
import json
from datetime import datetime
from collections import deque
import threading

class DataSource:
    def __init__(self, source_id):
        self.source_id = source_id
        self.data_queue = deque()
        self.is_running = True

    def generate_data(self):
        while self.is_running:
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'source_id': self.source_id,
                'value': random.uniform(1.0, 100.0)
            }
            self.data_queue.append(data_point)
            time.sleep(random.uniform(0.1, 1.0))

    def stop(self):
        self.is_running = False

class DataProcessor(threading.Thread):
    def __init__(self, data_source, output_queue):
        super().__init__()
        self.data_source = data_source
        self.output_queue = output_queue
        self.is_running = True

    def run(self):
        while self.is_running:
            if self.data_source.data_queue:
                data_point = self.data_source.data_queue.popleft()
                filtered_data = self.filter_data(data_point)
                if filtered_data:
                    self.output_queue.append(filtered_data)
            time.sleep(0.05)

    def filter_data(self, data_point):
        if data_point['value'] > 50:
            return data_point
        return None

    def stop(self):
        self.is_running = False

class Aggregator(threading.Thread):
    def __init__(self, input_queue):
        super().__init__()
        self.input_queue = input_queue
        self.is_running = True
        self.aggr_data = []

    def run(self):
        while self.is_running:
            if self.input_queue:
                data_point = self.input_queue.pop(0)
                self.aggr_data.append(data_point)
                if len(self.aggr_data) >= 5:
                    self.aggregate()
            time.sleep(1)

    def aggregate(self):
        avg_value = sum(item['value'] for item in self.aggr_data) / len(self.aggr_data)
        print(f"Aggregated data: Avg Value = {avg_value:.2f} from {len(self.aggr_data)} entries")
        self.aggr_data.clear()

    def stop(self):
        self.is_running = False

def main():
    data_sources = [DataSource(i) for i in range(3)]
    output_queue = []
    
    processors = [DataProcessor(source, output_queue) for source in data_sources]
    aggregator = Aggregator(output_queue)

    for source in data_sources:
        threading.Thread(target=source.generate_data, daemon=True).start()
    
    for processor in processors:
        processor.start()

    aggregator.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping all processes...")
        for source in data_sources:
            source.stop()
        for processor in processors:
            processor.stop()
        aggregator.stop()
    
    print("All processes stopped.")

if __name__ == "__main__":
    main()