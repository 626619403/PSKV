import time
import psutil
import threading
from contextlib import contextmanager
import torch

class ResourceMonitor:
    def __init__(self):
        self.peak_cpu_mb = 0
        self.peak_gpu_reserved_mb = {}   

        self.peak_system_reserved_mb = 0 
        
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.monitoring = False
        self.monitor_thread = None
        
        for i in range(self.num_gpus):
            self.peak_gpu_reserved_mb[i] = 0
        
    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        while self.monitoring:
            try:

                process = psutil.Process()
                current_cpu_mb = process.memory_info().rss / 1024 / 1024
                self.peak_cpu_mb = max(self.peak_cpu_mb, current_cpu_mb)

                if torch.cuda.is_available():
                    current_moment_total_reserved = 0  
                    
                    for i in range(self.num_gpus):

                        reserved_mb = torch.cuda.memory_reserved(i) / 1024 / 1024

                        current_moment_total_reserved += reserved_mb

                        self.peak_gpu_reserved_mb[i] = max(self.peak_gpu_reserved_mb[i], reserved_mb)

                    self.peak_system_reserved_mb = max(self.peak_system_reserved_mb, current_moment_total_reserved)
                
                time.sleep(0.5)
            except Exception:
                pass
    
    def get_stats(self):
        return {
            'peak_cpu_mb': self.peak_cpu_mb,
            'peak_gpu_reserved_mb': self.peak_gpu_reserved_mb,    
            'peak_system_reserved_mb': self.peak_system_reserved_mb, 
            'num_gpus': self.num_gpus
        }
    
    def get_total_gpu_allocated(self):
        
        if not torch.cuda.is_available():
            return 0
        total = 0
        for i in range(self.num_gpus):
            total += torch.cuda.memory_allocated(i)
        return total / 1024 / 1024  # MB
        
@contextmanager
def resource_monitor(logger):
    monitor = ResourceMonitor()
    start_time = time.time()
    
    try:
        monitor.start_monitoring()
        yield monitor
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        raise e
    finally:
        monitor.stop_monitoring()
        end_time = time.time()
        total_time = end_time - start_time
        
        stats = monitor.get_stats()

        logger.info("=" * 50)
        logger.info("RESOURCE USAGE SUMMARY (CACHE / RESERVED)")
        logger.info("=" * 50)
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Peak CPU memory usage: {stats['peak_cpu_mb']:.2f} MB")
        
        if torch.cuda.is_available():

            for i in range(stats['num_gpus']):
                logger.info(f"Peak GPU {i} Reserved (Cache): {stats['peak_gpu_reserved_mb'][i]:.2f} MB")
            

            logger.info("-" * 30)
            logger.info(f"System-wide Peak Reserved Memory: {stats['peak_system_reserved_mb']:.2f} MB")
            logger.info("-" * 30)

            
            for i in range(stats['num_gpus']):
                allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                logger.info(f"GPU {i} Final State -> Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
            
        else:
            logger.info("GPU not available")
        
        logger.info("=" * 50)