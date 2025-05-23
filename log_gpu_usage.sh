#!/bin/bash

nvidia-smi --query-gpu=timestamp,name,pci.bus_id,utilization.gpu,utilization.memory --format=csv -lms 1000 > gpu_usage.log
