import sys
sys.path.insert(0, r"/home/CI_Pipeline/AI_Glasses/Jetson Code/_bench_overlay")
sys.path.insert(1, r"/home/CI_Pipeline/AI_Glasses/Jetson Code/Code")
import BLE as _ble_stub
sys.modules["BLE"] = _ble_stub
print("sys.path:", sys.path[:5])
exec(open(r"/home/CI_Pipeline/AI_Glasses/Jetson Code/Code/main.py").read())
