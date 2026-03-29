"""
Atlas 200I DK A2 资源监控

后台线程定时采样：
  - NPU 算力 (AICore %)     ← npu-smi info
  - NPU 显存 (MB used/total) ← npu-smi info
  - 系统内存 (MB used/total) ← /proc/meminfo 或 psutil

用法:
    mon = ResourceMonitor(interval=1.0)
    mon.start()
    # ... 推理循环 ...
    stats = mon.stop()
    ResourceMonitor.print_summary(stats)
"""

import re
import subprocess
import threading
import time

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# 底层采样函数
# ---------------------------------------------------------------------------

def _npu_smi_info(chip_id: int = 0):
    """
    调用 npu-smi info，解析芯片级指标。
    返回 (aicore_pct, mem_used_mb, mem_total_mb) 或 (None, None, None)。
    行格式示例:
      | 0       0         | 0000:01:00.0    | 5           406  / 3931       |
    """
    try:
        raw = subprocess.check_output(
            ["npu-smi", "info"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode("utf-8", errors="ignore")
    except Exception:
        return None, None, None

    # chip 行：| chip_id  device_id | bus_id | aicore%  mem_used / mem_total |
    pat = re.compile(
        r'\|\s*' + str(chip_id) + r'\s+\d+\s*\|'
        r'\s*[\w:\.]+\s*\|\s*(\d+)\s+(\d+)\s*/\s*(\d+)'
    )
    for line in raw.splitlines():
        m = pat.search(line)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


def _sys_mem_mb():
    """返回 (used_mb, total_mb)，失败返回 (None, None)。"""
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        return vm.used >> 20, vm.total >> 20
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                info[k.strip()] = int(v.strip().split()[0])  # kB
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        return (total - avail) >> 10, total >> 10  # kB → MB
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# ResourceMonitor 类
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """
    后台采样线程；提供 current()（实时快照）和 stop()（汇总统计）。
    """

    def __init__(self, interval: float = 1.0, chip_id: int = 0):
        self.interval = interval
        self.chip_id = chip_id
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._data = {
            "aicore": [],
            "npu_used": [],
            "npu_total": 0,
            "sys_used": [],
            "sys_total": 0,
        }

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            aicore, npu_used, npu_total = _npu_smi_info(self.chip_id)
            sys_used, sys_total = _sys_mem_mb()
            with self._lock:
                d = self._data
                if aicore is not None:
                    d["aicore"].append(aicore)
                if npu_used is not None:
                    d["npu_used"].append(npu_used)
                    d["npu_total"] = npu_total or d["npu_total"]
                if sys_used is not None:
                    d["sys_used"].append(sys_used)
                    d["sys_total"] = sys_total or d["sys_total"]
            time.sleep(self.interval)

    def current(self) -> dict:
        """返回最新一次采样值（用于实时打印）。"""
        with self._lock:
            d = self._data
            return {
                "aicore_pct":      d["aicore"][-1]   if d["aicore"]   else None,
                "npu_used_mb":     d["npu_used"][-1] if d["npu_used"] else None,
                "npu_total_mb":    d["npu_total"],
                "sys_used_mb":     d["sys_used"][-1] if d["sys_used"] else None,
                "sys_total_mb":    d["sys_total"],
            }

    def stop(self) -> dict:
        """停止采样线程，返回汇总统计字典。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        def avg(lst): return sum(lst) / len(lst) if lst else None
        def peak(lst): return max(lst) if lst else None

        with self._lock:
            d = self._data
            return {
                "aicore_avg":      avg(d["aicore"]),
                "aicore_peak":     peak(d["aicore"]),
                "npu_mem_avg_mb":  avg(d["npu_used"]),
                "npu_mem_peak_mb": peak(d["npu_used"]),
                "npu_mem_total_mb": d["npu_total"],
                "sys_mem_avg_mb":  avg(d["sys_used"]),
                "sys_mem_peak_mb": peak(d["sys_used"]),
                "sys_mem_total_mb": d["sys_total"],
                "samples":         len(d["aicore"]),
            }

    @staticmethod
    def format_current(cur: dict) -> str:
        """把 current() 结果格式化成单行字符串，便于嵌入循环日志。"""
        parts = []
        if cur.get("aicore_pct") is not None:
            parts.append(f"NPU算力={cur['aicore_pct']}%")
        if cur.get("npu_used_mb") is not None:
            total = f"/{cur['npu_total_mb']}MB" if cur["npu_total_mb"] else ""
            parts.append(f"NPU显存={cur['npu_used_mb']}{total}")
        if cur.get("sys_used_mb") is not None:
            total = f"/{cur['sys_total_mb']}MB" if cur["sys_total_mb"] else ""
            parts.append(f"系统内存={cur['sys_used_mb']}{total}")
        return "  ".join(parts) if parts else "（监控数据不可用）"

    @staticmethod
    def print_summary(stats: dict):
        """打印汇总统计表。"""
        print("\n" + "=" * 52)
        print("资源使用汇总")
        print("=" * 52)

        npu_total = stats.get("npu_mem_total_mb") or 0
        sys_total = stats.get("sys_mem_total_mb") or 0

        if stats.get("aicore_avg") is not None:
            print(f"  NPU 算力    均值: {stats['aicore_avg']:5.1f}%   "
                  f"峰值: {stats['aicore_peak']:5.1f}%")
        else:
            print("  NPU 算力    无法获取（npu-smi 不可用）")

        if stats.get("npu_mem_avg_mb") is not None:
            tot = f"/ {npu_total} MB" if npu_total else ""
            print(f"  NPU 显存    均值: {stats['npu_mem_avg_mb']:5.0f} MB  "
                  f"峰值: {stats['npu_mem_peak_mb']:5.0f} MB  {tot}")
        else:
            print("  NPU 显存    无法获取")

        if stats.get("sys_mem_avg_mb") is not None:
            tot = f"/ {sys_total} MB" if sys_total else ""
            print(f"  系统内存    均值: {stats['sys_mem_avg_mb']:5.0f} MB  "
                  f"峰值: {stats['sys_mem_peak_mb']:5.0f} MB  {tot}")
        else:
            print("  系统内存    无法获取")

        print(f"  采样次数: {stats.get('samples', 0)}")
        print("=" * 52)
