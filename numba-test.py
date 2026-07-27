
from numba import njit, prange
import numpy as np
import time

CHARSET = np.arange(32, 126, dtype=np.uint8)
BASE = 94
TOTAL = BASE ** 4

@njit
def hash_rev(c0, c1, c2, c3, c19):
    h = np.uint64(0)
    h = h * 31 + np.uint64(c0)
    h = h * 31 + np.uint64(c1)
    h = h * 31 + np.uint64(c2)
    h = h * 31 + np.uint64(c3)
    return (h ^ np.uint64(c19)) & np.uint64(0xFFFFFFFF)

@njit
def bin_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] < target:
            left = mid + 1
        elif arr[mid] > target:
            right = mid
        else:
            return True
    return False

# ============ 方案：并行生成布尔掩码 + numpy 收集结果 ============
@njit(parallel=True)
def search_parallel_mask(c19, forward_hashes, charset, mask):
    n = len(charset)
    total = n ** 4
    for idx in prange(total):
        i3 = idx % n
        t1 = idx // n
        i2 = t1 % n
        t2 = t1 // n
        i1 = t2 % n
        i0 = t2 // n
        h = hash_rev(charset[i0], charset[i1], charset[i2], charset[i3], c19)
        if bin_search(forward_hashes, h):
            mask[idx] = True

# 生成 forward 表
np.random.seed(42)
forward = np.random.randint(0, 2**32, size=1_000_000, dtype=np.uint64)
forward.sort()

# 预热
print("编译预热...")
mask_test = np.zeros(1000, dtype=np.bool_)
search_parallel_mask(np.uint64(0x1234), forward[:100], CHARSET, mask_test)
print("编译完成")

# 测试完整流程
print(f"\n开始并行搜索 {TOTAL:,} 条...")
t0 = time.time()

# 1. 并行生成掩码
mask = np.zeros(TOTAL, dtype=np.bool_)
search_parallel_mask(np.uint64(0xDEADBEEF), forward, CHARSET, mask)
t1 = time.time()

# 2. 收集匹配索引
matched_indices = np.nonzero(mask)[0]
t2 = time.time()

# 3. 解码为字符（向量化）
n = BASE
idx = matched_indices
c3 = idx % n
t1_arr = idx // n
c2 = t1_arr % n
t2_arr = t1_arr // n
c1 = t2_arr % n
c0 = t2_arr // n

results = np.column_stack([CHARSET[c0], CHARSET[c1], CHARSET[c2], CHARSET[c3]])
t3 = time.time()

print(f"  匹配数: {len(matched_indices)}")
print(f"  并行搜索: {t1-t0:.3f}s")
print(f"  np.nonzero: {t2-t1:.3f}s")
print(f"  解码结果: {t3-t2:.3f}s")
print(f"  总耗时: {t3-t0:.3f}s")
print(f"  预估 12 个 classname: {12*(t3-t0):.1f}s")
print(f"\n  前5个结果（bytes）:")
for i in range(min(5, len(results))):
    print(f"    {bytes(results[i])}")
