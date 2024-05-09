
def run(arr, target):
    if len(arr) == 1 and abs(arr[0] - target) < 1e-6:
        return True
    if len(arr) == 1:
        return False
    ops = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / y if y != 0 else 0]
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i == j: continue
            for op in ops:
                # print(len(arr), i, j)
                k = op(arr[i], arr[j])
                tmp = [i for i in arr]
                del tmp[max(i, j)]
                del tmp[min(i, j)]
                ans = run(tmp + [k], target)
                if ans:
                    # print(arr)
                    return True
    return False

for i in range(1, 13):
    for j in range(1, i + 1):
        for k in range(1, j + 1):
            for l in range(1, k + 1):
                ans = run([i, j, k, l], 24)
                if not ans: print(i, j, k, l)
