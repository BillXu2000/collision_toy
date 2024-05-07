
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
                    print(arr)
                    return True
    return False

run([5, 5, 13, 13], 24)
