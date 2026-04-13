import numpy as np



def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif x == b:
        return 1
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def fuzzy_membership(score):
    # Không bẩn
    clean = triangular(score, 0.0, 0.0, 0.4)

    # Nhẹ
    low = triangular(score, 0.2, 0.5, 0.7)

    # Trung bình
    medium = triangular(score, 0.5, 0.75, 0.9)

    # Nặng
    high = triangular(score, 0.7, 1.0, 1.0)

    return {
        "clean": clean,
        "low": low,
        "medium": medium,
        "high": high
    }




def fuzzy_rules(score):
    m = fuzzy_membership(score)

    # Lấy giá trị lớn nhất
    max_label = max(m, key=m.get)

    if max_label == "clean":
        return "Không có vết bẩn"
    elif max_label == "low":
        return "Vết bẩn nhẹ"
    elif max_label == "medium":
        return "Vết bẩn trung bình"
    else:
        return "Vết bẩn nặng"