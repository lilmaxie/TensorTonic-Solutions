import numpy as np

def decision_tree_split(X: list, y: list) -> list:
    """
    Returns the best feature index and threshold.
    """
    # 1. Chuyển đổi dữ liệu sang mảng NumPy
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y)

    n_samples, n_features = X_mat.shape

    def _gini(labels: np.ndarray) -> float:
        """Helper tính độ vẩn đục Gini cho một tập nhãn."""
        n = len(labels)
        if n == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / n
        return float(1.0 - np.sum(probs ** 2))

    # 2. Tính Gini của nút cha ban đầu
    parent_gini = _gini(y_vec)

    best_gain = -1.0
    best_feature = None
    best_threshold = None

    # 3. Duyệt qua từng đặc trưng (từng cột)
    for j in range(n_features):
        col = X_mat[:, j]
        unique_vals = np.unique(col)

        # Nếu tất cả các mẫu đều có cùng một giá trị thì không thể phân nhánh
        if len(unique_vals) <= 1:
            continue

        # 4. Tính các ngưỡng trung điểm giữa các giá trị duy nhất liên tiếp
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for th in thresholds:
            left_mask = col <= th
            y_left = y_vec[left_mask]
            y_right = y_vec[~left_mask]

            n_l = len(y_left)
            n_r = len(y_right)

            # Bỏ qua phép chia nếu một bên không có mẫu
            if n_l == 0 or n_r == 0:
                continue

            # 5. Tính Gini có trọng số và Information Gain
            gini_split = (n_l / n_samples) * _gini(y_left) + (n_r / n_samples) * _gini(y_right)
            gain = parent_gini - gini_split

            # 6. Cập nhật nghiệm tối ưu và xử lý hòa điểm (Tie-breaking)
            # Dùng ngưỡng sai số 1e-9 để tránh sai lệch dấu phẩy động
            if gain > best_gain + 1e-9:
                best_gain = gain
                best_feature = j
                best_threshold = float(th)
            elif abs(gain - best_gain) <= 1e-9:
                # Nếu gain bằng nhau: chọn feature index nhỏ hơn, rồi đến threshold nhỏ hơn
                if best_feature is None or j < best_feature:
                    best_gain = gain
                    best_feature = j
                    best_threshold = float(th)
                elif j == best_feature and th < best_threshold:
                    best_gain = gain
                    best_feature = j
                    best_threshold = float(th)

    return [best_feature, best_threshold]