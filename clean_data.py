import os

def clean_and_validate_dataset(dataset_path):
    valid_extensions = ['.jpg', '.png']

    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy thư mục '{dataset_path}'. Vui lòng tạo và cho ảnh vào.")
        return

    print("KIỂM TRA VÀ DỌN DẸP DỮ LIỆU...\n" + "-"*40)

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)

        if not os.path.isdir(folder_path):
            continue

        valid_images_count = 0

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            if os.path.isfile(file_path):
                ext = os.path.splitext(file_name)[1].lower()
                
                if ext in valid_extensions:
                    valid_images_count += 1
                else:
                    print(f"[XÓA FILE RÁC] Đã xóa: {file_path}")
                    os.remove(file_path)

        if valid_images_count < 5:
            print(f"CẢNH BÁO: Thư mục '{folder_name}' chỉ có {valid_images_count} ảnh hợp lệ. Cần bổ sung thêm để đạt ít nhất 5 ảnh!")
        else:
            print(f"HỢP LỆ: Thư mục '{folder_name}' đã có {valid_images_count} ảnh.")

    print("-" * 40 + "\nHOÀN TẤT!")

clean_and_validate_dataset('dataset')