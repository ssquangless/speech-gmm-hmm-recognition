import os

# Danh sách các thư mục và file muốn BỎ QUA
IGNORE_DIRS = {'.venv_speech', 'venv_speech', '__pycache__', '.git', '.vscode', '.idea', 'build', 'dist','data','debug_audio'}
IGNORE_FILES = {'.DS_Store', 'desktop.ini'}

def draw_tree(directory, prefix=""):
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        return

    # Lọc bỏ thư mục/file không cần thiết
    entries = [e for e in entries if e not in IGNORE_DIRS and e not in IGNORE_FILES]
    
    count = len(entries)
    for index, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        is_last = (index == count - 1)
        connector = "└── " if is_last else "├── "
        
        print(f"{prefix}{connector}{entry}")
        
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            draw_tree(path, prefix + extension)

if __name__ == "__main__":
    print(".")
    draw_tree(".")