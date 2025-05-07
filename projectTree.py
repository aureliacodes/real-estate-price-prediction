import os

def print_tree(directory, indent=""):
    files = sorted(os.listdir(directory))
    for file in files:
        path = os.path.join(directory, file)
        if os.path.isdir(path):
            print(f"{indent}+---{file}")
            print_tree(path, indent + "|   ")
        else:
            print(f"{indent}|   {file}")

# Specifică calea proiectului tău
project_path = "C:\\Users\\Aurelia\\Desktop\\WORKS\\real-estate-price-prediction"
print_tree(project_path)
