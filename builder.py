import os.path
import shutil

classes = [
    'dislike',
    'like',
    'stop',
    'timeout'
]
annotations_dir = 'annotations'
m_train_dir = 'm_train'
m_val_dir = 'm_val'
cwd = os.getcwd()


for clazz in classes:
    train_class_path = os.path.join(cwd, m_train_dir, clazz)
    val_class_path = os.path.join(cwd, m_val_dir, clazz)
    if not os.path.exists(train_class_path) and not os.path.exists(val_class_path):
        os.makedirs(train_class_path)
        os.makedirs(val_class_path)

    annotation_class_dir = os.path.join(cwd, annotations_dir, clazz)
    old_train_class_path = os.path.join(cwd, 'train', clazz)
    old_val_class_path = os.path.join(cwd, 'val', clazz)
    c = os.listdir(annotation_class_dir)
    ann_class_files_no_ext = [f.split(".")[0] for f in c]
    for file_name in os.listdir(old_val_class_path):
        file_name_without_ext = os.path.splitext(file_name)[0]
        if ann_class_files_no_ext.__contains__(file_name_without_ext):
            shutil.copy(os.path.join(old_val_class_path, file_name), os.path.join(val_class_path, file_name))
