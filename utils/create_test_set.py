import os
import random
import shutil

import connection_settings as cs

# IMPORTANT: Only run this once!

subdir_list = ["Barren", "Cultivated", "Developed", "Forest", "Herbaceous", "Shrub"]

valdir = os.path.join(cs.IMAGE_DIR, "ValData")
testdir = os.path.join(cs.IMAGE_DIR, "TestData")

for subdir in subdir_list:
    val = os.path.join(valdir, subdir)
    test = os.path.join(testdir, subdir)
    if not os.path.exists(test):
        os.makedirs(test)
    files = os.listdir(val)
    files_to_move = random.sample(files, len(files)//2)
    for file in files_to_move:
        old = os.path.join(val, file)
        new = os.path.join(test, file)
        dest = shutil.move(old, new)
    print(subdir)
    print("val: ", len(os.listdir(val)))
    print("test:", len(os.listdir(test)))