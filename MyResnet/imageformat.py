import Image
import glob
import os

def jpg_to_png(dir1, dir2):
    fpath = os.path.join(dir1, '*.jpg')
    file_list = glob.glob(fpath)

    for i in range(len(file_list)):
	jpeg_r = Image.open(file_list[i])
	name = file_list[i].replace(".jpg", ".png")
	name = name.replace(dir1, dir2)
	jpeg_r.save(name)

    return 0

dirname="BSR/BSDS500/data/images"
dir2="BSDS500_PNG"

print("Start changing image format")
fpath = os.path.join(dirname, 'train')
fpath2 = os.path.join(dir2, 'train')
jpg_to_png(fpath, fpath2)
print("Finish Train Data Changing")

fpath = os.path.join(dirname, 'test')
fpath2 = os.path.join(dir2, 'test')
jpg_to_png(fpath, fpath2)
print("Finish Test Data Changing")

fpath = os.path.join(dirname, 'val')
fpath2 = os.path.join(dir2, 'val')
jpg_to_png(fpath, fpath2)
print("Finish Valdidation Data Changing")

