import sys
from PIL import Image

def show_image(image_name):
    im = Image.open(image_name)
    im.show()

if __name__=='__main__':
    show_image(str(sys.argv[1]))
