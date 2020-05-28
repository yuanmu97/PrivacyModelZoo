import tesserocr
from PIL import Image
import sys 

print(tesserocr.tesseract_version())  # print tesseract-ocr version
print(tesserocr.get_languages())  # prints tessdata path and list of available languages

image = Image.open(sys.argv[1])
print(tesserocr.image_to_text(image))  # print ocr text from image
# or
# print(tesserocr.file_to_text('sample.jpg'))