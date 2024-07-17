from PIL import Image
import os

def convert_bit_depth(directory):
    i = 0
    print(directory)
    for filename in os.listdir(directory):
        print(filename)
        filenum = (filename[2:])[:-4]
        print(filenum)
        img = Image.open(directory+filename)
        new_img = img.convert("P", palette=Image.ADAPTIVE, colors=64)
        new_img.save("./test_bit_conversion/r_"+str(filenum)+".png")
        i += 1

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    convert_bit_depth(os.getcwd() + "\\test_bit_conversion\\")