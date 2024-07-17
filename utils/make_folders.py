
import os
import shutil
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.

    for k in range(100):
        os.makedirs("subsection_"+str(k), exist_ok=True)
        print(os.getcwd())
        for l in range(40*k+1, 40*(k+1)+1):
            shutil.move(r"C:/Users/bowmo/Documents/ademi/code/utils/data_no_bg/frame_"+str(l)+"_no_bg.png", os.path.join("subsection_"+str(k)))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')