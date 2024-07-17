# This is a sample Python script.
import cv2


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    file_path = 'HPT_video1.mp4'
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # frames_to_skip = int(fps * sample_frequency)
    print("yo")
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    print("yo")
    i = 0
    while (cap.isOpened()):
        # print("yo")
        ret, frame = cap.read()
        if (ret == True and frame is not None and (i > 300)):
            cv2.imwrite('frame_hpt1_' + str(i-300) + '.png', frame)
            print(i)
        i+=1
    # cv2.imshow('Frame', frame)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
