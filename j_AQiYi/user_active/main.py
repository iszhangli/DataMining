# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_dir = "E:/Dataset/爱奇艺用户留存预测/"
    video_info_dir = input_dir + 'video_related_data.csv'

    video_info = pd.read_csv(video_info_dir, nrows=10000)

    video_info['father_id'] = video_info.father_id.astype('object')

    video_info.father_id.str.split(';')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
