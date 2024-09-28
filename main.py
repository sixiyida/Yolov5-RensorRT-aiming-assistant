import os
import sys
from pathlib import Path
# import serial
import torch
import win32api
import win32con
from grabscreen import grab_screen
from PID import PID
import ghub_mouse as ghub
import pyautogui as pag
from ctypes import *
import numpy as np
import numpy.ctypeslib as npct
import d3dshot
import cv2
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

com_text = ""
configs_dict = {}
config_list = []
with open('configs.txt', 'r', encoding="utf-8") as f:
    for config_line in f:
        config_list.append(list(config_line.strip('\n').split(',')))
f.close()
config_list.remove(['# 范围调节'])
config_list.remove(['# PID控制调节'])
for i in range(10):
    config_list.remove([''])
config_list.remove([''])
index1 = config_list[0][0].find("=")
index2 = config_list[0][0].find("#")
# com参数
com_text = config_list[0][0][index1 + 1:index2].strip()
del config_list[0]
# 1-9参数
print(com_text)
print("配置读取如下\n*************************************************")
last_configs_list = []
for i in range(len(config_list)):
    index1 = config_list[i][0].find("=")
    index2 = config_list[i][0].find("#")
    last_configs_list.append(float(config_list[i][0][index1 + 1:index2]))
    configs_dict[i + 1] = float(config_list[i][0][index1 + 1:index2])

print(f"配置写入：{configs_dict}")

# ser = serial.Serial(f'{com_text}', 115200)
# ser.write('import km\r\n'.encode('utf-8'))

time.sleep(0.1)

# print('kmbox 成功导入模块:', str(ser.read(ser.inWaiting()), 'utf-8'))


y_correction_factor = configs_dict[1]  # 截图位置修正， 值越大截图窗口向上
x_correction_factor = 0  # 截图位置修正， 值越大截图窗口向右移动
screen_x, screen_y = configs_dict[2], configs_dict[3]  # 电脑显示器分辨率
window_x, window_y = configs_dict[4], configs_dict[5]  # x,y 截图窗口大小
screen_x_center = screen_x / 2
screen_y_center = screen_y / 2
PID_time = configs_dict[6]
Kp = configs_dict[7]
Ki = configs_dict[8]
Kd = configs_dict[9]
y_portion = configs_dict[10]  # 数值越小，越往下， 从上往下头的比例距离
max_step = configs_dict[11]  # 每次位移的最大步长
pid = PID(PID_time, max_step, -max_step, Kp, Ki, Kd)

grab_window_location = (
    int(screen_x_center - window_x / 2 + x_correction_factor),
    int(screen_y_center - window_y / 2 - y_correction_factor),
    int(screen_x_center + window_x / 2 + x_correction_factor),
    int(screen_y_center + window_y / 2 - y_correction_factor))

edge_x = screen_x_center - window_x / 2
edge_y = screen_y_center - window_y / 2

# 自瞄范围设置
aim_x = configs_dict[13]  # aim width
aim_x_left = int(screen_x_center - aim_x / 2)  # 自瞄左右侧边距
aim_x_right = int(screen_x_center + aim_x / 2)

aim_y = configs_dict[14]  # aim width
aim_y_up = int(screen_y_center - aim_y / 2 - y_correction_factor)  # 自瞄上下侧边距
aim_y_down = int(screen_y_center + aim_y / 2 - y_correction_factor)
time.sleep(2)


@torch.no_grad()

def visualize(img,bbox_array):
    for temp in bbox_array:
        bbox = [temp[0],temp[1],temp[2],temp[3]]  #xywh
        cv2.rectangle(img,(int(temp[0]-0.5*temp[2]),int(temp[1]-0.5*temp[3])),(int(temp[0]+0.5*temp[2]),int(temp[1]+0.5*temp[3])), (105, 237, 249), 2)
    return img


class Detector():
    def __init__(self,model_path,dll_path):
        self.yolov5 = CDLL(dll_path)
        self.yolov5.Detect.argtypes = [c_void_p,c_int,c_int,POINTER(c_ubyte),npct.ndpointer(dtype = np.float32, ndim = 2, shape = (50, 6), flags="C_CONTIGUOUS")]
        self.yolov5.Init.restype = c_void_p
        self.yolov5.Init.argtypes = [c_void_p]
        self.yolov5.cuda_free.argtypes = [c_void_p]
        self.c_point = self.yolov5.Init(model_path)

    def predict(self,img):
        rows, cols = img.shape[0], img.shape[1]
        res_arr = np.zeros((50,6),dtype=np.float32)
        self.yolov5.Detect(self.c_point,c_int(rows), c_int(cols), img.ctypes.data_as(POINTER(c_ubyte)),res_arr)
        self.bbox_array = res_arr[~(res_arr==0).all(1)]
        return self.bbox_array

    def free(self):
        self.yolov5.cuda_free(self.c_point)

def visualize(img,bbox_array):
    for temp in bbox_array:
        bbox = [temp[0],temp[1],temp[2],temp[3]]  #xywh
        cv2.rectangle(img,(int(temp[0]),int(temp[1])),(int(temp[0]+temp[2]),int(temp[1]+temp[3])), (105, 237, 249), 2)
    return img

det = Detector(model_path=b"apex2.engine",dll_path=r"apex.dll")

def find_target():
    last_state=0
    is_active=0
   #cv2.namedWindow('cs', cv2.WINDOW_NORMAL)
    counter = 0
    start_time = time.time()
    while True:
        counter += 1
        timegap=time.time()-start_time
        if (timegap):
            print(float('%.1f' % (counter / timegap)))
        img0 = grab_screen(grab_window_location)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)
        result = det.predict(img0)

        target_distance_list = []
        target_xywh_list = []

        if counter%7==0:
            now_state = win32api.GetAsyncKeyState(win32con.VK_XBUTTON2)
            print(now_state)
            if now_state<0 and not last_state:
              is_active = not is_active
              print(f"当前状态：{is_active}")
            last_state = now_state

        if len(result):
            # print('move 回码：', str(ser.read(ser.inWaiting()), 'utf-8'))
            img0 = visualize(img0, result)
            for temp in result:
                #print([temp[0], temp[1], temp[2], temp[3]])
                xywh=([temp[0]+0.5*temp[2], temp[1]+0.5*temp[3], temp[2], temp[3]])
                # print('\033[0;40;40m' + f'   xywh = {xywh}   \n' + '\033[0m')
                target_xywh_list.append(xywh)
                target_distance = abs(edge_x + xywh[0] - screen_x_center)+abs(edge_y + xywh[1] - screen_y_center)
                target_distance_list.append(target_distance)
            # print(f"target_distance_list= {target_distance_list}")
            min_index = target_distance_list.index(min(target_distance_list))
            target_xywh = target_xywh_list[min_index]

            target_xywh_x = target_xywh[0] + edge_x
            target_xywh_y = target_xywh[1] + edge_y
            #print('\033[0;33;40m' + f"target-X = {target_xywh_x}  target—Y = {target_xywh_y}" + '\033[0m')

            if aim_x_left < target_xywh_x < aim_x_right and aim_y_up < target_xywh_y < aim_y_down:

                if configs_dict[12] == 3:
                    aim_mouse = win32api.GetAsyncKeyState(win32con.VK_RBUTTON) \
                                or win32api.GetAsyncKeyState(win32con.VK_LBUTTON)
                elif configs_dict[12] == 2:
                    aim_mouse = win32api.GetAsyncKeyState(win32con.VK_RBUTTON)

                elif configs_dict[12] == 1:
                    aim_mouse = win32api.GetAsyncKeyState(win32con.VK_LBUTTON)

                else:
                    print("请填入正确的鼠标瞄准模式数字 1 或 2 或 3, Please fill the correct aim mod number 1 or 2 or 3")
                    break


                if aim_mouse and is_active:
                    # 鼠标计算相对移动距离 (calculate mouse relative move distance)
                    x, y = pag.position()
                    pid_x = int(pid.calculate(target_xywh_x, x))
                    pid_y = int(pid.calculate(target_xywh_y-y_portion*target_xywh[3], y))
                    #pid_y = int(pid.calculate(target_xywh_y, y))
                    print(f"Mouse-Move X Y = ({pid_x}, {pid_y})")
                    ghub.mouse_xy(pid_x,pid_y)
                    # ser.write(f'km.move({pid_x},{pid_y})\r\n'.encode('utf-8'))
                else:
                    pid.set_zero_integral()



        #else:
            #print('\033[0;31;40m' + f'  no target   ' + '\033[0m')
        cv2.imshow("cs", img0)
        cv2.waitKey(1)

    # Print time (total circle)



if __name__ == "__main__":
    find_target()
