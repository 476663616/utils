import SimpleITK as sitk
import numpy as np


class findVesselTerminal():
    '''
    从细化的nii图像中找到左冠和右冠的终端
    '''
    def __init__(self) -> None:
        # 血管末端的像素坐标
        self.ax_left = []
        self.ay_left = []
        self.az_left = []
        
        self.theta = None          # 细化体素的邻域中其他体素之间的夹角
        self.thetaThreshold = 85   # 夹角阈值，若一旦有夹角超过85，则不是顶端
        self.t = []                # 夹角阈值
        self.num = 7               # 邻域

    def angle_cul(self, point1, point2):
        # 计算两个向量的角度
        data_M = np.sqrt(np.sum(point1*point1,axis=0))
        data_N = np.sqrt(np.sum(point2*point2,axis=0))
        cos_theta = np.sum(point1 * point2, axis=0)/(data_M*data_N)
        self.theta = np.degrees(np.arccos(cos_theta))
        return self.theta

    def is_terminal(self, cube):
        # 判断是不是终端
        f, h, w = cube.shape
        points = []
        for x in range(h):
            for y in range(w):
                for z in range(f):
                    if cube[z][x][y] == 1:
                        point = np.zeros((3, 1))
                        point[0] = z - f//2
                        point[1] = x - h//2
                        point[2] = y - w//2

                        points.append(point)
        self.t = []
        # 若除了中心体素，只有一个体素是细化体素时，返回空列表
        if len(points) < 2:
            return self.t

        # 计算所有细化体素与中心体素向量之间的夹角
        for i in range(len(points)-1):
            for j in range(i+1, len(points)):
                theta = self.angle_cul(points[i], points[j])
                self.t.append(theta)

        return self.t

    def forward(self, image_array):
        # image_array = sitk.GetArrayFromImage(image)
        f, h, w = image_array.shape  # 注意顺序
        ax = []
        ay = []
        az = []
        dx = []
        dy = []
        dz = []

        for x in range(h):
            for y in range(w):
                for z in range(f):
                    if image_array[z][x][y] == 1:  # 注意顺序
                        is_term = True
                        # 取邻域计算夹角  注意顺序
                        t = self.is_terminal(image_array[z-(self.num//2):z+(self.num//2)+1, 
                                                    x-(self.num//2):x+(self.num//2)+1, 
                                                    y-(self.num//2):y+(self.num//2)+1])
                        if t == []:
                            is_term = False
                            print("increase the num")
                        else:
                            for i in range(len(t)):
                                if t[i] > self.thetaThreshold:
                                    is_term = False
                                    break

                        if is_term == True:
                            ax.append(x)
                            ay.append(y)
                            az.append(z)
                        else:
                            dx.append(x)
                            dy.append(y)
                            dz.append(z)

        return ax, ay, az