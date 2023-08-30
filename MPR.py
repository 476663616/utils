# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
from utils.centerline import vessel2ctl
import os


windowsize = 46 #偶数，越小速度越快
slicenum_min = 140 #生成拉直图像的最小层数，越小速度越快
slicenum_max = 600 #生成拉直图像的最大层数，越小速度越快


# 函数描述：将拉直图像还原到原始CT图像空间，简单的思路是对应坐标索引取整，这样的话就没必要采样距离很小或者中心线的任两点间距离很�?
def get_img_anti_straight(img_straight, img, index_interp):
    img_anti_straight = img_straight.reshape(img_straight.shape[0] * img_straight.shape[1] * img_straight.shape[2])
    imgArray_tmp = np.zeros(shape=sitk.GetArrayFromImage(img).shape, dtype=np.uint8)
    index_interp = (np.round(index_interp)).astype(np.int)
    for i in range(len(img_anti_straight)):
        if img_anti_straight[i] > 0.5:
            imgArray_tmp[index_interp[i, 0], index_interp[i, 1], index_interp[i, 2]] = 1

    return imgArray_tmp


# 函数描述：把世界坐标转换为图像索引�?
def WC2Index(position, spacing, origin):
    if position.shape[1] == 3:  # 如果输入纯三维坐标数�?
        t1 = position[:, 0]
        t2 = position[:, 1]
        t3 = position[:, 2]
        t1 = (t1 - origin[0]) / spacing[0]
        t2 = (t2 - origin[1]) / spacing[1]
        t3 = (t3 - origin[2]) / spacing[2]
        index = np.vstack([t1, t2, t3]).transpose()
    return index


# 函数描述：将最小路径获得的中心线进行b样条插值然后等间距取特定数量个�?
def get_centerline_interp(centerline_path, slicenum_min, slicenum_max):
    centerlinePoints = np.loadtxt(centerline_path)  # 最普通的loadtxt
    # print(centerlinePoints[0, 2])

    # # 绘制散点图
    # x = centerlinePoints[:, 0]
    # y = centerlinePoints[:, 1]
    # z = centerlinePoints[:, 2]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x, y, z)
    # # plt.show()

    if centerlinePoints[0, 0] > centerlinePoints[-1, 0]:  # 从下往上堆砌，对中心线起始点有要求
        centerlinePoints = centerlinePoints[::-1, :]

    # *********dim*10 = B样条点的数目***********
    num_interp_pts = np.int(centerlinePoints.shape[0]) * 3  ####################################!!!!!!!!!!!!!!
    x_sample = centerlinePoints[:, 2]
    y_sample = centerlinePoints[:, 1]
    z_sample = centerlinePoints[:, 0]

    # ***********todo:决定最终得到的MPR的切片个数**************
    # ***********cenpts_selected_number*****************
    if num_interp_pts >= slicenum_max:  # 针对slice数太多的数据
        cenpts_selected_number = slicenum_max
    elif num_interp_pts < slicenum_min:
        cenpts_selected_number = slicenum_min   # 针对slice数太少的数据
    else:
        cenpts_selected_number = np.int(num_interp_pts)  # slices nums, nii图像读进来是zyx顺序,dicom读入顺序是xyz
    print('cenpts_selected_number:  ', cenpts_selected_number)

    # todo:沿当前中心线构造B样条曲线
    # **************x_fine, y_fine, z_fine*******************
    tck, u = interpolate.splprep([x_sample, y_sample, z_sample],
                                 s=8 )  # s=30 Find the B-spline representation of an N-dimensional curve     ####################################!!!!!!!!!!!!!!
    u_fine = np.linspace(0, 1, cenpts_selected_number)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

    # ********todo:从B样条中选择距离相等的中心线点***************
    # *************cenpts_selected****************************
    cenpts_selected = np.zeros(shape=(cenpts_selected_number, 3), dtype=float)
    indx = 0
    for i in range(cenpts_selected_number):
        # indx = np.int(num_interp_pts / cenpts_selected_number * i)
        # cenpts_selected[i, :] = np.array([x_fine[indx], y_fine[indx], z_fine[indx]])
        cenpts_selected[i, :] = np.array([x_fine[i], y_fine[i], z_fine[i]])

    # 获得采样点之间平均距离
    array1 = cenpts_selected[10:-10, :]
    array2 = cenpts_selected[11:-9, :]
    meandistance = np.mean(np.sqrt(np.square(array1[:, 0] - array2[:, 0]) + np.square(array1[:, 1] - array2[:, 1]) + np.square(array1[:, 2] - array2[:, 2])))
    return cenpts_selected, meandistance


# 函数描述：拉直图像，根据中心线上中心点坐标以及对应的方向向量，通过求其对应的固定窗口尺寸的reslice图像，然后叠加为一个新的体数据
# 输入是CTA.nii图像数据，输出是拉直后的图像numpy array
# centerlinePts 是世界坐标，xyz顺序
def get_img_straight(imgArray, spacing, origin,centerlinePts, samplingDist, windowSize):
    # imgArray = sitk.GetArrayFromImage(img)    # zyx

    # spacing = img.GetSpacing()
    # # print("spacing is : ", spacing)
    # origin = img.GetOrigin()
    # todo:计算中心线点的方向向量
    # centerlinePts = np.unique(centerlinePts, axis=0)  # 避免出现方向向量为(0,0,0)的情况

    # ********cenPtsAndDirection**********
    cenPtsAndDirection = np.empty(shape=(0, 6))
    ind = []
    for i in range(centerlinePts.shape[0] - 1):
        v = centerlinePts[i + 1, :] - centerlinePts[i, :]
        if np.linalg.norm(v, ord=2) == 0:
            ind.append(i)

        cenPtsAndDirection = np.vstack(
        [cenPtsAndDirection, np.hstack([centerlinePts[i, :], v / np.linalg.norm(v, ord=2)])])  # 点坐标方向
    # ***********最后一个点没有方向**************
    if (ind):
        cenPtsAndDirection = np.delete(cenPtsAndDirection, ind, axis=0)
    wd = np.int((windowSize-1)/2)
    samplingDistance = samplingDist     # 考虑到使裁剪得到的窗口的世界坐标大小一致，主动脉直徿0mm上下，窗口大小定位两倍直径，叿.5mm

    # todo:构建从中心线点取周围体素所需偏移量矩阵（单位：mm）
    # *************mx,my***************************
    mx = np.zeros((windowSize, windowSize))  # 提取截面各点坐标用的 偏移矩阵
    for i in range(-wd, wd + 1, 1):
        t = np.zeros(shape=[1, windowSize]) + i
        mx[i + wd, :] = t

    my = mx

    # print('samplingDistance / spacing[0] is ', samplingDistance / spacing[0])
    mx = (mx.transpose()).reshape(windowSize * windowSize) * samplingDistance
    my = my.reshape(windowSize * windowSize) * samplingDistance

    # todo:要取的体素的世界坐标
    # ***************positions：xyz*****************
    num_reslice_img = cenPtsAndDirection.shape[0]
    positions = np.empty(shape=[0, 3])
    for i in range(0, num_reslice_img):
        a = cenPtsAndDirection[i, 0:3]
        # a = cenPtsAndDirection[i, 0:3] - cenPtsAndDirection[0, 0:3]

        v2_i = np.cross(a, cenPtsAndDirection[i, 3:6])   #
        v2_i = v2_i / np.linalg.norm(v2_i, ord=2)
        v1_i = np.cross(v2_i, cenPtsAndDirection[i, 3:6])
        v1_i = v1_i / np.linalg.norm(v1_i, ord=2)
        # 在原中心线点基础上，分别根据v1，v2所指方向按固定步长取坐标点
        px = cenPtsAndDirection[i, 0] + v1_i[0] * mx + v2_i[0] * my
        py = cenPtsAndDirection[i, 1] + v1_i[1] * mx + v2_i[1] * my
        pz = cenPtsAndDirection[i, 2] + v1_i[2] * mx + v2_i[2] * my

        p2 = (np.vstack([px, py, pz])).transpose()  # reslice的平面的所有点在CT世界坐标中的位置
        positions = np.vstack([positions, p2])

    # todo:世界坐标->图像坐标 & 变为zyx排列
    # **************index_interp：xyz***********************
    index_interp = WC2Index(positions, spacing, origin)  # reslice的平面的所有点在CT索引坐标中的位置
    # cenpindx = WC2Index(cenPtsAndDirection[0:1, 0:3], spacing, origin)    #  todo: mitk2matlab转换后的坐标indx对应不上
    # print('cenpindx: ', cenPtsAndDirection[0:1, 0:3], cenpindx)
    # 注意：python中读取的nii image的维度是zyx
    index_interp = index_interp[:, [2, 1, 0]]   # zyx

    x = np.max(index_interp[:, 0])
    y = np.max(index_interp[:, 1])
    z = np.max(index_interp[:, 2])
    if x > (imgArray.shape[0]-1) or y > (imgArray.shape[1]-1) or z > (imgArray.shape[2]-1):
        # 图像padding
        # imgArray = np.pad(imgArray, ((wd, wd), (wd, wd), (wd, wd)), 'edge')
        imgArray = np.pad(imgArray, ((150, 150), (wd, wd), (wd, wd)), 'edge')
        index_interp[:, 0] = index_interp[:, 0] + 150
        index_interp[:, 1] = index_interp[:, 1] + wd
        index_interp[:, 2] = index_interp[:, 2] + wd

    z_indx = np.array((list(range(imgArray.shape[0]))))
    y_indx = np.array((list(range(imgArray.shape[1]))))
    x_indx = np.array((list(range(imgArray.shape[2]))))
    # RegularGridInterpolator(points, values, method='linear', bounds_error=True, fill_value=nan)[source]
    # 在imgArray上定义三维网格并插值，这个网格可以被后续索引
    my_interp_func = RegularGridInterpolator((z_indx, y_indx, x_indx),
                                         imgArray)
    val = my_interp_func(index_interp)

    img_straight = val.reshape((num_reslice_img, windowSize, windowSize))

    return img_straight


def MPR(CTAimg,mask):
    vessel2ctl(mask,mode='mpr')  ####生成中心线txt
    centerline_path = './txt/' + 'vesselctl.txt'
    # CTASavePath = './MPR/' + "mprvolume.nii.gz"
    if not os.path.exists('./MPR/'):
        os.makedirs('./MPR/')
    print('process : ', "vesselctl.nii")
    spacing = CTAimg.GetSpacing()
    origin = CTAimg.GetOrigin()
    CTAimgarray = sitk.GetArrayFromImage(CTAimg)
    
    # samplingDistance = spacing[0]
    samplingDistance = 0.3
    print("samplingDistance: ", samplingDistance)
    print("windowsize: ", windowsize)
    
    centerlinePts, meandistance = get_centerline_interp(centerline_path, slicenum_min, slicenum_max)
    os.remove(centerline_path)
    print("meandistance: ", meandistance)

    # newspacing = (meandistance, meandistance, meandistance)
    newspacing = (samplingDistance, samplingDistance, samplingDistance)
    neworigin = (0, 0, 0)

    CTA_straight = get_img_straight(CTAimgarray, spacing, origin, centerlinePts, samplingDistance, windowsize)
    out1 = sitk.GetImageFromArray(CTA_straight[:, :-1, :-1])#
    out1.SetSpacing(newspacing)  # samplingDistance
    out1.SetOrigin(neworigin)
    # sitk.WriteImage(out1, CTASavePath)
    print('MPR process done')
    return out1