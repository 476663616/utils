import numpy as np
import SimpleITK as sitk
from PIL import Image
import math
import os
from utils.centerline import vessel2ctl




def rotate_img(img, rotation_center=None, theta_x=0,theta_y=0, theta_z=0, translation=(0,0,0), interp=sitk.sitkLinear, pixel_type=None, default_value=None):
    if not rotation_center:
        rotation_center = np.array(img.GetOrigin())+np.array(img.GetSpacing())*np.array(img.GetSize())/2
    if default_value is None:
        default_value = img.GetPixel(0,0,0)
    pixel_type = img.GetPixelIDValue()

    rigid_euler = sitk.Euler3DTransform(rotation_center, theta_x, theta_y, theta_z, translation)
    return sitk.Resample(img, img, rigid_euler, interp, default_value, pixel_type)

def update_list1(size, start, P, list1, last_point):
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                new_x, new_y, new_z = start[0] + i, start[1] + j, start[2] + k
                if new_x >= 0 and new_x < size[0] and new_y >= 0 and new_y < size[1] and new_z >= 0 and new_z < size[2]:
                    if P[new_x, new_y, new_z] == 0:
                        if [new_x, new_y, new_z] not in list1:
                            list1.append([new_x, new_y, new_z])
                            key = str(new_x) + '+' + str(new_y) + '+' + str(new_z)
                            if key in last_point.keys():
                                print('error')
                            last_point[key] = start


def cpr_process(img, path):
    y_list, p_list = [], []
    y_list.append(0)
    p_list.append(img[path[0][0], :, path[0][2]]) ###只取了y，固定了平面？那生成多视角是旋转原图和mask，然后依然取这个平面的cpr？
    for i in range(1, len(path)):
        delta_y = math.sqrt(math.pow(path[i][0] - path[i - 1][0], 2) + math.pow(path[i][2] - path[i - 1][2], 2))
        y_list.append(y_list[-1] + delta_y)
        p_list.append(img[path[i][0], :, path[i][2]])
    new_img = p_list[0][np.newaxis, :]
    for i in range(1, math.ceil(y_list[-1])):
        index = []
        for j in range(0, len(y_list)):
            if i + 1 >= y_list[j] >= i - 1:
                index.append(j)
        new_row = np.zeros((p_list[0].shape[0],))
        for j in index:
            new_row = new_row + p_list[j]
        new_row = new_row / len(index)
        new_img = np.concatenate([new_img, new_row[np.newaxis, :]])
    # print(new_img.shape)
    return new_img


def cpr(img_name,center_line_name):
    # img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))
    img = sitk.GetArrayFromImage(img_name)
    path = np.loadtxt(center_line_name,dtype=int)
    os.remove(center_line_name)
    path_b = path[0]
    path_e = path[-1]
    label = np.zeros(img.shape)
    for i in range(0, path.shape[0]):
        label[int(path[i][0]), int(path[i][1]), int(path[i][2])] = 1
    label = label.astype(np.float)
    ##插值
    for i in range(1, 6):
        path = np.concatenate([np.array([path_b[0] - i, path_b[1], path_b[2]])[np.newaxis, :], path], axis=0)
    for i in range(1, 6):
        path = np.concatenate([path, np.array([path_e[0] + i, path_e[1], path_e[2]])[np.newaxis, :]], axis=0)
    # print(path[:, 0].max() - path[:, 0].min(), path[:, 1].max() - path[:, 1].min(), path[:, 2].max() - path[:, 2].min())
    
    new_img = cpr_process(img, path)
    new_label = cpr_process(label, path)
    img_slicer = (((new_img - new_img.min()) / (new_img.max() - new_img.min())) * 255).astype(np.uint8)
    img_slicer = Image.fromarray(img_slicer)
    img_slicer = img_slicer.convert("RGB")
    img_slicer = np.array(img_slicer)
    index_label = np.where(new_label > 0.2)
    if np.max(index_label[1])-np.min(index_label[1])<img_slicer.shape[0]:
        extend = img_slicer.shape[0] - (np.max(index_label[1])-np.min(index_label[1]))
        img_slicer_crop = img_slicer[:,np.min(index_label[1])-extend//2:np.max(index_label[1])+extend//2,:]
    else:
        img_slicer_crop = img_slicer[:,np.min(index_label[1]):np.max(index_label[1]),:]
    # print(img_slicer_crop.shape)
    print('cpr is done')

    return img_slicer_crop


def get_cpr(img,mask,theta_x=0,theta_y=0,theta_z=0):
    theta_xr = theta_x/180.*np.pi
    theta_yr = theta_y/180.*np.pi
    theta_zr = theta_z/180.*np.pi
    img_new = rotate_img(img, theta_z=theta_zr, theta_y=theta_yr, theta_x=theta_xr)
    mask_new = rotate_img(mask, theta_z=theta_zr, theta_y=theta_yr, theta_x=theta_xr, interp=sitk.sitkNearestNeighbor, default_value=0)
    vessel2ctl(mask_new,mode='cpr')
    center_line = './txt/' + 'vesselctl.txt'
    return cpr(img_new,center_line)
