import SimpleITK as sitk
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize_3d
import os
from utils.findVesselTernimal import findVesselTerminal
import skimage


def get_spacing_res2(x,spacing_x,origin=0):
    return float(x*spacing_x+origin)

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

def find_point_list(thin_label, txtpath, spacing, origin, start, end, mode='mpr'):
    data = thin_label.copy().astype(np.float)
    data[data < 0.005] = 0.005
    size = data.shape
    cost = 1 / data
    last_point = {}
    P = np.zeros(cost.shape)
    key = str(start[0]) + '+' + str(start[1]) + '+' + str(start[2])
    P[start[0], start[1], start[2]] = 1
    last_point[key] = [-1, -1, -1]
    list1 = []
    update_list1(size, start, P, list1, last_point)
    iter_num = 0
    while P[end[0], end[1], end[2]] == 0:
        iter_num = iter_num + 1
        if iter_num > 30000:
            print('失败')
            return
        if iter_num % 100 == 0:
            print(len(list1), len(last_point))
        cost_min = 301
        index = -1
        for i in range(0, len(list1)):
            if cost_min > cost[list1[i][0], list1[i][1], list1[i][2]]:
                cost_min = cost[list1[i][0], list1[i][1], list1[i][2]]
                index = i
        P[list1[index][0], list1[index][1], list1[index][2]] = 1
        update_list1(size, [list1[index][0], list1[index][1], list1[index][2]], P, list1, last_point)
        del list1[index]
    last = end.copy()
    path = []
    while last[0] != -1:
        path.append(np.array(last))
        last = last_point[str(last[0]) + '+' + str(last[1]) + '+' + str(last[2])]
    path_arr = np.array(path[0])[np.newaxis, :]
    for i in range(1, len(path)):
        path_arr = np.concatenate([path_arr, np.array(path[i])[np.newaxis, :]])
    ctllist = path_arr.tolist()  ####坐标顺序仍为zyx
    ctltxt = open(txtpath + 'vesselctl.txt', 'w')
    n = 0
    for pp in ctllist:
        m = 0
        for p in pp:
            if mode=='mpr':
                p = get_spacing_res2(p,spacing[::-1][m],origin[::-1][m])   #####在做cpr时需要写入像素坐标，mpr需要物理坐标
            ctltxt.write(str(p))
            m = m + 1
            if m!=len(pp):
                ctltxt.write(' ')
        n = n + 1
        if n != (len(ctllist)):
            ctltxt.write('\n')
    ctltxt.close()


def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    largestCC = largestCC.astype('int16')
    return largestCC


def vessel2ctl(mask,mode):
    txtpath = './txt/'
    vesselctl_path = './vessel/'
    if not os.path.exists(txtpath):
        os.makedirs(txtpath)
    if not os.path.exists(vesselctl_path):
        os.makedirs(vesselctl_path)
    spacing = mask.GetSpacing()
    origin = mask.GetOrigin()
    direction = mask.GetDirection()
    mask_array = sitk.GetArrayFromImage(mask)

    if mode=='cpr':
        kernel = skimage.morphology.ball(1)
        mask_array = skimage.morphology.opening(mask_array, kernel)
        
    largestCC = getLargestCC(mask_array)
    kernel = skimage.morphology.ball(1)
    largestCC = skimage.morphology.opening(largestCC, kernel)
    skeleton_array = skeletonize_3d(largestCC)
    vesselctl = sitk.GetImageFromArray(largestCC)
    vesselctl.SetDirection(direction)
    vesselctl.SetOrigin(origin)
    vesselctl.SetSpacing(spacing)
    sitk.WriteImage(vesselctl,vesselctl_path+'patient_vessel.nii')

    vessel_ctl = sitk.GetImageFromArray(skeleton_array)
    vessel_ctl.SetOrigin(origin)
    vessel_ctl.SetSpacing(spacing)
    vessel_ctl.SetDirection(direction)
    sitk.WriteImage(vessel_ctl,vesselctl_path+'vesselctl.nii')
    print('vessel thinning is Done')
    findend = findVesselTerminal()
    terminal = findend.forward(skeleton_array)
    print(terminal)
    beginindex = np.argmax(terminal[2])
    endindex = np.argmin(terminal[2])

    find_point_list(skeleton_array,txtpath,spacing,origin,start=[terminal[2][beginindex],terminal[0][beginindex],terminal[1][beginindex]],end=[terminal[2][endindex],terminal[0][endindex],terminal[1][endindex]],mode=mode)
    print('vessel centerline is Done')



