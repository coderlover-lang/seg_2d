import SimpleITK
import os 

def load_patient(src_dir):
    '''
        读取某文件夹内的所有dicom文件
    :param src_dir: dicom文件夹路径
    :return: dicom list
    '''
    files = os.listdir(src_dir)
    slices = []
    for s in files:
        # if is_dicom_file(src_dir + '/' + s):
        instance = src_dir + '/' + s
        slices.append(instance)
        # print(instance)
    # try:
    #     slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    # except:
    #     slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    # for s in slices:
    #     s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu_by_simpleitk(dicom_dir):
    '''
        获取某文件夹内所有dicom文件的像素值
    :param src_dir: dicom文件夹路径
    :return: image array
    '''
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(image)
    img_array[img_array == -2000] = 0
    return img_array

if __name__ == "__main__":
    arr = get_pixels_hu_by_simpleitk('/home/kathy/dataset/CHAOS/Test_Sets/MR/11/T1DUAL/DICOM_anon/InPhase') 
    print(arr.shape)