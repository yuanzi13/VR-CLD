import os
import shutil
import pandas as pd
import numpy as np 
import math
from pathlib import Path
from sklearn import preprocessing
from scipy import interpolate
from feature_extraction import fixation_calculate, pupil_calculate, calculate_mean_heart_rate, calculate_blink_statistics, wavelet_denoising
import matplotlib.pyplot as plt

#未改之前的80hz频率测量的数据先插值为120hz    

def Interpolate_80hz_to_120hz(df):
    # 插值操作
    data = np.array(df)
    row_num = math.ceil(df.shape[0] * 120 / 80)
    col_num = df.shape[1]
    new_data = np.empty((row_num, col_num))

    for i in range(col_num):
        # 生成时间戳，假设每个数据点之间的时间间隔为1/80秒
        timestamps = np.arange(0, len(data[:, i])/80, 1/80)
        
        # 创建 DataFrame 临时变量
        temp_df = pd.DataFrame({'timestamp': timestamps, 'value': data[:, i]})
        
        # 前向插值
        temp_df['value'] = temp_df['value'].ffill()
        
        # 确保数据类型为浮点数
        temp_df['timestamp'] = temp_df['timestamp'].astype(float)
        temp_df['value'] = temp_df['value'].astype(float)

        # 生成120Hz的时间戳
        new_timestamps = np.arange(0, len(data[:, i])/80, 1/120)

        # 使用前向插值计算120Hz的插值数据
        new_data[:, i] = np.interp(new_timestamps, temp_df['timestamp'], temp_df['value'])

    # 将NumPy数组转换为DataFrame
    new_df = pd.DataFrame(new_data)
    new_df.columns = df.columns
    return new_df

#将眨眼数据插值成正常数据但保留眨眼标志
def Interpolate_blink(df):
    df = df.copy()
    leftEye_openness = df['leftEye_openness'].copy()
    rightEye_openness = df['rightEye_openness'].copy()

    #把瞳孔直径归-1作为眨眼标准
    df.loc[df['leftEye_pupil_dilation'] < 2, :] = np.nan
    df.loc[df['rightEye_pupil_dilation'] < 2, :] = np.nan
    

    # # 对所有列进行前向插值
    df.ffill(inplace=True)
    #对所有列进行线性插值
    #df.interpolate(method='linear', inplace=True)
    # 恢复保留列的数据
    df['leftEye_openness'] = leftEye_openness
    df['rightEye_openness'] = rightEye_openness
    
    return df

#特征提取
def feature_extraction(df, df_feature, bg_image_path, is_have_heartrate): #bg_image_path注视点的背景图路径

    velocity_threshold = np.deg2rad(100)  # 100度/秒的角速度阈值转换为弧度/秒
    clustering_threshold = 0.5 #聚类的阈值
    distance_to_screen = 1.5 #眼睛距屏幕的距离，大致估计值，不准确
    #fixations中包括该样本中的所有注视点的坐标和注视时长
    fixations_position, fixations_times, fixations = fixation_calculate(df, velocity_threshold, clustering_threshold, distance_to_screen,bg_image_path)
    #注视点个数
    fixation_num = len(fixations)
    #先对瞳孔直径信号进行滤波
    df = wavelet_denoising(df)
    #计算瞳孔平均大小，标准差，中位数，最大值，最小值，变化范围，变化率
    pupil_mean, pupil_std, pupil_median, pupil_max, pupil_min, pupil_range, pupil_change_rate = pupil_calculate(df)
    #计算眨眼次数，每次眨眼时长，总眨眼时长，眨眼频率，总眨眼占比，眨眼间隔
    total_blinks, blink_durations, total_blink_duration, blink_rate, total_blinks_ratio, blink_intervals = calculate_blink_statistics(df)

    #如果是有心率的数据集，可以计算心率特征
    if (is_have_heartrate):
        mean_heart_rate, std_heart_rate, median_heart_rate = calculate_mean_heart_rate(df)
        new_row = pd.DataFrame({'fixation_num': [fixation_num], 'fixations_position': [fixations_position], 'fixations_times': [fixations_times], 
                                'pupil_mean': [pupil_mean], 'pupil_std': [pupil_std], 'pupil_median': [pupil_median], 'pupil_max': pupil_max, 'pupil_min': [pupil_min], 'pupil_range': [pupil_range],'pupil_change_rate': [pupil_change_rate],
                                'total_blinks': [total_blinks], 'blink_durations': [blink_durations], 'total_blink_duration': [total_blink_duration], 'blink_rate': [blink_rate], 'total_blinks_ratio': [total_blinks_ratio], 'blink_intervals': [blink_intervals],
                                'mean_heart_rate': [mean_heart_rate], 'std_heart_rate': [std_heart_rate], 'median_heart_rate': [median_heart_rate]})
    else:
        new_row = pd.DataFrame({'fixation_num': [fixation_num], 'fixations_position': [fixations_position], 'fixations_times': [fixations_times], 
                                'pupil_mean': [pupil_mean], 'pupil_std': [pupil_std], 'pupil_median': [pupil_median], 'pupil_max': pupil_max, 'pupil_min': [pupil_min], 'pupil_range': [pupil_range],'pupil_change_rate': [pupil_change_rate],
                                'total_blinks': [total_blinks], 'blink_durations': [blink_durations], 'total_blink_duration': [total_blink_duration], 'blink_rate': [blink_rate], 'total_blinks_ratio': [total_blinks_ratio], 'blink_intervals': [blink_intervals]
                                })
    df_feature = pd.concat([df_feature, new_row], ignore_index=True)
    #print(df_feature.to_string(index=False))
    return df, df_feature

def save_to_folder(df, i, is_have_heartrate, label, file, index, type):
    df = df.copy()
    folder_name = ['Data', 'Feature']
    save_to_path_WH_HC = f'/public/home/seu_test3/RML/{folder_name[type]}/WH/HC/'
    save_to_path_WH_MCI = f'/public/home/seu_test3/RML/{folder_name[type]}/WH/MCI/'
    save_to_path_WOH_HC = f'/public/home/seu_test3/RML/{folder_name[type]}/WOH/HC/'
    save_to_path_WOH_MCI = f'/public/home/seu_test3/RML/{folder_name[type]}/WOH/MCI/'
    

    
    if(i==1): scene_name = 'A'
    elif(i==2): scene_name = 'B'
    elif(i==3): scene_name = 'C'
    elif(i==4): scene_name = 'D'

    if(is_have_heartrate):
        if label == 0:
            if(i==1): index[0] += 1
            folder_path = os.path.join(save_to_path_WH_HC, f'HC S{index[0]} WH/')
            # 文件夹路径不存在则自动创建
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # 保存文件
            df['SID'] = f'HC S{index[0]}'
            df.to_csv(os.path.join(folder_path, f'{scene_name}.csv'), index=False)
            print(os.path.join(folder_path, f'{scene_name}.csv'))
            file.write(os.path.join(folder_path, f'{scene_name}.csv')+'\n')
        elif label == 1:
            if(i==1):index[1] += 1
            folder_path = os.path.join(save_to_path_WH_MCI, f'MCI S{index[1]} WH/')
            # 文件夹路径不存在则自动创建
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # 保存文件
            df['SID'] = f'MCI S{index[1]}'
            df.to_csv(os.path.join(folder_path, f'{scene_name}.csv'), index=False)
            print(os.path.join(folder_path, f'{scene_name}.csv'))
            file.write(os.path.join(folder_path, f'{scene_name}.csv')+'\n')
    else:
        if label == 0:
            if(i==1):index[2] += 1
            folder_path = os.path.join(save_to_path_WOH_HC, f'HC S{index[2]} WOH/')
            # 文件夹路径不存在则自动创建
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # 保存文件
            df.to_csv(os.path.join(folder_path, f'{scene_name}.csv'), index=False)
            print(os.path.join(folder_path, f'{scene_name}.csv'))
            file.write(os.path.join(folder_path, f'{scene_name}.csv')+'\n')
        elif label == 1:
            if(i==1):index[3] += 1
            folder_path = os.path.join(save_to_path_WOH_MCI, f'MCI S{index[3]} WOH/')
            # 文件夹路径不存在则自动创建
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # 保存文件
            df.to_csv(os.path.join(folder_path, f'{scene_name}.csv'), index=False)
            print(os.path.join(folder_path, f'{scene_name}.csv'))
            file.write(os.path.join(folder_path, f'{scene_name}.csv')+'\n')


#预处理数据
def preprocess(index, path_to_folders, info_path):
    
    #默认为有心率数据
    is_have_heartrate = True

    result_txt = '/public/home/seu_test3/RML/result.txt'

    for folder_name in os.listdir(path_to_folders):
        folder_path = os.path.join(path_to_folders, folder_name)
        file = open(result_txt, 'a+')
        print(folder_name)
        file.write(f'{folder_name}\n')
        for i in range(1,5):
            #处理任务文件task
            print(f'第{i}个场景:')
            file.write(f'第{i}个场景:\n')
            csv_file_t=os.path.join(folder_path, f'cldata_{i}.csv')
            csv_file_s=os.path.join(folder_path, f'sensordata{i}.csv')
            if os.path.isfile(csv_file_t) & os.path.isfile(csv_file_s):
                df_t = pd.read_csv(csv_file_t)
                df_s = pd.read_csv(csv_file_s)
                #删除记录时间的行
                df_s = df_s[pd.notna(df_s['leftEye_openness'])]
                #标记是否有心率数据,检查 heartRate 列中是否存在至少一个非空值
                if df_s['heartRate'].notnull().any() and (df_s['heartRate'].fillna(0) != 0).any():
                    is_have_heartrate = True
                    print(f'{folder_name}的第{i}个场景有心率')
                    file.write(f'{folder_name}的第{i}个场景有心率\n')
                else:
                    is_have_heartrate = False
                    print(f'{folder_name}的第{i}个场景没有心率')
                    file.write(f'{folder_name}的第{i}个场景没有心率\n')
            
                # 对有心率的数据，进行后向插值（前面会有一段没有心率）
                if is_have_heartrate:
                    # 使用向后填充方法填补缺失值
                    df_s['heartRate'] = df_s['heartRate'].fillna(method='bfill')
                else:
                    df_s.drop(columns=['heartRate'], inplace = True)
                #删除不需要的列
                df_s.drop(columns=['heartRateVariability_Rmssd', 'heartRateVariability_Sndd'], inplace = True)
                if 'time' in df_s.columns:
                    df_s.drop(columns=['time'], inplace = True)
                if 'cognitiveLoadValue' in df_s.columns:
                    df_s.drop(columns=['cognitiveLoadValue'], inplace = True)
                if'cognitiveLoad' in df_s.columns:
                    df_s.drop(columns=['cognitiveLoad'], inplace = True)
                if'standardDeviation' and 'dataState' in df_s.columns:
                    df_s.drop(columns=['standardDeviation', 'dataState'], inplace = True)
                # 查找包含 'confidence' 的列
                columns_to_drop = df_s.filter(like='confidence').columns
                # 删除包含 'confidence' 的列
                df_s = df_s.drop(columns=columns_to_drop)

                #仅当80hz文件夹需要这句
                if '80hz' in path_to_folders:
                    df_s = Interpolate_80hz_to_120hz(df_s)

                # 对眨眼数据进行插值
                df_s = Interpolate_blink(df_s)

                # 合并原始信息
                df_s['task_acc'] = int(df_t.columns[1])/10.0
                df_s['task_time'] = df_s.shape[0]/120.0

                df_info = pd.read_csv(info_path)
                df_s['MMSE'] = df_info.loc[df_info['id'] == np.int64(folder_name), 'MMSE'].values[0]
                df_s['edu'] = df_info.loc[df_info['id'] == np.int64(folder_name), 'edu'].values[0]
                df_s['age'] = df_info.loc[df_info['id'] == np.int64(folder_name), 'age'].values[0]
                df_s['gender'] = df_info.loc[df_info['id'] == np.int64(folder_name), 'gender'].values[0]
                df_s['subjectNumber'] = df_info.loc[df_info['id'] == np.int64(folder_name), 'subjectNumber'].values[0]
                
                # 对一些整数列的插值进行四舍五入
                columns_to_round = ['leftEye_openness', 'rightEye_openness']
                df_s[columns_to_round] = df_s[columns_to_round].round(0)
                if 'heartRate' in df_s.columns:
                    df_s['heartRate'] = df_s['heartRate'].round(0) 

                # #去除头尾4s的数据，划分样本，4s->480行
                df_s = df_s.iloc[480:-480, :]

                #对HC和MCI的label分类：按照MMSE分数标准分类
                #label为0时是HC， label为1时是MCI
                label = df_info.loc[df_info['id'] == np.int64(folder_name), 'label'].values[0]
                
                #把处理好的原始数据存放到正确的文件夹中
                #type=0的时候存放数据集的内容
                save_to_folder(df_s, i, is_have_heartrate, label, file, index[0], type=0)
                
                # 按每240行划分为多个样本
                sample_time = 2 #每个样本的总时长
                sample_size = 240 #样本大小
                samples = [df_s.iloc[i:i + sample_size] for i in range(0, len(df_s), sample_size) if i + sample_size <= len(df_s)]

                # 设置打印格式为不超过4位小数
                pd.options.display.float_format = '{:.4f}'.format

                #分开处理WH和WOH的特征集
                if(is_have_heartrate):
                    #特征列表
                    df_feature_WH = []
                    #遍历samples
                    for idx, sample in enumerate(samples):
                        # print(folder_name)
                        # print(f'第{i}个场景:')
                        # print(f"Sample {idx}:\n")
                        #为每个样本添加时间戳
                        sample = sample.copy()
                        sample.loc[:, 'timestamp'] = np.linspace(0, sample_time, sample_size, endpoint=False)
                        
                        df_feature_sample = pd.DataFrame(columns=['fixation_num', 'fixations_position', 'fixations_times', 
                                                                  'pupil_mean', 'pupil_std', 'pupil_median', 'pupil_max', 'pupil_min', 'pupil_range','pupil_change_rate',
                                                                  'total_blinks', 'blink_durations', 'total_blink_duration', 'blink_rate', 'total_blinks_ratio', 'blink_intervals',
                                                                  'mean_heart_rate', 'std_heart_rate', 'median_heart_rate'
                                                                  ])
                        _, df_feature_sample = feature_extraction(sample, df_feature_sample, f'/public/home/seu_test3/RML/bgimage/bg_image_{i}.png', is_have_heartrate)
                        df_feature_WH.append(df_feature_sample)
                    
                    #print(df_feature_WH)
                    #把处理好的原始数据存放到正确的文件夹中
                    #type=1的时候存放特征集的内容
                    df_feature_WH_df = pd.concat(df_feature_WH, ignore_index=True)
                    save_to_folder(df_feature_WH_df, i,is_have_heartrate, label, file, index[1], type=1)
                else:
                    #特征列表
                    df_feature_WOH = []
                    #遍历samples
                    for idx, sample in enumerate(samples):
                        # print(folder_name)
                        # print(f'第{i}个场景:')
                        # print(f"Sample {idx}:\n")
                        #为每个样本添加时间戳
                        sample = sample.copy()
                        sample.loc[:, 'timestamp'] = np.linspace(0, sample_time, sample_size, endpoint=False)
                        
                        df_feature_sample = pd.DataFrame(columns=['fixation_num', 'fixations_position', 'fixations_times', 
                                                                  'pupil_mean', 'pupil_std', 'pupil_median', 'pupil_max', 'pupil_min', 'pupil_range','pupil_change_rate',
                                                                  'total_blinks', 'blink_durations', 'total_blink_duration', 'blink_rate', 'total_blinks_ratio', 'blink_intervals'
                                                                  ])
                        _, df_feature_sample = feature_extraction(sample, df_feature_sample, f'/public/home/seu_test3/RML/bgimage/bg_image_{i}.png', is_have_heartrate)
                        df_feature_WOH.append(df_feature_sample)
                    #print(df_feature_WOH)
                    #把处理好的原始数据存放到正确的文件夹中
                    #type=1的时候存放特征集的内容
                    df_feature_WOH_df = pd.concat(df_feature_WOH, ignore_index=True)
                    save_to_folder(df_feature_WOH_df, i,is_have_heartrate, label, file, index[1], type=1)
                
                # 还原默认的打印格式（可选）
                pd.reset_option('display.float_format')

def cccc(data, window_size, overlap):
    # 将数据转换为numpy数组
    data = data.values
    m, n = data.shape
    if m > n:
        data = data.transpose()
    X, Y = [], []
    length = (data.shape[1]-window_size)//(window_size-int(window_size*overlap)) + 1
    return length


def calculate_sample_number(path_to_folder, folder_num, strr):

    for i in range(1, folder_num+1):
        file_path_A = path_to_folder + 'S' + str(i) + strr+ '/A.csv'
        file_path_B = path_to_folder + 'S' + str(i) + strr+ '/B.csv'
        file_path_C = path_to_folder + 'S' + str(i) + strr+ '/C.csv'
        file_path_D = path_to_folder + 'S' + str(i) + strr+ '/D.csv'
        df_A = pd.read_csv(file_path_A)
        df_B = pd.read_csv(file_path_B)
        df_C = pd.read_csv(file_path_C)
        df_D = pd.read_csv(file_path_D)
        len1 = cccc(df_A, 240, 0)
        len2 = cccc(df_B, 240, 0)
        len3 = cccc(df_C, 240, 0)
        len4 = cccc(df_D, 240, 0)
  
        print(f'{file_path_A}的A样本数量为：{len1}')
        print(f'{file_path_B}的B样本数量为：{len2}')
        print(f'{file_path_C}的C样本数量为：{len3}')
        print(f'{file_path_D}的D样本数量为：{len4}')


if __name__ == "__main__":
    path_to_folders_80hz = '/public/home/seu_test3/RML/80hz'
    path_to_folders_120hz = '/public/home/seu_test3/RML/120hz'
    info_path = '/public/home/seu_test3/RML/info.csv'
    #分别代表四个文件夹里受试者当前序号
    index = [[0, 0, 0, 0], [0, 0, 0, 0]]
    preprocess(index, path_to_folders_80hz, info_path)
    preprocess(index, path_to_folders_120hz, info_path)
    # hc_path_wh = f'C:\\Users\\Administrator\\Desktop\\Data_new\\数据集\\WH\\HC\\HC '
    # hc_path_wh_num = 14
    # mci_path_wh = f'C:\\Users\\Administrator\\Desktop\\Data_new\\数据集\\WH\\MCI\\MCI '
    # mci_path_wh_num = 11
    # hc_path_woh = f'C:\\Users\\Administrator\\Desktop\\Data_new\\数据集\\WOH\\HC\\HC '
    # hc_path_woh_num = 3
    # mci_path_woh = f'C:\\Users\\Administrator\\Desktop\\Data_new\\数据集\\WOH\\MCI\\MCI '
    # mci_path_woh_num = 7
    # str1 = ' WH'
    # str2 = ' WOH'
    # calculate_sample_number(hc_path_wh, hc_path_wh_num, str1)
    # calculate_sample_number(mci_path_wh, mci_path_wh_num, str1)
    # calculate_sample_number(hc_path_woh, hc_path_woh_num, str2)
    # calculate_sample_number(mci_path_woh, mci_path_woh_num, str2)