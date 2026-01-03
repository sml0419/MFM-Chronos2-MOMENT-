
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta
import numpy as np
from pathlib import Path
from typing import List, Tuple



class Dataset_Custom(Dataset):
    def __init__(self, data, flag, configs, scale = True):

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len 
        self.pred_len = configs.pred_len

        if type(self.label_len) is not int: self.label_len = 0
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale

        self.target = configs.target
        self.minute_interval = configs.minute_interval

        self.data = data
        self.using_index_list = self.get_using_index()
        

        self.__read_data__()

    def __read_data__(self):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        df_raw = self.data

        '''
        df_raw.columns: ['TimeStamp', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('TimeStamp')
        if 'outlier' in cols:
            cols.remove('outlier')

        # Reconstruct df_raw with TimeStamp, cols, target, and outlier (if present)
        ordered_cols = ['TimeStamp'] + cols + [self.target]
        if 'outlier' in df_raw.columns:
             ordered_cols.append('outlier')
        
        df_raw = df_raw[ordered_cols]

        num_train = int(len(self.using_index_list) * 0.8)
        num_test = int(len(self.using_index_list) * 0.1)
        num_vali = len(self.using_index_list) - num_train - num_test

        train_border_index = self.using_index_list[:num_train]
        val_border_index = self.using_index_list[num_train:num_train + num_vali]
        test_border_index = self.using_index_list[num_train + num_vali:]

        border1s = [0, min(val_border_index) - self.seq_len - self.pred_len, 
                    min(test_border_index) - self.seq_len - self.pred_len]
        border2s = [min(val_border_index), min(test_border_index), len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            train_data_x = train_data[cols].values
            train_data_y = train_data[[self.target]].values

            self.x_scaler.fit(train_data_x)
            self.y_scaler.fit(train_data_y)

            data_x = self.x_scaler.transform(df_data[cols].values)
            data_y = self.y_scaler.transform(df_data[[self.target]].values)

            data = np.concatenate((data_x,data_y), axis = 1)

            # Update self.data with normalized values
            df_raw[cols] = data_x
            df_raw[[self.target]] = data_y
            self.data = df_raw

        else:
            data = df_data[cols + [self.target]].values

        df_stamp = df_raw[['TimeStamp']][border1:border2]
        df_stamp['TimeStamp'] = pd.to_datetime(df_stamp['TimeStamp'])

        df_stamp['month'] = df_stamp['TimeStamp'].dt.month
        df_stamp['day'] = df_stamp['TimeStamp'].dt.day
        df_stamp['weekday'] = df_stamp['TimeStamp'].dt.weekday
        df_stamp['hour'] = df_stamp['TimeStamp'].dt.hour
        df_stamp['minute'] = df_stamp['TimeStamp'].dt.minute
        df_stamp['minute'] = df_stamp['minute'].floordiv(self.minute_interval)

        data_stamp = df_stamp.drop(columns = ['TimeStamp'], axis = 1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


        if self.set_type == 0:
            self.using_index_list = train_border_index
        elif self.set_type == 1:
            self.using_index_list = val_border_index
        else:
            self.using_index_list = test_border_index

    def __getitem__(self, index):
        r_end = self.using_index_list[index] - min(self.using_index_list) + self.seq_len + self.pred_len
        r_begin = r_end - self.pred_len - self.label_len
        s_end = r_begin + self.label_len
        s_begin = s_end - self.seq_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.using_index_list)

    def get_using_index(self):
        #연속된 시간 구간이며(outlier 없고), 
        #seq_len + pred_len 만큼의 길이를 만족하는 경우의 인덱스 리스트 반환
        one = 1
        index_list = []
        for i in range((self.seq_len + self.pred_len), len(self.data)):
            temp = self.data.iloc[i - self.seq_len - self.pred_len:i]
            continuous = (max(temp.TimeStamp) - min(temp.TimeStamp)).seconds == \
                timedelta(minutes=self.minute_interval * (self.seq_len + self.pred_len - 1)).seconds

            if continuous:
                if 'outlier' in temp.columns:
                    # Drop windows containing any outlier flag
                    if temp['outlier'].sum() > 0:
                        continue
                index_list.append(i)

        return index_list

def get_dataloader(data, configs, flag, shuffle=False, drop_last = True):


    data_set = Dataset_Custom(data, flag, configs)

    y_scaler = data_set.y_scaler

    data_loader = DataLoader(
        data_set,
        batch_size=configs.batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=drop_last
    )

    if flag == 'train': return data_loader, y_scaler
    else: return data_loader

class PretrainDataset(Dataset):
    """Dataset for continual pretraining with reconstruction task

    Note: Input data is already normalized by load_manufacturing_data()
    """

    def __init__(self, data, context_length=512):
        self.context_length = context_length
        # Data is already normalized, no need to normalize again
        self.data = data

    def __len__(self):
        return max(1, len(self.data) - self.context_length + 1)

    def __getitem__(self, idx):
        # Get context window
        end_idx = min(idx + self.context_length, len(self.data))
        context = self.data[idx:end_idx]

        # 부족하면 패딩
        if len(context) < self.context_length:
            pad_length = self.context_length - len(context)
            context = np.pad(context, ((0, pad_length), (0, 0)), mode='edge')

        # Transpose to [n_channels, seq_len] format
        context = torch.FloatTensor(context.T)

        # Create input mask (all valid for now, padding handled by edge mode)
        input_mask = torch.ones(self.context_length)

        return context, input_mask


def load_manufacturing_data(data_dir, pretrain_files) -> List[np.ndarray]:
    """Load and concatenate manufacturing datasets for continual pretraining

    Only keeps variables with meaningful temporal patterns for MOMENT's
    channel-independent architecture. Excludes binary labels, constant features,
    and low-cardinality variables.

    Args:
        data_dir: Directory containing data files
        pretrain_files: List of file names to load

    Returns:
        List of preprocessed numpy arrays
    """
    # Pretrain 데이터에서 유지할 컬럼들 -> 의미있는 시계열 특성들 (Numeric only)
    STEEL_COLUMNS = [
        'Usage_kWh',
        'Lagging_Current_Reactive.Power_kVarh',
        'Lagging_Current_Power_Factor'
    ]

    KEEP_COLUMNS = {
        'ai4i2020.csv': [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ],
        'IoT.csv': [
            'Vibration (mm/s)',
            'Temperature (°C)',
            'Pressure (bar)'
        ],
        'Steel_industry.csv': STEEL_COLUMNS,
        'Steel_industry_downsampled.csv': STEEL_COLUMNS
    } # Steel_industry 는 데이터가 너무 많아 downsampled 버전 사용

    all_data = []

    for file_name in pretrain_files:
        file_path = Path(data_dir) / file_name
        print(f"Loading {file_name}...")

        df = pd.read_csv(file_path)

        # Select only the columns we want to keep
        if file_name in KEEP_COLUMNS:
            selected_cols = KEEP_COLUMNS[file_name]
            # Check if all columns exist
            missing_cols = [col for col in selected_cols if col not in df.columns]
            if missing_cols:
                print(f"  Warning: Missing columns {missing_cols}, using all numeric columns instead")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                data = df[numeric_cols].values.astype(np.float32)
            else:
                data = df[selected_cols].values.astype(np.float32)
                print(f"  Selected {len(selected_cols)} temporal features: {selected_cols}")
        else:
            # Fallback: use all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            data = df[numeric_cols].values.astype(np.float32)
            print(f"  Using all {len(numeric_cols)} numeric columns")

        # Remove any NaN or inf values
        mask = np.isfinite(data).all(axis=1)
        data = data[mask]

        # Normalize each feature to prevent numerical instability
        # Use StandardScaler per feature
        data_mean = data.mean(axis=0, keepdims=True)
        data_std = data.std(axis=0, keepdims=True)
        data_std = np.where(data_std == 0, 1.0, data_std)  # Prevent division by zero
        data = (data - data_mean) / data_std

        print(f"  Shape: {data.shape}, Features: {data.shape[1]}")
        print(f"  Data range after normalization: [{data.min():.4f}, {data.max():.4f}]")
        all_data.append(data)

    return all_data


def load_samyang_data(data_dir, samyang_file, target_column) -> Tuple[pd.DataFrame, int]:
    """Load SAMYANG dataset (최근 1년치, 이상치 제거)

    Args:
        data_dir: Directory containing data files
        samyang_file: SAMYANG CSV file name
        target_column: Name of target column

    Returns:
        df: Filtered DataFrame with TimeStamp column
        target_idx: Index of target column (in feature list, excluding TimeStamp/outlier)
    """
    file_path = Path(data_dir) / samyang_file
    print(f"Loading {samyang_file}...")

    df = pd.read_csv(file_path)
    print(f"  Original shape: {df.shape}")

    # 1. Filter: 2021-04-30 이후 데이터만 사용
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    cutoff_date = pd.Timestamp('2021-04-30')
    df = df[df['TimeStamp'] > cutoff_date].reset_index(drop=True)
    print(f"  After date filter (> 2021-04-30): {df.shape}")

    # 2. Keep outlier column for Dataset_Custom to handle
    # Dataset_Custom.get_using_index() will automatically skip sequences
    # that contain time gaps (where outliers exist)
    if 'outlier' in df.columns:
        outlier_count = (df['outlier'] == 1).sum()
        print(f"  Found {outlier_count:,} outlier samples ({outlier_count/len(df)*100:.2f}%)")
        print(f"  Note: Sequences containing outliers will be excluded by Dataset_Custom's time continuity check")

    # Get feature columns (exclude TimeStamp and outlier)
    exclude_cols = ['TimeStamp', 'outlier']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Get target index
    target_idx = feature_cols.index(target_column)

    print(f"  Final shape: {df.shape}, Features: {len(feature_cols)}")
    print(f"  Target: {target_column} (index: {target_idx})")
    print(f"  Date range: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")

    return df, target_idx
