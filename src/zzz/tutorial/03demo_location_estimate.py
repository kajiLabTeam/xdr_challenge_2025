#! /usr/bin/env -S python3 -O
#! /usr/bin/env -S python3

import sys
import lzma
import requests
import time
from parse import parse
import numpy as np
from scipy.spatial.transform import Rotation as R

from evaalapi import statefmt, estfmt

server = "http://127.0.0.1:5000/evaalapi/"
trialname = "onlinedemo"


def spherical_to_cartesian(distance, azimuth_deg, elevation_deg):
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)
    
    x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    z = distance * np.sin(elevation_rad)
    
    return x, y, z


class DemoLocalizer:
    def __init__(self):
        self.acce_data = []
        self.gyro_data = []
        self.magn_data = []
        self.ahrs_data = []
        self.uwbp_data = []
        self.uwbt_data = []
        self.gpos_data = []
        self.viso_data = []
        self.last_est = (0, 0, 0)
        self.last_orientation = None
        self.estimation_log = []  # NLOS情報を含む推定ログ
        
    def __str__(self):
        str_data = "Stored data \n"
        str_data +=  f"acce: {self.acce_data} \n"
        str_data +=  f"gyro: {self.gyro_data} \n"
        str_data +=  f"magn: {self.magn_data} \n"
        str_data +=  f"ahrs: {self.ahrs_data} \n"
        str_data +=  f"uwbp: {self.uwbp_data} \n"
        str_data +=  f"uwbt: {self.uwbt_data} \n"
        str_data +=  f"gpos: {self.gpos_data} \n"
        str_data +=  f"viso: {self.viso_data} \n"
        return str_data
    
    def callback_acce(self, data):
        self.acce_data.append(data)

    def callback_gyro(self, data):
        self.gyro_data.append(data)
        
    def callback_magn(self, data):
        self.magn_data.append(data)
        
    def callback_ahrs(self, data):
        self.ahrs_data.append(data)
        
    def callback_uwbp(self, data):
        self.uwbp_data.append(data)
        
    def callback_uwbt(self, data):
        self.uwbt_data.append(data)
        
    def callback_gpos(self, data):
        self.gpos_data.append(data)
        
    def callback_viso(self, data):
        self.viso_data.append(data)
        
    def callback(self, sensor_type, data):
        if sensor_type == "ACCE":
            self.callback_acce(data)
        if sensor_type == "GYRO":
            self.callback_gyro(data)
        if sensor_type == "MAGN":
            self.callback_magn(data)
        if sensor_type == "AHRS":
            self.callback_ahrs(data)
        if sensor_type == "UWBP":
            self.callback_uwbp(data)
        if sensor_type == "UWBT":
            self.callback_uwbt(data)
        if sensor_type == "GPOS":
            self.callback_gpos(data)
        if sensor_type == "VISO":
            self.callback_viso(data)

    def get_latest_tag_pose(self, tag_id):
        latest_gpos = None
        for d in reversed(self.gpos_data):  # 最新のものを見つけるため逆順
            if d["object_id"] == tag_id:
                latest_gpos = d
                break

        if latest_gpos is None:
            return None, None

        loc = np.array([
            latest_gpos["location_x"],
            latest_gpos["location_y"],
            latest_gpos["location_z"]
        ])

        quat_xyz = np.array([
            latest_gpos["quat_x"],
            latest_gpos["quat_y"],
            latest_gpos["quat_z"]
        ])
        quat_w = np.sqrt(max(0.0, 1.0 - np.sum(quat_xyz**2)))
        quat = np.concatenate([quat_xyz, [quat_w]])
        quat = quat / np.linalg.norm(quat)

        return loc, quat

    def estimate_orientation_from_ahrs(self):
        """AHRS（慣性センサー）データから向きを推定"""
        if not self.ahrs_data:
            return self.last_orientation
            
        # 最新のAHRSデータを取得
        latest_ahrs = self.ahrs_data[-1]
        
        # AHRSのyaw角度を直接使用（度単位で記録されていると仮定）
        if 'yaw_z' in latest_ahrs:
            yaw_degrees = latest_ahrs['yaw_z']
            
            # 角度を-180~180度の範囲に正規化
            while yaw_degrees > 180:
                yaw_degrees -= 360
            while yaw_degrees < -180:
                yaw_degrees += 360
                
            self.last_orientation = yaw_degrees
            return self.last_orientation
        
        # クォータニオンからyawを計算（quat_2, quat_3, quat_4がx,y,z成分）
        if all(key in latest_ahrs for key in ['quat_2', 'quat_3', 'quat_4']):
            qx = latest_ahrs['quat_2']
            qy = latest_ahrs['quat_3'] 
            qz = latest_ahrs['quat_4']
            qw = np.sqrt(max(0.0, 1.0 - (qx*qx + qy*qy + qz*qz)))
            
            # クォータニオンからyaw角を計算
            yaw_rad = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            yaw_degrees = np.degrees(yaw_rad)
            
            # 角度を-180~180度の範囲に正規化
            while yaw_degrees > 180:
                yaw_degrees -= 360
            while yaw_degrees < -180:
                yaw_degrees += 360
                
            self.last_orientation = yaw_degrees
            return self.last_orientation
            
        return self.last_orientation

    def estimate_orientation_from_uwb(self, uwbt_data=None, uwbp_data=None):
        """UWBデータから向きを推定（補助的な用途）"""
        # 注意: UWBから得られる方向は「タグからデバイスへの方向」であり、
        # デバイスの向き（yaw）とは異なる。主にAHRSを使用し、UWBは補助的に使用。
        
        if uwbt_data is not None:
            # UWBT角度データから相対方向を計算（参考値として）
            azimuth = np.radians(uwbt_data["aoa_azimuth"])
            elevation = np.radians(uwbt_data["aoa_elevation"])
            
            # タグからデバイスへの方向ベクトル（これはデバイスの向きではない）
            direction_to_device = np.array([
                np.cos(elevation) * np.sin(azimuth),
                np.cos(elevation) * np.cos(azimuth),
                np.sin(elevation)
            ])
            
            # デバイスの向きではなく、タグからの相対方向として記録
            relative_bearing = np.degrees(np.arctan2(direction_to_device[0], direction_to_device[1]))
            
            # デバイス向きの推定は行わず、相対方向のみ返す
            return relative_bearing
            
        if uwbp_data is not None:
            # UWBP方向ベクトルから相対方向を計算
            direction = np.array([
                uwbp_data["direction_vec_x"],
                uwbp_data["direction_vec_y"],
                uwbp_data["direction_vec_z"]
            ])
            
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                direction = direction / direction_norm
                relative_bearing = np.degrees(np.arctan2(direction[0], direction[1]))
                return relative_bearing
        
        return None

    def estimate_location_per_tag(self, tag_id):
        """
        指定されたタグからのデータのみを使用した位置推定
        """
        tag_estimates = []
        
        # 指定されたタグからのUWBT推定
        for uwbt_data in self.uwbt_data[-10:]:
            if uwbt_data["tag_id"] != tag_id:
                continue
                
            # NLOS判定
            if uwbt_data.get("nlos", 0.0) > 0.5:
                print(f"UWBT - Tag {tag_id}: NLOS detected, skipping")
                continue
                
            tag_loc, tag_q = self.get_latest_tag_pose(tag_id)
            
            if tag_loc is not None and tag_q is not None:
                # 球面座標 → ローカルデカルト座標
                distance = uwbt_data["distance"]
                azimuth = np.radians(uwbt_data["aoa_azimuth"])
                elevation = np.radians(uwbt_data["aoa_elevation"])

                x = distance * np.cos(elevation) * np.sin(azimuth)
                y = distance * np.cos(elevation) * np.cos(azimuth)
                z = distance * np.sin(elevation)
                local_point = np.array([x, y, z])

                # ローカル座標をタグの姿勢で回転し、タグ位置に平行移動
                R_tag = R.from_quat(tag_q)
                uwbt_est = R_tag.apply(local_point) + tag_loc
                
                # 信頼度を距離とNLOS値から計算
                confidence = 1.0 / (1.0 + distance * 0.1)
                confidence *= (1.0 - uwbt_data.get("nlos", 0.0))
                
                tag_estimates.append({
                    'estimate': uwbt_est,
                    'confidence': confidence,
                    'type': 'UWBT',
                    'distance': distance
                })
        
        # 指定されたタグからのUWBP推定
        for uwbp_data in self.uwbp_data[-10:]:
            if uwbp_data["tag_id"] != tag_id:
                continue
                
            tag_loc, tag_q = self.get_latest_tag_pose(tag_id)
            
            if tag_loc is not None and tag_q is not None:
                distance = uwbp_data["distance"]
                direction_vec = np.array([
                    uwbp_data["direction_vec_x"],
                    uwbp_data["direction_vec_y"],
                    uwbp_data["direction_vec_z"]
                ])
                
                direction_vec_norm = np.linalg.norm(direction_vec)
                if direction_vec_norm > 0:
                    direction_vec = direction_vec / direction_vec_norm
                
                local_point = distance * direction_vec
                R_tag = R.from_quat(tag_q)
                uwbp_est = R_tag.apply(local_point) + tag_loc
                
                confidence = 1.0 / (1.0 + distance * 0.1)
                
                tag_estimates.append({
                    'estimate': uwbp_est,
                    'confidence': confidence,
                    'type': 'UWBP',
                    'distance': distance
                })
        
        # 重み付き平均による位置推定
        if tag_estimates:
            estimates = np.array([est['estimate'] for est in tag_estimates])
            weights = np.array([est['confidence'] for est in tag_estimates])
            total_weight = np.sum(weights)
            
            if total_weight > 0:
                final_estimate = np.sum(estimates * weights.reshape(-1, 1), axis=0) / total_weight
                return final_estimate, len(tag_estimates), total_weight
        
        return None, 0, 0

    def estimate_location_all_tags(self):
        """
        全タグからの個別推定を統合した位置推定
        """
        # 使用可能なタグIDを取得
        all_tag_ids = set()
        for uwbt_data in self.uwbt_data[-10:]:
            all_tag_ids.add(uwbt_data["tag_id"])
        for uwbp_data in self.uwbp_data[-10:]:
            all_tag_ids.add(uwbp_data["tag_id"])
        
        tag_results = {}
        valid_estimates = []
        
        # 各タグごとに位置推定
        for tag_id in all_tag_ids:
            estimate, count, confidence = self.estimate_location_per_tag(tag_id)
            if estimate is not None:
                tag_results[tag_id] = {
                    'estimate': estimate,
                    'count': count,
                    'confidence': confidence
                }
                valid_estimates.append({
                    'estimate': estimate,
                    'weight': confidence,
                    'tag_id': tag_id
                })
                print(f"Tag {tag_id}: estimate=({estimate[0]:.3f}, {estimate[1]:.3f}, {estimate[2]:.3f}), count={count}, confidence={confidence:.3f}")
        
        # 全タグの推定を統合
        if valid_estimates:
            estimates = np.array([est['estimate'] for est in valid_estimates])
            weights = np.array([est['weight'] for est in valid_estimates])
            total_weight = np.sum(weights)
            
            if total_weight > 0:
                final_estimate = np.sum(estimates * weights.reshape(-1, 1), axis=0) / total_weight
                
                print(f"=== Per-Tag Position Estimation Results ===")
                print(f"Used {len(valid_estimates)} tags")
                print(f"Final estimate (x, y, z): ({final_estimate[0]:.3f}, {final_estimate[1]:.3f}, {final_estimate[2]:.3f})")
                print(f"Total confidence weight: {total_weight:.3f}")
                print("==========================================")
                
                self.last_est = tuple(final_estimate)
                return self.last_est, tag_results
        
        print("No valid tag estimates available")
        return self.last_est, {}

    def estimate_location(self):
        # 複数ロボットからのUWBデータを使用した高精度位置推定
        
        # 全てのロボットからのLOS（Non-NLOS）データを収集
        valid_uwbt_estimates = []
        valid_uwbp_estimates = []
        
        # UWBT（Time of Flight + Angle of Arrival）による推定
        for uwbt_data in self.uwbt_data[-10:]:  # 最新10個のデータを確認
            # NLOS判定（0.0はLOS、1.0はNLOS）
            if uwbt_data.get("nlos", 0.0) > 0.5:
                print(f"UWBT - Tag {uwbt_data['tag_id']}: NLOS detected, skipping")
                continue
                
            tag_id = uwbt_data["tag_id"]
            tag_loc, tag_q = self.get_latest_tag_pose(tag_id)

            if tag_loc is not None and tag_q is not None:
                # 球面座標 → ローカルデカルト座標
                distance = uwbt_data["distance"]
                azimuth = np.radians(uwbt_data["aoa_azimuth"])
                elevation = np.radians(uwbt_data["aoa_elevation"])

                x = distance * np.cos(elevation) * np.sin(azimuth)
                y = distance * np.cos(elevation) * np.cos(azimuth)
                z = distance * np.sin(elevation)
                local_point = np.array([x, y, z])

                # ローカル座標をタグの姿勢で回転し、タグ位置に平行移動
                R_tag = R.from_quat(tag_q)
                uwbt_est = R_tag.apply(local_point) + tag_loc
                
                # 信頼度を距離とNLOS値から計算
                confidence = 1.0 / (1.0 + distance * 0.1)  # 距離が遠いほど信頼度低下
                confidence *= (1.0 - uwbt_data.get("nlos", 0.0))  # NLOS値で重み調整
                
                valid_uwbt_estimates.append({
                    'estimate': uwbt_est,
                    'confidence': confidence,
                    'tag_id': tag_id,
                    'distance': distance
                })
                
                print(f"UWBT - Tag {tag_id}: estimate=({uwbt_est[0]:.3f}, {uwbt_est[1]:.3f}, {uwbt_est[2]:.3f}), confidence={confidence:.3f}")

        # UWBP（Phase Difference）による推定
        for uwbp_data in self.uwbp_data[-10:]:  # 最新10個のデータを確認
            tag_id = uwbp_data["tag_id"]
            tag_loc, tag_q = self.get_latest_tag_pose(tag_id)

            if tag_loc is not None and tag_q is not None:
                # 方向ベクトルと距離から相対位置を計算
                distance = uwbp_data["distance"]
                direction_vec = np.array([
                    uwbp_data["direction_vec_x"],
                    uwbp_data["direction_vec_y"],
                    uwbp_data["direction_vec_z"]
                ])
                
                # 方向ベクトルを正規化
                direction_vec_norm = np.linalg.norm(direction_vec)
                if direction_vec_norm > 0:
                    direction_vec = direction_vec / direction_vec_norm
                
                # ローカル座標での相対位置
                local_point = distance * direction_vec

                # ローカル座標をタグの姿勢で回転し、タグ位置に平行移動
                R_tag = R.from_quat(tag_q)
                uwbp_est = R_tag.apply(local_point) + tag_loc
                
                # 信頼度を距離から計算
                confidence = 1.0 / (1.0 + distance * 0.1)
                
                valid_uwbp_estimates.append({
                    'estimate': uwbp_est,
                    'confidence': confidence,
                    'tag_id': tag_id,
                    'distance': distance
                })
                
                print(f"UWBP - Tag {tag_id}: estimate=({uwbp_est[0]:.3f}, {uwbp_est[1]:.3f}, {uwbp_est[2]:.3f}), confidence={confidence:.3f}")

        # 重み付き平均による位置推定
        final_estimate = None
        
        if valid_uwbt_estimates or valid_uwbp_estimates:
            all_estimates = []
            all_weights = []
            
            # UWBT推定値を追加（重み1.2倍）
            for est_data in valid_uwbt_estimates:
                all_estimates.append(est_data['estimate'])
                all_weights.append(est_data['confidence'] * 1.2)  # UWBTを少し重視
                
            # UWBP推定値を追加
            for est_data in valid_uwbp_estimates:
                all_estimates.append(est_data['estimate'])
                all_weights.append(est_data['confidence'])
            
            if all_estimates:
                # 重み付き平均計算
                all_estimates = np.array(all_estimates)
                all_weights = np.array(all_weights)
                total_weight = np.sum(all_weights)
                
                if total_weight > 0:
                    final_estimate = np.sum(all_estimates * all_weights.reshape(-1, 1), axis=0) / total_weight
                    
                    # 推定結果の統計情報
                    estimate_std = np.std(all_estimates, axis=0)
                    avg_distance = np.mean([est['distance'] for est in valid_uwbt_estimates + valid_uwbp_estimates])
                    
                    print(f"=== Multi-Robot Position Estimation Results ===")
                    print(f"Used {len(valid_uwbt_estimates)} UWBT + {len(valid_uwbp_estimates)} UWBP estimates")
                    print(f"Final estimate (x, y, z): ({final_estimate[0]:.3f}, {final_estimate[1]:.3f}, {final_estimate[2]:.3f})")
                    print(f"Estimate std deviation: ({estimate_std[0]:.3f}, {estimate_std[1]:.3f}, {estimate_std[2]:.3f})")
                    print(f"Average distance to robots: {avg_distance:.3f}m")
                    print(f"Total confidence weight: {total_weight:.3f}")
                    if self.last_orientation is not None:
                        print(f"Estimated orientation: {self.last_orientation:.1f} degrees")
                    print("================================================")
                    
                    # 推定ログに記録（NLOS情報含む）
                    nlos_flags = []
                    for est_data in valid_uwbt_estimates:
                        nlos_flags.append(0)  # UWBTはNLOSフィルタ済み
                    for est_data in valid_uwbp_estimates:
                        nlos_flags.append(0)  # UWBPは現在NLOSフラグなし
                    
                    self.estimation_log.append({
                        'estimate': final_estimate,
                        'uwbt_count': len(valid_uwbt_estimates),
                        'uwbp_count': len(valid_uwbp_estimates),
                        'has_nlos': False,  # LOSデータのみ使用
                        'confidence': total_weight
                    })
                    
                    self.last_est = tuple(final_estimate)
                    return self.last_est
        
        # データがない場合は前回の推定値を返す
        print("No valid UWB data available (all NLOS or no data), using last estimate")
        return self.last_est



def split_lines(r):
    if False and r.headers['content-type'].startswith("application/x-xz"):
        l = lzma.decompress(r.content).decode('ascii').splitlines()
    else:
        l = r.text.splitlines()
    return l


def do_req (req, n=2):
    r = requests.get(server+trialname+req)
    print("\n==>  GET " + req + " --> " + str(r.status_code))
    l = split_lines(r)
    if len(l) <= 2*n+1:
        print(r.text + '\n')
    else:
        print('\n'.join(l[:n]
                        + ["   ... ___%d lines omitted___ ...   " % len(l)]
                        + l[-n:] + [""]))
    
    return r


def parse_data(sensor_type, data_row):
    
    # Column names for each sensor type
    columns = {
        'ACCE': ['app_timestamp', 'sensor_timestamp', 'acc_x', 'acc_y', 'acc_z', 'accuracy'],
        'GYRO': ['app_timestamp', 'sensor_timestamp', 'gyr_x', 'gyr_y', 'gyr_z', 'accuracy'],
        'MAGN': ['app_timestamp', 'sensor_timestamp', 'mag_x', 'mag_y', 'mag_z', 'accuracy'],
        'AHRS': ['app_timestamp', 'sensor_timestamp', 'pitch_x', 'roll_y', 'yaw_z', 'quat_2', 'quat_3', 'quat_4', 'accuracy'],
        'UWBP': ['app_timestamp', 'sensor_timestamp', 'tag_id', 'distance', 'direction_vec_x', 'direction_vec_y', 'direction_vec_z'],
        'UWBT': ['app_timestamp', 'sensor_timestamp', 'tag_id', 'distance', 'aoa_azimuth', 'aoa_elevation', 'nlos'],
        'GPOS': ['app_timestamp', 'sensor_timestamp', 'object_id', 'location_x', 'location_y', 'location_z', 'quat_x', 'quat_y', 'quat_z'],
        'VISO': ['app_timestamp', 'sensor_timestamp', 'location_x', 'location_y', 'location_z', 'quat_x', 'quat_y', 'quat_z']
    }
    if sensor_type not in columns: 
        return None
    
    row_dict = {}
    for i, col_name in enumerate(columns[sensor_type]):
        if i < len(data_row):  # Ensure we don't go out of bounds
            # Convert numeric values to float, except for specific ID fields
            if col_name not in ['tag_id', 'object_id']:
                try:
                    row_dict[col_name] = float(data_row[i])
                except (ValueError, TypeError):
                    row_dict[col_name] = data_row[i]
            else:
                row_dict[col_name] = data_row[i]
    return row_dict


def process_data(localizer, recv_data):
    recv_sensor_lines = split_lines(recv_data)

    for line in recv_sensor_lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Split the line by semicolon
        parts = line.strip().split(';')
        
        # Get sensor type (first part of the line)
        sensor_type = parts[0]
        data_row = parts[1:]
    
        row_dict = parse_data(sensor_type, data_row)
        if row_dict is not None:
            localizer.callback(sensor_type, row_dict)
    
    # AHRSデータから向きを推定
    orientation = localizer.estimate_orientation_from_ahrs()
    if orientation is not None:
        print(f"Estimated orientation: {orientation:.1f} degrees")
    
    est = localizer.estimate_location()
    
    # 位置と向きを含めた結果を返す
    if orientation is not None:
        return est + (orientation,)  # (x, y, z, yaw)
    else:
        return est + (0.0,)  # デフォルトのyaw値 

def demo (maxw):
    localizer = DemoLocalizer()

    ## First of all, reload
    r = do_req("/reload")

    ## Check initial state
    r = do_req("/state")
    s = parse(statefmt, r.text); print(s.named)

    ## Get first 0.5s worth of data
    time.sleep(maxw)
    r = do_req("/nextdata?horizon=0.5")
    est = process_data(localizer, r)
    print("---")
    if len(est) == 4:  # (x, y, z, yaw)
        print(f"Position: ({est[0]:.3f}, {est[1]:.3f}, {est[2]:.3f}), Yaw: {est[3]:.1f}°")
    else:
        print(est)

    ## Look at remaining time
    time.sleep(maxw)
    r = do_req("/state")
    s = parse(statefmt, r.text); print(s.named)

    ## Loop until no more data
    time.sleep(maxw)
    while True:
        r = do_req("/nextdata?position=%.1f,%.1f,%.1f" % (est[0], est[1], est[2]))
        
        if r.status_code != 200 or not r.text.strip():
            print("No more sensor data. Exiting loop.")
            break

        est = process_data(localizer, r)
        print("---")
        if len(est) == 4:  # (x, y, z, yaw)
            print(f"Position: ({est[0]:.3f}, {est[1]:.3f}, {est[2]:.3f}), Yaw: {est[3]:.1f}°")
        else:
            print(est)
        time.sleep(maxw)

    ## Get estimates
    r = do_req("/estimates", 3)
    s = parse(estfmt, r.text.splitlines()[-1]); print(s.named)

    ## Get log
    time.sleep(maxw)
    r = do_req("/log", 12)

    ## We finish here
    print("Demo stops here")



# def demo (maxw):
#     localizer = DemoLocalizer()

#     ## First of all, reload
#     r = do_req("/reload")

#     ## Check initial state
#     r = do_req("/state")
#     s = parse(statefmt, r.text); print(s.named)

#     ## Get first 0.5s worth of data
#     time.sleep(maxw)
#     r = do_req("/nextdata?horizon=0.5")
#     est = process_data(localizer, r)
#     print("---")
#     print(est)

#     ## Look at remaining time
#     time.sleep(maxw)
#     r = do_req("/state")
#     s = parse(statefmt, r.text); print(s.named)
    
#     ## Set estimates
#     time.sleep(maxw)
#     for pos in range(20):
#         r = do_req("/nextdata?position=%.1f,%.1f,%.1f" % (est[0], est[1], est[2]))
#         est = process_data(localizer, r)
#         print("---")
#         print(est)
#         time.sleep(maxw)

#     ## Get estimates
#     r = do_req("/estimates", 3)
#     s = parse(estfmt, r.text.splitlines()[-1]); print(s.named)

#     ## Get log
#     time.sleep(maxw)
#     r = do_req("/log", 12)

#     ## We finish here
#     print("Demo stops here")

################################################################

if __name__ == '__main__':
        
    if len(sys.argv) != 3:
        print("""A demo for the EvAAL API.  Usage is
                %s [trial] [server]

                if omitted, TRIAL defaults to '%s' and SERVER to %s""" %
              (sys.argv[0], trialname, server))
    else:
        trialname = sys.argv[1]
        server = sys.argv[2]
    maxw = 0.5
    demo(maxw)
    exit(0)
