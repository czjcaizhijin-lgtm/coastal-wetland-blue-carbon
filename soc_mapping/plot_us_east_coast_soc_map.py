import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import glob
import io
import warnings
from scipy.stats import binned_statistic_2d

# 忽略常规警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 基础配置
# ==========================================
folder_path = r'D:\PyCharm2020(64bit)\pythonProject\处理\筛选后的反演点'
model_path = r'D:\PyCharm2020(64bit)\pythonProject\处理\15个指标best_lgbm_model_refined.pkl'
save_directory = r'D:\PyCharm2020(64bit)\pythonProject\处理'

feature_cols = ['ndvi', 'BIO1_Annual_Temp', 'Salinity_psu', 'tsm_p', 'BIO14_Precip_Dry_Month',
                'Vapr_kPa', 'sst_k', 'tidal_range_m', 'soil_moisture', 'bdod_1m',
                'clay_1m', 'LSWI', 'Night_Light', 'BIO4_Temp_Seasonality', 'BIO6_Min_Temp_Coldest']


def process_data_local_prediction(df, model):
    raw_soc_log = model.predict(df[feature_cols])
    soc_real_percent = (10 ** raw_soc_log) - 0.01
    soc_real_percent = np.clip(soc_real_percent, 0, None)
    df['predicted_soc_percent'] = soc_real_percent
    return df[['longitude', 'latitude', 'predicted_soc_percent']]


# ==========================================
# 2. 读取与推演数据 (北美东海岸空间过滤)
# ==========================================
print("🚀 正在加载模型并提取北美东海岸光滩数据...")
model = joblib.load(model_path)
file_list = glob.glob(os.path.join(folder_path, "*.csv"))
df_list = []

# 🌟 锁定美国东海岸核心区域的提取范围
lon_min, lon_max = -82.0, -68.0
lat_min, lat_max = 25.0, 45.0

for file in file_list:
    try:
        with open(file, 'rb') as f:
            content = f.read().replace(b'\x00', b'')
        df = pd.read_csv(io.BytesIO(content), sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
        if not df.empty:
            df_region = df[(df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) &
                           (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)]
            if not df_region.empty:
                df_list.append(process_data_local_prediction(df_region, model))
    except:
        continue

if not df_list:
    print("⚠️ 未在该范围内找到有效点，请检查数据。")
else:
    df_local = pd.concat(df_list, ignore_index=True)

    # ==========================================
    # 3. 栅格化处理
    # ==========================================
    grid_res = 0.05
    lon_bins = np.arange(lon_min, lon_max + grid_res, grid_res)
    lat_bins = np.arange(lat_min, lat_max + grid_res, grid_res)

    soc_grid, x_edge, y_edge, _ = binned_statistic_2d(
        df_local['longitude'],
        df_local['latitude'],
        df_local['predicted_soc_percent'],
        statistic='mean',
        bins=[lon_bins, lat_bins]
    )
    soc_grid = soc_grid.T

    # 🌟 核心：强制使用全球统一的固定十分位阈值
    valid_soc = soc_grid[~np.isnan(soc_grid)]
    if len(valid_soc) > 0:
        fixed_bounds = [19.9, 22.7, 25.2, 27.4, 29.8, 32.4, 35.4, 40.2, 42.9]
        cmap = plt.get_cmap('turbo')

        norm = mcolors.BoundaryNorm(boundaries=fixed_bounds, ncolors=cmap.N, extend='both')

        # ==========================================
        # 4. 局部制图
        # ==========================================
        print("🗺️ 正在渲染统一色标体系下的北美东海岸高清地图...")
        fig = plt.figure(figsize=(14, 12), dpi=300)

        ax = plt.axes(projection=ccrs.PlateCarree())
        # 精确锁定展示的视觉范围
        ax.set_extent([-80.0, -72.0, 30.0, 40.0], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none', zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='#eef7fa', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, color='#333333', zorder=4)

        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                          linestyle='--', alpha=0.6, color='gray', linewidth=0.5, zorder=5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 14, 'color': 'black'}
        gl.ylabel_style = {'size': 14, 'color': 'black'}

        mesh = ax.pcolormesh(lon_bins, lat_bins, soc_grid,
                             cmap=cmap,
                             norm=norm,
                             transform=ccrs.PlateCarree(),
                             zorder=3)

        cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.08, aspect=45, shrink=0.8, ticks=fixed_bounds)
        cbar.ax.set_xticklabels([f"{b:.1f}" for b in fixed_bounds], fontsize=12)
        cbar.set_label('Predicted Unvegetated Intertidal Flats Deep SOC - US East Coast', fontsize=16, fontweight='bold')

        #plt.title('US East Coast Tidal Flat SOC Patterns', fontsize=24, fontweight='bold', pad=20)

        save_path = os.path.join(save_directory, 'US_East_Coast_SOC_Zoom_Unified.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.2)

        print(f"🎉 统一色标的北美东岸大图生成完毕！查看路径: {save_path}")
        plt.show()
    else:
        print("⚠️ 栅格化后该区域无有效数据。")