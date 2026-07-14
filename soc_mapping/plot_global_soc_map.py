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

def process_data_global_prediction(df, model):
    raw_soc_log = model.predict(df[feature_cols])
    soc_real_percent = (10 ** raw_soc_log) - 0.01
    soc_real_percent = np.clip(soc_real_percent, 0, None)
    df['predicted_soc_percent'] = soc_real_percent
    return df[['longitude', 'latitude', 'predicted_soc_percent']]

# ==========================================
# 2. 读取与推演数据
# ==========================================
print("🚀 正在加载模型并推演全球光滩数据...")
model = joblib.load(model_path)
file_list = glob.glob(os.path.join(folder_path, "*.csv"))
df_list = []

for file in file_list:
    try:
        with open(file, 'rb') as f:
            content = f.read().replace(b'\x00', b'')
        df = pd.read_csv(io.BytesIO(content), sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
        if not df.empty:
            df_list.append(process_data_global_prediction(df, model))
    except:
        continue

df_global = pd.concat(df_list, ignore_index=True)

# ==========================================
# 3. 栅格化处理 (0.35度网格以保证可见度)
# ==========================================
grid_res = 0.35
lon_bins = np.arange(-180, 180 + grid_res, grid_res)
lat_bins = np.arange(-90, 90 + grid_res, grid_res)

soc_grid, x_edge, y_edge, _ = binned_statistic_2d(
    df_global['longitude'],
    df_global['latitude'],
    df_global['predicted_soc_percent'],
    statistic='mean',
    bins=[lon_bins, lat_bins]
)
soc_grid = soc_grid.T

# 计算分位数边界用于高对比度显示
valid_soc = soc_grid[~np.isnan(soc_grid)]
quantiles = np.linspace(0.02, 0.98, 12)
bounds = np.nanquantile(valid_soc, quantiles)
bounds = np.unique(bounds)

cmap = plt.get_cmap('turbo')
norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend='both')

# ==========================================
# 4. 制图 (加入经纬网控制)
# ==========================================
print("🗺️ 正在渲染带有经纬网的高清面状地图...")
fig = plt.figure(figsize=(22, 12), dpi=300) # 稍微加宽一点以容纳坐标标签

ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()

# 添加地理要素
ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none')
ax.add_feature(cfeature.OCEAN, facecolor='#eef7fa')
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color='#333333')

# 🌟 核心改进：添加经纬网
# draw_labels=True 会在边缘显示经纬度数值
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                  linestyle='--', alpha=0.4, color='gray', linewidth=0.5)

# 优化标签显示，避免四周都有数字显得凌乱
gl.top_labels = False    # 关闭顶部标签
gl.right_labels = False  # 关闭右侧标签
gl.xlabel_style = {'size': 12, 'color': 'gray'}
gl.ylabel_style = {'size': 12, 'color': 'gray'}

# 绘制面状彩色网格
mesh = ax.pcolormesh(lon_bins, lat_bins, soc_grid,
                     cmap=cmap,
                     norm=norm,
                     transform=ccrs.PlateCarree(),
                     zorder=3)

# 渲染离散色阶条
cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.08, aspect=50, shrink=0.6, ticks=bounds[1:-1])
cbar.ax.set_xticklabels([f"{b:.1f}" for b in bounds[1:-1]], fontsize=10)
cbar.set_label('Predicted Bare Tidal Flat Deep SOC - Decile Classification', fontsize=16, fontweight='bold')

plt.title('Global Bare Tidal Flat SOC Patterns with Graticules', fontsize=26, fontweight='bold', pad=30)

save_path = os.path.join(save_directory, 'Global_SOC_Gridded_with_Graticules.png')
plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.2)

print(f"🎉 地图生成完毕！经纬网已添加。查看路径: {save_path}")
plt.show()