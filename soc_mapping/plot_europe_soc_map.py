import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# 1. 加载数据与模型 (路径已更新为欧洲)
# 请确保你的 CSV 文件列名包含 longitude, latitude 以及 15 个特征
data_path = r'D:\PyCharm2020(64bit)\pythonProject\处理\筛选后的反演点\Europe_Standardized_Results.csv'
model_path = r'D:\PyCharm2020(64bit)\pythonProject\处理\15个指标best_lgbm_model_refined.pkl'

if not os.path.exists(data_path):
    print(f"❌ 找不到数据文件，请检查路径: {data_path}")
    exit()

df = pd.read_csv(data_path)
model = joblib.load(model_path)

# 2. 特征列定义
feature_cols = [
    'ndvi', 'BIO1_Annual_Temp', 'Salinity_psu', 'tsm_p',
    'BIO14_Precip_Dry_Month', 'Vapr_kPa', 'sst_k',
    'tidal_range_m', 'soil_moisture', 'bdod_1m',
    'clay_1m', 'LSWI', 'Night_Light',
    'BIO4_Temp_Seasonality', 'BIO6_Min_Temp_Coldest'
]

# 3. 预测 SOC
print("🚀 正在计算欧洲区域预测值...")
df['predicted_soc'] = model.predict(df[feature_cols])

# 4. 创建画布 (使用 Robinson 投影在展示欧洲时视觉效果更好，或者沿用 PlateCarree)
fig = plt.figure(figsize=(15, 12), dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())

# --- 【关键修改】设置欧洲显示范围 ---
# [西经, 东经, 南纬, 北纬]
# 这个范围覆盖了从冰岛到里海，从挪威到北非的完整欧洲区域
ax.set_extent([-25, 45, 30, 72], crs=ccrs.PlateCarree())

# 5. 添加底图图层 (加入离线安全处理)
print("🗺️ 正在覆盖欧洲地理底图...")
try:
    # 尝试加载高精度底图 (50m)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#D6EAF8', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#FDFEFE', zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#2C3E50', linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#BDC3C7', linewidth=0.3, linestyle=':', zorder=2)
except Exception as e:
    print("⚠️ 联网下载底图失败，切换为本地内置基础底图...")
    ax.coastlines(resolution='110m', color='#2C3E50', linewidth=0.7, zorder=2)

# 6. 绘制预测散点
print("🔴 正在绘制欧洲反演散点...")
# 欧洲像元数通常也非常大，建议抽样
sample_size = min(800000, len(df))
plot_df = df.sample(n=sample_size, random_state=42)

sc = ax.scatter(plot_df['longitude'], plot_df['latitude'],
                c=plot_df['predicted_soc'],
                transform=ccrs.PlateCarree(),
                cmap='Spectral_r', # 欧洲冷暖色调对比明显，Spectral_r 效果很好
                s=0.15,            # 欧洲海岸线碎，点要更小一点
                alpha=0.7,
                zorder=3)

# 7. 添加颜色条和网格线
cbar = plt.colorbar(sc, orientation='horizontal', pad=0.08, aspect=50, shrink=0.6)
cbar.set_label('Predicted SOC Content (%)', fontsize=12, fontweight='bold')

gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3)
gl.top_labels = False
gl.right_labels = False

# 修改标题
plt.title('Spatial Distribution of Intertidal SOC in Europe (100m Resolution)', fontsize=16, pad=25, fontweight='bold')

# 8. 保存结果
output_name = 'Europe_SOC_Distribution_Map.png'
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"✅ 欧洲反演分布图已保存为: {output_name}")

plt.show()