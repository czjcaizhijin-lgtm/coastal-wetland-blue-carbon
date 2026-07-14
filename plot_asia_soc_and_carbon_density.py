import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# 1. 配置路径
file_path = r'D:\PyCharm2020(64bit)\pythonProject\处理\Asia_Final_Standardized.csv'
output_folder = r'D:\PyCharm2020(64bit)\pythonProject\处理'

# 2. 安全读取数据
print("📂 正在加载定稿数据...")
try:
    # 只读取必要列，节省内存
    df = pd.read_csv(file_path, usecols=['longitude', 'latitude', 'predicted_soc_perc', 'Carbon_Density'])
    print(f"✅ 成功加载 {len(df)} 个样点数据。")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit()

# 3. 绘图抽样 (658万点太多，抽样 50万点可以保证 300DPI 下依然密集成图)
if len(df) > 500000:
    print("🚀 正在进行 50万点学术抽样以优化渲染速度...")
    df_plot = df.sample(n=500000, random_state=42)
else:
    df_plot = df


# 4. 定义离线制图函数
def draw_offline_map(data, target_col, title, unit, cmap_name, save_name):
    print(f"🎨 正在绘制: {title} ...")
    plt.figure(figsize=(14, 10))

    # 使用 PlateCarree 投影
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 设置亚洲范围
    ax.set_extent([60, 150, -10, 50], crs=ccrs.PlateCarree())

    # --- 【关键修正】离线要素处理 ---
    # 使用内置的粗略海岸线 (通常无需联网)
    ax.coastlines(resolution='110m', linewidth=0.8, color='#444444', zorder=2)

    # 注意：这里去掉了 ax.add_feature(cfeature.LAND) 和 BORDERS，防止触发联网下载

    # --- 绘制散点图 ---
    sc = ax.scatter(data['longitude'], data['latitude'],
                    c=data[target_col],
                    cmap=cmap_name,
                    s=0.15,  # 点的大小，0.15 在 300DPI 下效果很好
                    alpha=0.8,  # 透明度
                    edgecolors='none',
                    transform=ccrs.PlateCarree(),
                    zorder=3)

    # 网格线 (内置功能，无需联网)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3, color='gray')
    gl.top_labels = False
    gl.right_labels = False

    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.04, extend='both')
    cbar.set_label(f'{target_col} ({unit})', fontsize=12, fontweight='bold')

    plt.title(title, fontsize=16, pad=20, fontweight='bold')

    # 保存结果
    save_path = os.path.join(output_folder, f"{save_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 地图已保存至: {save_path}")
    plt.close()


# 5. 执行出图
# 颜色条说明：
# SOC 含量推荐用 'YlOrBr' (黄色到棕色)
# 碳密度推荐用 'Spectral_r' (彩虹色反转，深蓝到深红) 或 'viridis'
draw_offline_map(df_plot, 'predicted_soc_perc',
                 'Spatial Distribution of Tidal Flat SOC Content (0-1m)',
                 '%', 'YlOrBr', 'Asia_SOC_Map_Final')

draw_offline_map(df_plot, 'Carbon_Density',
                 'Spatial Distribution of Tidal Flat Carbon Density (0-1m)',
                 'Mg C/ha', 'Spectral_r', 'Asia_Carbon_Density_Map_Final')

print("\n✨ 制图任务已完成！若需要更精细的国界，建议将 CSV 导入 ArcGIS 处理。")