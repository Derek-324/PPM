import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as font_manager

# 设置 Arial 字体
arial_font = font_manager.FontProperties(family='Arial')

# 假设你的数据存储在一个名为 'haikou-peak.xlsx' 的 Excel 文件中
df = pd.read_excel('haikou-peak.xlsx')

# 提取数据
x = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5']  # 修改 x 轴标签
y1 = df.iloc[:, 1]  # 第二列从第二行开始作为第一条曲线
y2 = df.iloc[:, 2]  # 第三列从第二行开始作为第二条曲线
y3 = df.iloc[:, 3]  # 第四列从第二行开始作为第三条曲线
y4 = df.iloc[:, 4]  # 第五列从第二行开始作为第四条曲线

# 定义颜色，使用提供的 RGB 颜色值
colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2']  # 将 RGB 转换为 HEX

# 绘制折线图
plt.figure(figsize=(12, 8))  # 增加图表大小
plt.plot(x, y1, label="Optimal Revenue", marker='o', linewidth=2, markersize=8, color=colors[0])
plt.plot(x, y2, label="Revenue with Bounded Laplace mechanism", marker='o', linewidth=2, markersize=8, color=colors[1])
plt.plot(x, y3, label="Revenue without personalization", marker='o', linewidth=2, markersize=8, color=colors[2])
plt.plot(x, y4, label="Revenue with Unbounded Laplace mechanism", marker='o', linewidth=2, markersize=8, color=colors[3])

# 设置纵坐标范围 800-4000
plt.ylim(500, 4000)

# 添加标签
plt.xlabel('Batch', fontsize=22, fontproperties=arial_font)  # 增加横坐标字体大小
plt.ylabel('Revenue', fontsize=22, fontproperties=arial_font)  # 增加纵坐标字体大小

# 设置图例字体和位置
legend = plt.legend(loc='lower left', ncol=1, frameon=False)
for text in legend.get_texts():
    text.set_fontproperties(arial_font)
    text.set_fontsize(18)  # 增加图例文字的字体大小

# 隐藏标题
plt.title('', fontproperties=arial_font)

# 增加坐标轴刻度大小
plt.tick_params(axis='both', which='major', labelsize=15)  # 设置主刻度标记的大小

# 保存图表到桌面为 PDF 格式
desktop_path = os.path.expanduser("~/Desktop")
plt.savefig(os.path.join(desktop_path, 'haikou-peak.pdf'), format='pdf', dpi=300)

# 显示图表
plt.show()
