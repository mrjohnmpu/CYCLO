############################################################
# 这段代码来自于CSDN博客
# https://blog.csdn.net/qq_40292148/article/details/142145949
############################################################


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
 
 
def multi_violin_box_scatter(data_dict, colors=None):
    """
    将多组数据的数据分布（以多种形式），展示在同一个图中。
    data_dict: key是数据名称，会显示在x轴上。
    """
    if colors is None:
        colors = ["#75A478", "#D75B5B", "#D6A232", "#858585"]
    if len(colors) < len(data_dict):
        return -1
 
    # 准备数据
    data = []
    categories = []
    for key in data_dict:
        data.append(data_dict[key])
        categories.append(key)
 
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
 
    # 绘制右半边的小提琴图
    for i in range(len(data)):
        parts = ax.violinplot(data[i], positions=[i], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:  # 手动调整小提琴图形状，使其只显示右半边
            verts = pc.get_paths()[0].vertices  # 获取小提琴图的顶点
            verts[:, 0] = np.clip(verts[:, 0], i, np.inf)  # 只保留x方向大于中心位置的部分
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
 
    # 绘制箱线图（在每个类别的中间位置）
    box_positions = np.arange(len(data))
    ax.boxplot(data, positions=box_positions, widths=0.1, patch_artist=True,
               boxprops=dict(facecolor='white', color='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               flierprops=dict(marker='o', color='black', markersize=1),  # 设置异常值点的样式
               showfliers=True  # 显示异常值
               )
 
    # 绘制散点图
    for i, d in enumerate(data):
        jittered_x = np.random.normal(loc=i - 0.1, scale=0.05, size=len(d))  # 添加随机偏移，使点分散开、不要在一条竖线上
        ax.scatter(jittered_x, d, color=colors[i], alpha=0.8, s=1)
 
    # 设置x轴和y轴
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel('No. of BGCs / genome')
 
    # 显示图像
    plt.show()
 
 
if __name__ == '__main__':
    # 小提琴+箱线+散点
    data_dict = {
        'subtilis': np.random.normal(15, 5, 500),
        'cereus': np.random.normal(15, 5, 500),
        'megaterium': np.random.normal(8, 3, 500),
        'circulans': np.random.normal(4, 2, 500)
    }
 
    multi_violin_box_scatter(data_dict, ['#7FA557', '#AE5259', '#CF992C', '#7E7E7E'])
