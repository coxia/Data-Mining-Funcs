import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import igraph

sns.set(style='ticks', palette='Set2')
plt.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']                                # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False
sns.set_style('white')

# 创建一个空对象
g = igraph.Graph()
# 添加网络中的点
vertex = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
g.add_vertices(vertex)
# 添加网络中的边
edges = [('a', 'c'), ('a', 'e'), ('a', 'b'), ('b', 'd'), ('b', 'g'), ('c', 'e'),
         ('d', 'f'), ('d', 'g'), ('e', 'f'), ('e', 'g'), ('f', 'g')]
g.add_edges(edges)

# 给边赋予属性
g.es['l1'] = [1,2,3,4,5,6,7,8,9,10,11]
# 查看第三条边的属性
print(g.es[3]['l1'])


# -----------------------其它信息-----------------------------
# 国家名称
g.vs['label'] = ['齐', '楚', '燕', '韩', '赵', '魏', '秦']
# 国家实力
g.vs['power'] = ['较强', '较强', '中等', '中等', '中等', '较弱', '较弱']
# 国家大致相对面积
g.vs['aera'] = [50, 100, 70, 40, 60, 40, 80]
# 统计日期
g['Date'] = '公元前279年'


layout = g.layout('kk')
m = igraph.plot(g, layout=layout)
m.show()
color_map = {'较强': 'pink', '中等': 'light blue', '较弱': 'light green'}
# 边默认为黑色
edge_color = dict(zip(edges, ['black']*11))

# -----------------------设置参数-----------------------------
# 参数集合
visual_style = {}
# 点的大小
visual_style["vertex_size"] = g.vs['aera']
# 点的颜色
visual_style["vertex_color"] = [color_map[power] for power in g.vs["power"]]
# 边的粗细
visual_style['edge_width'] = [1 + 5*np.random.rand() for i in np.arange(11)]
# 图尺寸
visual_style["bbox"] = (600, 480)
# 边距离
visual_style["margin"] = 50
# 布局
visual_style["layout"] = layout
# -----------------------画图-----------------------------
m = igraph.plot(g, **visual_style)
m.show()
m.save(r'C:\Users\user\Pictures\python图\igraph\1.2.png')

# 点的度
numbers = g.degree()
# 不同国家邻国数量
neighbors = dict(zip(g.vs['label'], numbers))
print(neighbors)

# 最短路径
path = g.get_shortest_paths('c', 'd')[0]
seq = g.vs.select(path)
path = seq['name']
print('燕韩之间的最短路径: ', seq['label'])

# 标记最短路径
# path里的边映射为红色。映射时需要考虑元组中对象顺序，这里按字母从小到大排序
for i in np.arange(np.size(path)-1):
    if path[i] < path[i+1]:
        edge_color[(path[i], path[i + 1])] = 'red'
    else:
        edge_color[(path[i + 1], path[i])] = 'red'
visual_style['edge_color'] = [edge_color[edge] for edge in edges]


# 计算中介中心性
betweenness = g.betweenness()
# 保留一位小数
betweenness = [round(i, 1) for i in betweenness]
# 与国家名对应
country_betweenness = dict(zip(g.vs['label'], betweenness))
print('不同国家的中介中心性（枢纽作用）：\n', country_betweenness)


plt.figure(figsize=(10, 7))
data = pd.DataFrame({'country': g.vs['label'], 'betweenness': betweenness})
data = data.sort(['betweenness'], ascending=False)
sns.barplot(x='country', y='betweenness', data=data)
# 设置刻度字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# 设置坐标标签字体大小
plt.xlabel('', fontsize=20)
plt.ylabel('Betweenness', fontsize=20)
plt.savefig(r'C:\Users\user\Pictures\python图\igraph\1.3.png', dpi=150)
plt.show()

