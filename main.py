import fp_growth as fpg
import pandas as pd
import sys
import matplotlib.pyplot as plt

# 数据集预处理
data = pd.read_csv('freq-winemag-data-130k-v2.csv', index_col=0, encoding='utf-8').fillna("None")
data = data.drop(labels=['description'], axis=1)
dataset = data.values.tolist()
data_num = len(dataset)
head = ['country', 'designation', 'points', 'price', 'province', 'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'title', 'variety', 'winery']
for i in range(len(dataset)):
    for j in range(len(dataset[i])):
        dataset[i][j] = head[j] + ':' + str(dataset[i][j])

if __name__ == '__main__':

    '''
    调用find_frequent_itemsets()生成频繁项
    @:param minimum_support表示设置的最小支持度,即若支持度大于等于inimum_support,保存此频繁项,否则删除
    @:param include_support表示返回结果是否包含支持度,若include_support=True,返回结果中包含itemset和support,否则只返回itemset
    '''
    frequent_itemsets = fpg.find_frequent_itemsets(dataset, minimum_support=3000, include_support=True)
    print(type(frequent_itemsets))

    result = []
    for itemset, support in frequent_itemsets:    # 将generator结果存入list
        result.append((itemset, support))
    
    # 输出频繁模式
    with open('freq.txt', 'w+', encoding="utf-8") as f:
        for itemset, support in result:
            print('频繁模式：{} 支持度：{}'.format(str(itemset), str(support)), file=f)
            print('', file=f)

    rdict = {}
    for itemset, support in result:
        itemset = sorted(itemset)
        rdict[str(itemset)] = support
    
    # 输出关联规则
    with open('rule.txt', 'w+', encoding="utf-8") as f:
        sup_list = []
        con_list = []
        lift_list = []
        allconf_list = []
        for itemset, support in result:
            res = [[]]
            for i in itemset:
                res = res + [[i] + num for num in res]
            res.pop(0)
            for i in range(len(res)):
                if len(res[i]) == len(itemset):
                    continue
                sup = rdict[str(sorted(itemset))]
                sup_list.append(sup)
                X = sorted(res[i])
                X_sup = rdict[str(X)]
                Y = sorted(list(set(itemset) - set(X)))
                Y_sup = rdict[str(Y)]
                con = sup/X_sup
                con_list.append(con)
                lift = sup/(X_sup * Y_sup)*data_num
                lift_list.append(lift)
                allconf = sup/X_sup if X_sup>Y_sup else sup/Y_sup
                allconf_list.append(allconf)
                print('关联规则：', X, '->', Y, '(支持度={}, 置信度={}, Lift={}, Allconf={}'.format(sup, con, lift, allconf), file=f)
                print('', file=f)
    
    # 可视化
    plt.bar(range(len(sup_list)), sup_list)
    plt.xticks([])
    plt.savefig('support.png')
    plt.clf()
    plt.bar(range(len(con_list)), con_list)
    plt.xticks([])
    plt.savefig('confidence.png')
    plt.clf()
    plt.bar(range(len(lift_list)), lift_list)
    plt.xticks([])
    plt.savefig('lift.png')
    plt.clf()
    plt.bar(range(len(allconf_list)), allconf_list)
    plt.xticks([])
    plt.savefig('allconf.png')
    plt.clf()