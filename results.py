from collections import OrderedDict
import pdb
#b = OrderedDict([('bbox', OrderedDict([('AP', 0.3075), ('AP50', 0.5092), ('AP75', 0.3289), (3, {'AP': 0.192, 'AP50': 0.3842, 'AP75': 0.1695}), (7, {'AP': 0.1945, 'AP50': 0.2983, 'AP75': 0.2195}), (18, {'AP': 0.3104, 'AP50': 0.4888, 'AP75': 0.3413}), (19, {'AP': 0.2341, 'AP50': 0.3989, 'AP75': 0.241}), (41, {'AP': 0.1731, 'AP50': 0.3056, 'AP75': 0.1829}), (54, {'AP': 0.1049, 'AP50': 0.1803, 'AP75': 0.1082}), (53, {'AP': 0.0335, 'AP50': 0.0618, 'AP75': 0.0293}), (78, {'AP': 0.2184, 'AP50': 0.2911, 'AP75': 0.2815}), (76, {'AP': 0.1013, 'AP50': 0.1636, 'AP75': 0.1012}), (38, {'AP': 0.0762, 'AP50': 0.1226, 'AP75': 0.0916})]))])

#b = OrderedDict([('bbox', OrderedDict([('AP', 0.3087), ('AP50', 0.5127), ('AP75', 0.3284), (2, {'AP': 0.0965, 'AP50': 0.2274, 'AP75': 0.0656}), (3, {'AP': 0.1545, 'AP50': 0.3311, 'AP75': 0.1237}), (18, {'AP': 0.261, 'AP50': 0.4105, 'AP75': 0.2825}), (20, {'AP': 0.2446, 'AP50': 0.3916, 'AP75': 0.2618}), (53, {'AP': 0.0308, 'AP50': 0.0544, 'AP75': 0.0308}), (42, {'AP': 0.14, 'AP50': 0.3159, 'AP75': 0.1084}), (78, {'AP': 0.1729, 'AP50': 0.2456, 'AP75': 0.2421}), (59, {'AP': 0.1827, 'AP50': 0.2901, 'AP75': 0.2007}), (73, {'AP': 0.2713, 'AP50': 0.4193, 'AP75': 0.3047}), (82, {'AP': 0.1698, 'AP50': 0.2277, 'AP75': 0.2002}), (34, {'AP': 0.308, 'AP50': 0.4508, 'AP75': 0.378})]))])
b =  OrderedDict([('bbox', OrderedDict([('AP', 0.2704), ('AP50', 0.4446), ('AP75', 0.2872), (1, {'AP': 0.1299, 'AP50': 0.2013, 'AP75': 0.1489}), (2, {'AP': 0.0422, 'AP50': 0.0947, 'AP75': 0.0266}), (3, {'AP': 0.1847, 'AP50': 0.3385, 'AP75': 0.1768}), (4, {'AP': 0.0374, 'AP50': 0.0789, 'AP75': 0.0345}), (5, {'AP': 0.1181, 'AP50': 0.1662, 'AP75': 0.1405}), (6, {'AP': 0.0869, 'AP50': 0.1251, 'AP75': 0.0994}), (7, {'AP': 0.0886, 'AP50': 0.1383, 'AP75': 0.0946}), (9, {'AP': 0.0308, 'AP50': 0.0701, 'AP75': 0.0237}), (16, {'AP': 0.0167, 'AP50': 0.0271, 'AP75': 0.0171}), (17, {'AP': 0.2466, 'AP50': 0.396, 'AP75': 0.2843}), (18, {'AP': 0.0925, 'AP50': 0.1377, 'AP75': 0.1006}), (19, {'AP': 0.143, 'AP50': 0.2185, 'AP75': 0.1655}), (20, {'AP': 0.0785, 'AP50': 0.1155, 'AP75': 0.088}), (21, {'AP': 0.125, 'AP50': 0.1891, 'AP75': 0.1391}), (44, {'AP': 0.0844, 'AP50': 0.1303, 'AP75': 0.0923}), (62, {'AP': 0.0336, 'AP50': 0.0733, 'AP75': 0.0274}), (63, {'AP': 0.1069, 'AP50': 0.1948, 'AP75': 0.0984}), (64, {'AP': 0.0338, 'AP50': 0.0645, 'AP75': 0.0341}), (67, {'AP': 0.0336, 'AP50': 0.0587, 'AP75': 0.0306}), (72, {'AP': 0.2318, 'AP50': 0.3421, 'AP75': 0.2454})]))])

d = b['bbox']
keys = d.keys()
sum_all = 0
for k, v in d.items():
    if k == 'AP' or k == 'AP50' or k == 'AP75':
        continue
    print( v['AP75'])
    sum_all = sum_all + v['AP75']

print('mean', sum_all/20.0)        
