工程文件目录说明：    
data --- 数据集存放位置  
model --- 模型代码存放位置  
result --- 模型保存位置  
utils --- 一些工具包，数据集划分、数据预处理、数据处理、评价指标等  

工程文件使用指南：  
1.安装环境依赖，安装方式：首先在工程文件目录位置打开cmd命令行；然后激活虚拟环境（没有请创建）；最后，pip install -r requirements.txt 即可完成安装  
2.使用data_split.py文件根据原始数据生成train.csv、test.csv、val.csv  
3.运行main.py 即可进行模型训练与预测  
