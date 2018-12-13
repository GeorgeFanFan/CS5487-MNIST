import csv

#要保存后csv格式的文件名
file_name_string="digits4000_trainset.csv"
with open(file_name_string, 'w') as csvfile:

    #编码风格，默认为excel方式，也就是逗号(,)分隔
    spamwriter = csv.writer(csvfile, dialect='excel')
    # 读取txt文件，每行按逗号分割提取数据
    with open('F:\Cityu Course\Machine Learning Wednesday\Group Project\digits4000_txt\digits4000_txt\digits4000_trainset.txt', 'r') as file_txt:
        for line in file_txt:
            line_datas= line.strip('\n').split(',')
            spamwriter.writerow(line_datas)
