import os
import re

# 文件路径
directory = '/mnt/d/J/Desktop/language technology/course/5LN700/parse/results'

# 存储结果
results = []

# 遍历目录中的所有文件
for filename in sorted(os.listdir(directory)):
    if ('conllu' in filename) and (filename.endswith('.txt') or filename.endswith('.conllu')):
        # 提取epoch编号
        epoch_match = re.search(r'epoch[_\s](\d+)', filename, re.IGNORECASE)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            
            # 读取文件
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    # 查找LAS行
                    # 使用正则表达式匹配LAS行并提取数值
                    las_match = re.search(r'LAS\s+\|\s+(\d+\.\d+)', content)
                    if las_match:
                        las_score = float(las_match.group(1))
                        results.append((epoch, las_score))
                        print(f"文件 {filename}: Epoch {epoch}, LAS {las_score}")
            except Exception as e:
                print(f"读取 {filename} 时出错: {e}")

# 按epoch排序
results.sort(key=lambda x: x[0])

# 打印结果
if results:
    print("\n提取的LAS分数:")
    for epoch, las in results:
        print(f"Epoch {epoch}: {las:.2f}")
    
    # 保存结果到文本文件
    output_path = os.path.join(directory, 'las_results.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for epoch, las in results:
            f.write(f"Epoch {epoch}: {las:.2f}\n")
    
    print(f"\n结果已保存至: {output_path}")
else:
    print("\n未能从任何文件中提取LAS数据。")