import pandas as pd
import os


def audit_hla_alleles(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    try:
        # 读取 CSV 文件
        # 考虑到生信数据可能有不同的编码，默认使用 utf-8，不行则尝试 gb2312
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gb2312')

        # 检查 'hla' 列是否存在 (有时 CSV 里可能叫 'allele' 或 'HLA')
        column_name = 'hla'
        if column_name not in df.columns:
            # 自动适配大小写或类似的名称
            potential_columns = [c for c in df.columns if c.lower() in ['hla', 'allele', 'mhc']]
            if potential_columns:
                column_name = potential_columns[0]
                print(f"警告：未找到 'hla' 列，自动识别到 '{column_name}' 列。")
            else:
                print(f"错误：文件中不存在 'hla' 或相关列。现有列名为: {list(df.columns)}")
                return

        # 获取唯一的 HLA 等位基因
        unique_alleles = df[column_name].unique()

        print("-" * 30)
        print(f"数据审计报告：{os.path.basename(file_path)}")
        print(f"总记录数: {len(df)}")
        print(f"HLA 等位基因种类总数: {len(unique_alleles)}")
        print("-" * 30)
        print("所有 HLA 等位基因种类列表：")

        # 排序打印，方便你对照 CWD 名单
        for i, allele in enumerate(sorted(unique_alleles), 1):
            print(f"{i}. {allele}")

    except Exception as e:
        print(f"运行过程中发生错误: {e}")


if __name__ == "__main__":
    # 你的文件路径
    path = r"D:\Ben_Plan\paper_factorary\LyraMHC_Project\data\processed\transpMHC_train\pmhc_test.csv"
    audit_hla_alleles(path)