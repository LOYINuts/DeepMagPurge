import os
import argparse
import sys


def process_fasta(input_file: str, output_dir: str):
    """处理Progenomes数据，将其根据基因组拆分为单个文件(即，属于相同基因组的会被放在同一个文件)

    Args:
        input_file: 输入的progenome3fasta文件
        output_dir: 输出的文件夹
    Return:
        无，会在输出文件夹产生所有基因组的fasta文件
    """
    current_genome = None
    current_file = None
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, "r") as f_in:
        for line in f_in:
            if line.startswith(">"):
                # 处理头部行
                header = line[1:].strip()
                parts = header.rsplit(" ")
                prefix = parts[0]
                prefix_parts = prefix.rsplit("_")
                genome = prefix_parts[0] + prefix_parts[1]
                if genome != current_genome:
                    # 关闭当前文件
                    print(genome)
                    if current_file is not None:
                        current_file.close()
                    # 打开新文件
                    filename = f"{genome}.fasta"
                    output_path = os.path.join(output_dir,filename)
                    current_file = open(output_path, "a")
                    current_genome = genome

                # 写入当前头部行
                if current_file is not None:
                    current_file.write(line)
            else:
                # 写入序列行
                if current_file is not None:
                    current_file.write(line)

    # 关闭最后一个打开的文件
    if current_file is not None:
        current_file.close()


if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='Split FASTA by genome prefix')
    parser.add_argument('-i', '--input', required=True, help='Input FASTA file')
    parser.add_argument('-o', '--output', default='output', help='Output directory (default: ./output)')
    args = parser.parse_args()

    # 验证输入文件是否存在
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)

    # 执行处理
    process_fasta(args.input, args.output)
    print(f"Processing completed. Results saved to: {os.path.abspath(args.output)}")
