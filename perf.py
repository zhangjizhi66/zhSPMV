import os
import subprocess
from pathlib import Path

def main():
    # 设置执行次数
    N = 10  # 可根据需要修改
    
    # 参数化的标签列表
    TAGS = ["avx256d", "avx512d", "xsimd256d", "xsimd512d", "xsimd256d_complex_test", "xsimd512d_complex_test"]  # 可根据需要扩展

    # 矩阵文件目录
    matrices_dir = Path("matrices")
    
    # 自动生成可执行文件列表
    executables = [(f"spmv_{tag}", tag) for tag in TAGS]
    
    # 创建perf目录（如果不存在）并清空其中内容
    perf_dir = Path("perf")
    perf_dir.mkdir(exist_ok=True)
    
    # 清空perf目录中的所有文件
    for file_path in perf_dir.iterdir():
        if file_path.is_file():
            file_path.unlink()
    
    # 获取matrices目录下的所有mtx文件
    mtx_files = sorted([f for f in matrices_dir.iterdir() if f.is_file() and f.suffix == '.mtx'])
    
    # 对每个矩阵和每个可执行文件运行性能测试
    for mtx_file in mtx_files:
        matrix_name = mtx_file.stem
        print(f"Processing matrix: {matrix_name}")
        
        for exe_name, tag in executables:
            print(f"  Running {exe_name}...")
            
            # 执行N次
            for _ in range(N):
                try:
                    # 兼容Python 3.6的写法
                    process = subprocess.run(
                        [f"./{exe_name}", str(mtx_file)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                        universal_newlines=True
                    )
                    
                except subprocess.CalledProcessError as e:
                    print(f"  Error running {exe_name}: {e.stderr}")
    
    # 汇总所有矩阵的最佳时间到总览文件
    for tag in TAGS:
        summary_file = perf_dir / f"{tag}.txt"
        with open(summary_file, 'w') as f:
            for mtx_file in mtx_files:
                matrix_name = mtx_file.stem
                matrix_perf_file = perf_dir / f"{matrix_name}_{tag}.txt"
                
                if matrix_perf_file.exists():
                    try:
                        with open(matrix_perf_file, 'r') as mf:
                            best_time = mf.read().strip()
                            f.write(f"{matrix_name}\t{best_time}\n")
                    except Exception as e:
                        print(f"  Error reading {matrix_perf_file}: {e}")
                else:
                    f.write(f"{matrix_name}\tN/A\n")
    
    print("Benchmarking complete!")

if __name__ == "__main__":
    main()    