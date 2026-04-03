# RBF 网格变形器（CUDA + MPI）

基于 Wendland 紧支 RBF 的 Gmsh 网格变形示例程序：在 GPU 上组装/求解 RBF 线性系统并插值节点位移；支持 **MPI 多进程**（多卡时建议一进程绑定一张 GPU），在启用 `USE_CUSOLVERMP` 的构建下可尝试 **cuSOLVERMp** 分布式稠密求解。

**实现说明（复现来源）**：本项目为复现 He 等人在 2022 年发表的论文《基于径向基函数插值的 GPU 加速 3D 网格变形方法》（2022）。

**本仓库不包含任何网格或边界控制点数据。** 运行前需自备文件，或修改 `main.cu` 中的路径。

---

## 输入文件（与 `main.cu` 默认一致）

程序当前写死的相对路径为：

| 用途 | 默认路径 |
|------|----------|
| Gmsh 网格 | `data/1pinflue10mm_gmsh.msh` |
| 边界控制点（16 个） | `data/txt/sideset4_exe_0.txt` … `data/txt/sideset4_exe_15.txt` |

在工程根目录下可自行创建目录并放入数据，例如：

```bash
mkdir -p data/txt
# 将你的 .msh 与 16 个 txt 按上述命名放入对应位置
```

若目录或文件名不同，请直接编辑 `main.cu` 中的 `msh_filepath` 与 `snprintf(..., "data/txt/sideset4_exe_%d.txt", ...)` 格式字符串。

---

## 依赖与 MPI 说明

- **NVIDIA GPU**、**CUDA**（与驱动匹配的 CUDA 用户态）
- **MPI**（Open MPI）。在已安装 **NVHPC + HPCX** 的机器上，建议使用 HPCX 自带的 `mpirun`，避免与 conda/系统另一套 MPI 混用导致 `MPI_Init` 类错误。
- **默认 `make`**：需要 NVHPC 中的 **CUDA 13.x**、`math_libs` 里的 **cuSOLVER / cuSOLVERMp / NCCL** 等（见 `Makefile`）。
- **`make cuda12`**：使用系统 **`/usr/local/cuda-12.6`**（可通过 `CUDA12_HOME` 覆盖），**不**定义 `USE_CUSOLVERMP`，走 **cuSOLVERDn** 等；适合本机 `nvidia-smi` 仅报告 **CUDA 12.x**、与 CUDA 13 用户态不兼容的情况。

---

## 项目结构（源码仓库）

```
RBF_Project/
├── Makefile
├── common.h
├── main.cu
├── MeshLoader.h
├── MeshLoader.cpp
├── RBFDeformer.h
├── RBFDeformer.cu
└── README.md
```

`data/` 不在仓库中，由使用者在本地创建（见上文）。

---

## 编译

```bash
cd RBF_Project

# 方式 A：NVHPC CUDA 13 + cuSOLVERMp（需环境与驱动支持 CUDA 13 用户态）
make -j

# 方式 B：系统 CUDA 12.x（常见于本机驱动只支持 CUDA 12 的场景）
make cuda12 -j
```

生成可执行文件 **`rbf_deformer`**（勿提交到 Git，建议本地 `.gitignore` 忽略）。

---

## 运行

在**工程根目录**执行（保证相对路径 `data/...` 有效）：

```bash
# 单进程
mpirun -np 1 ./rbf_deformer

# 多进程（按机器调整 np 与 GPU 绑定策略）
mpirun -np 2 --bind-to none ./rbf_deformer
```

使用 **`make cuda12`** 编译时，可设置：

```bash
export RBF_DISABLE_CUSOLVERMP=1
```

### 环境变量（节选）

| 变量 | 含义 |
|------|------|
| `RBF_NUM_SIDESETS` | 读取的 sideset 文件个数（默认 16） |
| `RBF_SAMPLE_N` | 控制点下采样步长（默认 100） |
| `RBF_REQUIRE_CUSOLVERMP` | 若设置，cuSOLVERMp 失败时不允许回退 |
| `RBF_DISABLE_CUSOLVERMP` | 跳过 cuSOLVERMp，走回退求解路径 |
| `CUDA_VISIBLE_DEVICES` | 限制可见 GPU |

示例：

```bash
RBF_NUM_SIDESETS=16 RBF_SAMPLE_N=100 mpirun -np 2 --bind-to none ./rbf_deformer
```

多 rank 日志交错时可用：

```bash
mpirun -np 2 --tag-output --bind-to none ./rbf_deformer
```

---

## 程序在做什么（简要）

1. 加载 Gmsh 网格与多文件边界控制点（带下采样与 MPI 分工读取）。
2. 为控制点赋测试位移（当前示例为**悬臂梁弯曲**，可在 `main.cu` 中修改）。
3. 在 GPU 上建立 RBF 系统并求解系数（`cuSOLVERDn` 或 `cuSOLVERMp`，取决于编译与环境变量）。
4. 对网格节点插值位移；多进程时对分片并行处理。
5. 由 rank 0 写出 **`bending4.csv`**（`node_id,x,y,z,dx,dy,dz`）与 **`bending4.msh`**（在原网格基础上替换 `$Nodes` 后的变形网格）。

---

## 输出文件

- **`bending4.csv`**：每行一个节点，`node_id,x,y,z,dx,dy,dz`
- **`bending4.msh`**：Gmsh 格式变形网格（与输入网格拓扑一致，仅节点坐标更新）

二者生成于**当前工作目录**（一般为工程根目录）；体积可能较大，请勿纳入版本库。
