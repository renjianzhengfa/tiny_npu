# tiny_npu - Cheshire SoC / FPGA 原型验证集成分支

本仓库基于开源项目 tiny-NPU 进行修改，目标是将 tiny-NPU 的 Graph / ONNX 执行链路接入 Cheshire SoC 仿真与 FPGA 原型验证环境，用于验证 NPU 控制寄存器、Graph 工件加载、DDR DMA 数据搬运、SRAM0 中间数据访问和 block 级算子执行流程。

本仓库不是原始 tiny-NPU 的简单拷贝，而是面向 SoC 集成与 FPGA 原型验证场景做的适配分支。主要工作集中在：

* AXI 总线接口适配；
* Cheshire packed AXI 信号封装；
* AXI-Lite 控制寄存器窗口；
* Graph program / tensor descriptor 写入路径；
* DDR DMA 与 graph_compute SRAM0 的数据搬运；
* Graph 模式调度、状态观测与输出回读；
* VCS/Verdi 仿真调试与上板风险点整理。

## 1. 项目背景


原 tiny-NPU 项目主要提供 NPU 内部计算与仿真框架。为了将其接入 Cheshire SoC，需要解决以下问题：

1. Cheshire SoC 外部 AXI 总线与 tiny-NPU 控制口宽度、协议形式不一致；
2. 原 Graph 仿真流程依赖软件侧或 testbench backdoor 初始化，不适合 FPGA 原型验证；
3. Graph 模式需要独立的 program、tensor descriptor 和 DDR 数据区；
4. DMA_LOAD / DMA_STORE 需要从外部 DDR 搬运数据到 NPU 内部 SRAM0，再从 SRAM0 写回 DDR；
5. FPGA 原型验证需要可观测的寄存器、状态、PC、last_op、DMA 状态和输出回读路径。

因此，本仓库围绕 “SoC 可接入、控制可配置、数据可搬运、状态可观测、结果可回读” 进行修改。

## 2. 总体架构

系统链路如下：

```text
PC / UART-to-AXI Master
        |
        v
Cheshire SoC AXI Interconnect
        |
        +-- AXI-Lite Control Window
        |       |
        |       +-- control/status registers
        |       +-- graph program window
        |       +-- tensor descriptor window
        |
        +-- AXI Master DMA
                |
                +-- DDR read  -> DMA_LOAD  -> graph_compute SRAM0
                +-- DDR write <- DMA_STORE <- graph_compute SRAM0
```

NPU 内部 Graph 执行链路如下：

```text
Graph program
    -> graph_fetch
    -> graph_decode
    -> graph_dispatch
    -> DMA_LOAD / GEMM / EW_ADD / ReLU / MaxPool / DMA_STORE
    -> output readback / golden compare
```

## 3. 主要修改内容

### 3.1 AXI Full 64-bit 到 AXI-Lite 32-bit 控制桥

新增 `axi_full64_to_axil32_ctrl_bridge.sv`，用于适配 Cheshire SoC 侧 AXI 总线与 tiny-NPU 侧 AXI-Lite 控制口。

该桥接逻辑主要解决：

* Cheshire 侧 AXI 数据宽度与 NPU 控制口宽度不一致；
* SoC 地址空间访问需要转换为 NPU 内部寄存器读写；
* PC 端通过 UART-to-AXI Master 访问 NPU 控制寄存器时，需要稳定的 AXI-Lite 控制路径。

### 3.2 Cheshire wrapper 封装

新增 `npu_cheshire_wrap.sv`，用于连接 Cheshire packed AXI request/response 与 tiny-NPU 展开后的 AXI-Lite / AXI Master 信号。

该模块主要负责：

* 适配 Cheshire SoC 的 AXI 打包接口；
* 对接 NPU AXI-Lite slave 控制口；
* 对接 NPU AXI Master DMA 数据口；
* 降低顶层端口数量和集成复杂度。

### 3.3 tiny_npu_top 顶层整合

修改 `tiny_npu_top.sv`，将原 LLM 控制框架与 Graph / ONNX 执行链路整合到统一顶层中。

主要内容包括：

* 保留基础 control/status 寄存器；
* 增加 Graph mode 选择与启动逻辑；
* 接入 graph_top / graph_compute_core_stage0；
* 增加 graph_pc、graph_last_op、graph_status 等状态观测信号；
* 增加 Graph DMA command 与 AXI DMA read/write 之间的握手桥接。

### 3.4 Graph program 与 tensor descriptor 写入路径

为 Graph 模式增加 program 与 tensor descriptor 写入路径，使 testbench 或 UART-to-AXI Master 可以通过正式 AXI 访问方式写入 Graph 工件，而不是依赖 backdoor 初始化。

支持的典型工件包括：

* `program.bin`：Graph 指令流；
* `tdesc.bin`：tensor descriptor；
* `ddr_image.bin`：输入 feature / weight / bias 等 DDR 初始数据；
* `golden.bin`：输出结果参考数据。

该路径用于完成：

* Graph program 写入；
* tensor descriptor 写入；
* NPU 启动；
* 状态轮询；
* 输出回读；
* golden compare。

### 3.5 Graph DMA 与 SRAM0 数据搬运

Graph 模式下，DMA_LOAD / DMA_STORE 不再只停留在调度层，而是通过顶层桥接逻辑连接外部 DDR 与 graph_compute 内部 SRAM0。

数据流如下：

```text
DMA_LOAD:
DDR -> AXI DMA read -> tiny_npu_top -> graph_sram_dma_wr_* -> graph_compute SRAM0

DMA_STORE:
graph_compute SRAM0 -> graph_sram_dma_rd_* -> tiny_npu_top -> AXI DMA write -> DDR
```

由于 graph_compute 内部 SRAM0 是同步读结构，DMA_STORE 路径拆成三个阶段：

```text
READ    : 发出 SRAM0 读地址
CAPTURE : 捕获 SRAM0 返回数据
SEND    : 将数据送入 AXI DMA write 通路
```

这个修改解决了 STORE 方向上读地址、读数据和 AXI 写数据之间错拍的问题。

### 3.6 artifact_fast 与真实 DDR 路径区分

仓库中保留了 artifact_fast 仿真路径，用于快速 Graph artifact 调试；同时默认支持真实 DDR / DRAMSys 路径。

两种模式区别：

```text
artifact_fast_en = 1:
    使用内部 artifact_ddr_mem 进行快速仿真，适合早期 schedule/debug。

artifact_fast_en = 0:
    使用真实 AXI DMA read/write 访问外部 DDR，适合 Cheshire SoC / DRAMSys / FPGA 原型验证路径。
```

README 中建议以后以上板或 SoC 集成为主时，优先说明真实 DDR 路径。

## 4. 当前验证内容

当前仓库重点验证以下内容：

1. AXI-Lite 控制寄存器读写；
2. Graph program / tensor descriptor 写入；
3. NPU Graph mode 启动与状态轮询；
4. DMA_LOAD 从 DDR 到 SRAM0 的数据搬运；
5. DMA_STORE 从 SRAM0 到 DDR 的结果回写；
6. Graph block 阶段性执行流程；
7. 输出回读与 golden compare 流程；
8. VCS/Verdi 下的波形定位与问题复盘。

典型 Graph block 链路：

```text
program.bin / tdesc.bin / ddr_image.bin 写入
    -> 设置 DDR base / exec mode / start
    -> DMA_LOAD
    -> GEMM / EW_ADD / ReLU / MaxPool
    -> DMA_STORE
    -> output readback
    -> golden compare
```

## 5. 典型问题与修复记录

### 5.1 AXI 控制口宽度不匹配

问题：Cheshire 侧 AXI 数据宽度与 tiny-NPU AXI-Lite 控制口宽度不一致。

处理：增加 AXI Full 64-bit 到 AXI-Lite 32-bit 控制桥，使 SoC 侧可以稳定访问 NPU control/status 寄存器。

### 5.2 Graph 工件不能依赖 backdoor 初始化

问题：原仿真方式可以通过 testbench backdoor 直接写内部 memory，但 FPGA 上板无法使用该方式。

处理：增加 Graph program / tdesc 写入窗口，通过 AXI-Lite / UART-to-AXI Master 进行正式写入。

### 5.3 SRAM0 访问错拍

问题：Graph compute 内部 SRAM0 为同步读结构，DMA_STORE 时如果直接把读地址和写数据同拍使用，会导致 STORE 数据错拍。

处理：将 STORE 路径拆分为 READ / CAPTURE / SEND 三阶段，确保读地址、返回数据和 AXI 写数据时序对齐。

### 5.4 SRAM0 多处写入风险

问题：Graph DMA、EW/ReLU、GEMM/Pool 等路径都可能访问 SRAM0，若归属不清晰，容易出现多驱动或时序冲突。

处理：将 SRAM0 主要归属收敛到 graph_compute_core_stage0，顶层通过 graph_sram_dma_* 端口访问 SRAM0，降低多驱动风险。

### 5.5 DDR 地址映射与输出不一致

问题：Graph DMA 使用 DDR offset，实际 AXI 访问需要叠加 DDR base；若地址映射错误，容易导致 DMA_LOAD/STORE 访问错误区域，最终输出与 golden 不一致。

处理：在顶层统一处理 DDR base + graph_dma_ddr_addr 的绝对地址计算，并通过日志、波形、寄存器读写和输出回读定位问题。

## 6. 当前状态与限制

当前仓库适合作为 tiny-NPU 接入 Cheshire SoC / FPGA 原型验证的阶段性集成分支，重点展示：

* AXI 接口适配；
* Graph 控制路径；
* DDR DMA 数据搬运；
* SRAM0 访问时序；
* block 级验证流程；
* VCS/Verdi 波形调试方法。

需要继续完善的内容：

* 完整模型级 golden 对齐；
* 真实 GEMM / Pool 计算路径进一步收敛；
* SRAM0 从调试结构向更适合 FPGA 综合的 BRAM/URAM 结构整理；
* 仿真 `$display` / debug code 清理；
* Vivado 综合与 FPGA 上板约束收敛；
* README 中补充运行脚本、波形截图和验证日志。

## 7. 面向简历的项目总结

本项目可概括为：

> 基于 tiny-NPU 与 Cheshire SoC，完成 Graph 模式 NPU 在 SoC/FPGA 环境中的阶段性接入与验证，重点实现 AXI-Lite 控制窗口、Graph program/tdesc 写入、DDR DMA 数据搬运、SRAM0 读写时序修复和输出回读校验流程。使用 VCS/Verdi 定位 SRAM0 错拍、写使能与写数据不匹配、DDR 地址映射异常和 golden mismatch 等问题，形成从日志、波形、寄存器读写到回读比对的调试闭环。

```
```
