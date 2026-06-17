代码总体修改与设计参考：https://github.com/harishsg993010/tiny-NPU 

本项目的起点和目标最终是为了cheshire soc芯片进行交接；其中主要是对在实际仿真验证过程中遇到的一些问题进行讨论。因为更多精力是在移植，所以其中很多细致的代码描述并不一定清楚。

因为自定义的cheshire为48位地址位和64位数据位的axi总线，而npu在slave模式下为32位数据的axi—lite控制模式；所以首先设计为一个与npu交互的axi桥（axi_full64_to_axil32_ctrl_bridge）。
其中，Cheshire为了减小输出输入端口压力，axi总线上采用system verilog打包为两组信号，npu_cheshire_wrap模块为打包与交互的顶层模块。

其中以及原作者主要的top文件为跑llm模式的顶层；而graph模式顶层位于：https://github.com/harishsg993010/tiny-NPU/blob/main/sim/verilator/onnx_sim_top.sv
其中原作者的仿真主要采用软件仿真，即使用python和c等工具进行仿真和验证；而此处修改主要为使用在硬件上进行连接，并编写 .sv或 .v 文件进行仿真。
顶层模块 tiny_npu_top 为结合了 llm 模式与onnx模式的顶层得到；其中为防止顶层过于臃肿，所以其实graph模式的顶层文件 位于 rtl\graph\graph_compute_core_stage0.sv ；
该模块更类似于onnx_sim_top的格式。而 tiny_npu_top 更像一个把garph模式顶层寄生在llm模式顶层的模块。
