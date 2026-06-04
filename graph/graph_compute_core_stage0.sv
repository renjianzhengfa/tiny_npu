// =============================================================================
// graph_compute_core_stage0.sv  (owned SRAM0 + DMA SRAM port + real GEMM controller)
//
// Purpose:
//   - Own the single Graph SRAM0 inside graph_compute.
//   - Keep graph_top, graph program SRAM, and tensor_table inside this module.
//   - Accept Graph DMA commands from graph_top and expose a 64-bit SRAM0 DMA port
//     to tiny_npu_top's artifact_fast / AXI-DMA adapter.
//   - Connect EW_ADD/ReLU directly to the internal SRAM0 through graph_top's
//     ew_rd/ew_wr byte port.
//   - Connect real gemm_ctrl + systolic_array for GEMM, while keeping
//     non-critical Graph engines as connector-lite SRAM0 engines.
//   - Keep Softmax/Reduce/Math/Gather/Slice/Concat/Pool/Pad/Resize/Cast as
//     fixed-latency done stubs for schedule bring-up.
//
// IMPORTANT:
//   This is still not golden-correct YOLO/Conv.  GEMM is physically connected,
//   but quantization/layout still need alignment.  Non-GEMM Graph engines below
//   are connector-lite pass-through engines, not final mathematical operators.
// =============================================================================

`default_nettype none

`ifndef SYNTHESIS
  `ifdef NPU_SIM_DEBUG
    `define NPU_DBG(args) $display args
  `else
    `define NPU_DBG(args)
  `endif
`else
  `define NPU_DBG(args)
`endif

module graph_compute_core_stage0 #(
    parameter int PROG_SRAM_AW = 9,
    parameter int SRAM0_AW     = 16,
    parameter int TDESC_AW     = 8,
    parameter int ARRAY_M      = 16,
    parameter int ARRAY_N      = 16,
    parameter int DATA_W       = 8,
    parameter int ACC_W        = 32
)(
    input  logic                     clk,
    input  logic                     rst_n,

    input  logic                     start,
    input  logic [15:0]              prog_len,
    input  logic                     scheduler_mode,

    // Graph program write port from AXI-Lite window
    input  logic                     graph_prog_we,
    input  logic [PROG_SRAM_AW-1:0]  graph_prog_waddr,
    input  logic [127:0]             graph_prog_wdata,

    // Tensor descriptor write port from AXI-Lite window
    input  logic                     tdesc_we,
    input  logic [TDESC_AW-1:0]      tdesc_waddr,
    input  logic [255:0]             tdesc_wdata,

    // DMA command interface to the outer AXI/artifact DMA adapter
    output logic                     dma_cmd_valid,
    output logic [31:0]              dma_ddr_addr,
    output logic [15:0]              dma_sram_addr,
    output logic [15:0]              dma_length,
    output logic                     dma_direction,   // 0=LOAD DDR->SRAM, 1=STORE SRAM->DDR
    output logic                     dma_strided,
    output logic [31:0]              dma_stride,
    output logic [15:0]              dma_count,
    output logic [15:0]              dma_block_len,
    input  logic                     dma_done,

    // 64-bit SRAM0 DMA access port owned by tiny_npu_top adapter.
    // LOAD path writes up to 8 bytes per cycle into internal SRAM0.
    input  logic                     dma_sram_wr_en,
    input  logic [SRAM0_AW-1:0]      dma_sram_wr_addr,
    input  logic [63:0]              dma_sram_wr_data,
    input  logic [7:0]               dma_sram_wr_mask,

    // STORE path reads up to 8 bytes per cycle from internal SRAM0.
    input  logic                     dma_sram_rd_en,
    input  logic [SRAM0_AW-1:0]      dma_sram_rd_addr,
    output logic [63:0]              dma_sram_rd_data,

    // Status/debug
    output logic                     graph_done,
    output logic                     graph_busy,
    output logic [31:0]              graph_status,
    output logic [15:0]              graph_pc,
    output logic [7:0]               graph_last_op
);

    // -------------------------------------------------------------------------
    // Graph program SRAM
    // -------------------------------------------------------------------------
    logic                         prog_rd_en;
    logic [PROG_SRAM_AW-1:0]      prog_rd_addr;
    logic [127:0]                 prog_rd_data;
    logic [127:0]                 graph_prog_mem [0:(1 << PROG_SRAM_AW)-1];

    always @(posedge clk) begin
        if (graph_prog_we) begin
            graph_prog_mem[graph_prog_waddr] <= graph_prog_wdata;
        end
        if (prog_rd_en) begin
            prog_rd_data <= graph_prog_mem[prog_rd_addr];
        end
    end

    // -------------------------------------------------------------------------
    // Tensor descriptor table
    // -------------------------------------------------------------------------
    logic [TDESC_AW-1:0] td_rd0_addr;
    logic [TDESC_AW-1:0] td_rd1_addr;
    logic [TDESC_AW-1:0] td_rd2_addr;
    logic [255:0]        td_rd0_data;
    logic [255:0]        td_rd1_data;
    logic [255:0]        td_rd2_data;

    tensor_table #(
        .NUM_ENTRIES (1 << TDESC_AW),
        .ENTRY_BITS  (256)
    ) u_tensor_table (
        .clk      (clk),
        .rst_n    (rst_n),

        .wr_en    (tdesc_we),
        .wr_addr  (tdesc_waddr),
        .wr_data  (tdesc_wdata),

        .rd0_addr (td_rd0_addr),
        .rd0_data (td_rd0_data),
        .rd1_addr (td_rd1_addr),
        .rd1_data (td_rd1_data),
        .rd2_addr (td_rd2_addr),
        .rd2_data (td_rd2_data)
    );

    localparam int SCRATCH_DEPTH = 2048;
    localparam int SCR_AW = $clog2(SCRATCH_DEPTH);

    // -------------------------------------------------------------------------
    // Owned SRAM0
    // -------------------------------------------------------------------------

////////////////////////////////////////////
// -------------------------------------------------------------------------
// Debug helper: dump SRAM0[0x5a00 : 0x5a3f]
// Only for simulation/debug.
// -------------------------------------------------------------------------
`ifdef SYNTHESIS
task automatic dump_sram0_5a00(input string tag);
    logic [63:0] word64;
    int off;
    int b;
begin
    $display("[%0t] ===== SRAM0_DUMP %s base=0x5a00 =====", $time, tag);

    for (off = 0; off < 64; off = off + 8) begin
        word64 = 64'd0;
        for (b = 0; b < 8; b = b + 1) begin
            word64[8*b +: 8] = sram0[16'h5a00 + off + b];
        end

        $display("[%0t] SRAM0_DUMP %s addr=0x%04x data=0x%016x",
                 $time, tag, 16'h5a00 + off[15:0], word64);
    end

    $display("[%0t] ===== SRAM0_DUMP_END %s =====", $time, tag);
end
endtask
`endif

/////////////////////////////////////////


    // EW/ReLU SRAM byte port from graph_top
    logic                     ew_rd_en;
    logic [SRAM0_AW-1:0]      ew_rd_addr;
    logic [7:0]               ew_rd_data;
    logic                     ew_wr_en;
    logic [SRAM0_AW-1:0]      ew_wr_addr;
    logic [7:0]               ew_wr_data;
    logic                     ew_busy;

    always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ew_rd_data <= 8'd0;
    end else if (ew_rd_en) begin
        ew_rd_data <= sram0[ew_rd_addr];
    end
end

    // DMA STORE read port, little-endian lane mapping:
    // lane 0 -> byte at rd_addr + 0
    always_comb begin
        dma_sram_rd_data = 64'd0;
        for (int j = 0; j < 8; j++) begin
            dma_sram_rd_data[8*j +: 8] = sram0[(dma_sram_rd_addr + j[SRAM0_AW-1:0])];
        end
    end

    // -------------------------------------------------------------------------
    // Non-DMA engine command ports from graph_top
    // -------------------------------------------------------------------------
    logic        gm_cmd_valid;
    logic [15:0] gm_cmd_src0, gm_cmd_src1, gm_cmd_dst;
    logic [15:0] gm_cmd_M, gm_cmd_N, gm_cmd_K;
    logic [7:0]  gm_cmd_flags;
    logic [15:0] gm_cmd_imm;
    logic [1:0]  gm_cmd_dtype;
    logic        gm_done;

    logic        sm_cmd_valid;
    logic [15:0] sm_src_base, sm_dst_base, sm_length;
    logic [1:0]  sm_cmd_dtype;
    logic        sm_done;

    logic        re_cmd_valid;
    logic [7:0]  re_cmd_opcode;
    logic [15:0] re_cmd_src_base, re_cmd_dst_base;
    logic [15:0] re_cmd_reduce_dim, re_cmd_outer_count;
    logic        re_done;

    logic        me_cmd_valid;
    logic [7:0]  me_cmd_opcode;
    logic [15:0] me_cmd_src_base, me_cmd_dst_base, me_cmd_length;
    logic [1:0]  me_cmd_dtype;
    logic        me_done;

    logic        ga_cmd_valid;
    logic [15:0] ga_cmd_src_base, ga_cmd_idx_base, ga_cmd_dst_base;
    logic [15:0] ga_cmd_num_indices, ga_cmd_row_size, ga_cmd_num_rows;
    logic        ga_done;

    logic        sl_cmd_valid;
    logic [15:0] sl_cmd_src_base, sl_cmd_dst_base;
    logic [15:0] sl_cmd_src_row_len, sl_cmd_dst_row_len, sl_cmd_start_offset, sl_cmd_num_rows;
    logic        sl_done;

    logic        ct_cmd_valid;
    logic [15:0] ct_cmd_src0_base, ct_cmd_src1_base, ct_cmd_dst_base;
    logic [15:0] ct_cmd_src0_row_len, ct_cmd_src1_row_len, ct_cmd_num_rows;
    logic        ct_done;

    logic        ap_cmd_valid;
    logic [15:0] ap_cmd_src_base, ap_cmd_dst_base;
    logic [15:0] ap_cmd_C, ap_cmd_H, ap_cmd_W;
    logic [15:0] ap_cmd_kh, ap_cmd_kw, ap_cmd_sh, ap_cmd_sw;
    logic        ap_done;

    logic        mp_cmd_valid;
    logic [15:0] mp_cmd_src_base, mp_cmd_dst_base;
    logic [15:0] mp_cmd_C, mp_cmd_H, mp_cmd_W;
    logic [15:0] mp_cmd_kh, mp_cmd_kw, mp_cmd_sh, mp_cmd_sw;
    logic        mp_done;

    logic        pd_cmd_valid;
    logic [15:0] pd_cmd_src_base, pd_cmd_dst_base;
    logic [15:0] pd_cmd_C, pd_cmd_H, pd_cmd_W;
    logic [15:0] pd_cmd_pad_top, pd_cmd_pad_bottom, pd_cmd_pad_left, pd_cmd_pad_right;
    logic        pd_done;

    logic        rz_cmd_valid;
    logic [15:0] rz_cmd_src_base, rz_cmd_dst_base;
    logic [15:0] rz_cmd_C, rz_cmd_in_H, rz_cmd_in_W, rz_cmd_out_H, rz_cmd_out_W;
    logic        rz_done;

    logic        ca_cmd_valid;
    logic [15:0] ca_cmd_src_base, ca_cmd_dst_base, ca_cmd_length;
    logic [1:0]  ca_cmd_src_dtype, ca_cmd_dst_dtype;
    logic        ca_done;

    // -------------------------------------------------------------------------
    // Real GEMM engine connection: gemm_ctrl + systolic_array
    // -------------------------------------------------------------------------
    // graph_dispatch drives gm_cmd_*; gemm_ctrl reads/writes SRAM0 and controls
    // the systolic array.  Other Graph engines are still stubbed below unless
    // they are used by the current block.

    logic                     gemm_sram_rd_en;
    logic [SRAM0_AW-1:0]      gemm_sram_rd_addr;
    logic [7:0]               gemm_sram_rd_data;
    logic                     gemm_sram_wr_en;
    logic [SRAM0_AW-1:0]      gemm_sram_wr_addr;
    logic [7:0]               gemm_sram_wr_data;

    logic                     gemm_acc_rd_en;
    logic [7:0]               gemm_acc_rd_addr;
    logic signed [ACC_W-1:0]  gemm_acc_rd_data;
    logic                     gemm_acc_wr_en;
    logic [7:0]               gemm_acc_wr_addr;
    logic signed [ACC_W-1:0]  gemm_acc_wr_data;

    logic                     sa_clear;
    logic                     sa_en;
    logic                     sa_dtype_fp16;
    logic signed [DATA_W-1:0] sa_a_col      [ARRAY_M];
    logic signed [DATA_W-1:0] sa_b_row      [ARRAY_N];
    logic signed [15:0]       sa_a_col_fp16 [ARRAY_M];
    logic signed [15:0]       sa_b_row_fp16 [ARRAY_N];
    logic signed [ACC_W-1:0]  sa_acc        [ARRAY_M][ARRAY_N];
    logic                     sa_acc_valid;
    logic                     gemm_busy;

    logic signed [ACC_W-1:0]  gemm_acc_mem [0:255];
    logic [31:0]              dma_dbg_wr_count;
    logic [31:0]              gemm_dbg_wr_count;

    always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        gemm_sram_rd_data <= 8'd0;
    end else if (gemm_sram_rd_en) begin
        gemm_sram_rd_data <= sram0[gemm_sram_rd_addr];
    end
end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gemm_acc_rd_data <= '0;
        end else if (gemm_acc_rd_en) begin
            gemm_acc_rd_data <= gemm_acc_mem[gemm_acc_rd_addr];
        end

        if (gemm_acc_wr_en) begin
            gemm_acc_mem[gemm_acc_wr_addr] <= gemm_acc_wr_data;
        end
    end

    systolic_array #(
        .M      (ARRAY_M),
        .N      (ARRAY_N),
        .DATA_W (DATA_W),
        .ACC_W  (ACC_W)
    ) u_systolic_array (
        .clk          (clk),
        .rst_n        (rst_n),
        .clear_acc    (sa_clear),
        .en           (sa_en),
        .dtype_fp16   (sa_dtype_fp16),
        .a_col        (sa_a_col),
        .b_row        (sa_b_row),
        .a_col_fp16   (sa_a_col_fp16),
        .b_row_fp16   (sa_b_row_fp16),
        .acc_out      (sa_acc),
        .acc_valid    (sa_acc_valid)
    );

    gemm_ctrl #(
        .ARRAY_M     (ARRAY_M),
        .ARRAY_N     (ARRAY_N),
        .DATA_W      (DATA_W),
        .ACC_W       (ACC_W),
        .SRAM_ADDR_W (SRAM0_AW)
    ) u_gemm_ctrl (
        .clk          (clk),
        .rst_n        (rst_n),

        .cmd_valid    (gm_cmd_valid),
        .cmd_src0     (gm_cmd_src0),
        .cmd_src1     (gm_cmd_src1),
        .cmd_dst      (gm_cmd_dst),
        .cmd_M        (gm_cmd_M),
        .cmd_N        (gm_cmd_N),
        .cmd_K        (gm_cmd_K),
        .cmd_flags    (gm_cmd_flags),
        .cmd_imm      (gm_cmd_imm),
        .cmd_dtype    (gm_cmd_dtype[0]),

        .sram_rd_en   (gemm_sram_rd_en),
        .sram_rd_addr (gemm_sram_rd_addr),
        .sram_rd_data (gemm_sram_rd_data),

        .sram_wr_en   (gemm_sram_wr_en),
        .sram_wr_addr (gemm_sram_wr_addr),
        .sram_wr_data (gemm_sram_wr_data),

        .acc_rd_en    (gemm_acc_rd_en),
        .acc_rd_addr  (gemm_acc_rd_addr),
        .acc_rd_data  (gemm_acc_rd_data),
        .acc_wr_en    (gemm_acc_wr_en),
        .acc_wr_addr  (gemm_acc_wr_addr),
        .acc_wr_data  (gemm_acc_wr_data),

        .sa_clear     (sa_clear),
        .sa_en        (sa_en),
        .sa_a_col     (sa_a_col),
        .sa_b_row     (sa_b_row),
        .sa_acc       (sa_acc),

        .dtype_fp16   (sa_dtype_fp16),
        .sa_a_col_fp16(sa_a_col_fp16),
        .sa_b_row_fp16(sa_b_row_fp16),

        .busy         (gemm_busy),
        .done         (gm_done)
    );

// ================================================================
    // Softmax Engine
    // ================================================================
    logic        sm_busy;
    logic        sm_rd_en;
    logic [SRAM0_AW-1:0] sm_rd_addr;
    logic        sm_wr_en;
    logic [SRAM0_AW-1:0] sm_wr_addr;
    logic [7:0]  sm_wr_data;
    logic        sm_scr_wr_en;
    logic [15:0] sm_scr_wr_addr;
    logic [15:0] sm_scr_wr_data;
    logic        sm_scr_rd_en;
    logic [15:0] sm_scr_rd_addr;
    logic [15:0] sm_scr_rd_data;

    softmax_engine u_softmax (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (sm_cmd_valid),
        .cmd_ready       (),
        .length          (sm_length),
        .src_base        (sm_src_base),
        .dst_base        (sm_dst_base),
        .scale_factor    (16'd256),
        .causal_mask_en  (1'b0),
        .causal_limit    (16'd0),
        .cmd_dtype       (sm_cmd_dtype),
        .sram_rd_en      (sm_rd_en),
        .sram_rd_addr    (sm_rd_addr),
        .sram_rd_data    (sm_rd_data),
        .sram_wr_en      (sm_wr_en),
        .sram_wr_addr    (sm_wr_addr),
        .sram_wr_data    (sm_wr_data),
        .scratch_wr_en   (sm_scr_wr_en),
        .scratch_wr_addr (sm_scr_wr_addr),
        .scratch_wr_data (sm_scr_wr_data),
        .scratch_rd_en   (sm_scr_rd_en),
        .scratch_rd_addr (sm_scr_rd_addr),
        .scratch_rd_data (sm_scr_rd_data),
        .busy            (sm_busy),
        .done            (sm_done)
    );

    // SCRATCH SRAM (16-bit x SCRATCH_DEPTH)
    sram_dp #(.DEPTH(SCRATCH_DEPTH), .WIDTH(16)) u_scratch_sram (
        .clk    (clk),
        .en_a   (sm_scr_rd_en),
        .we_a   (1'b0),
        .addr_a (sm_scr_rd_addr[SCR_AW-1:0]),
        .din_a  (16'd0),
        .dout_a (sm_scr_rd_data),
        .en_b   (sm_scr_wr_en),
        .we_b   (sm_scr_wr_en),
        .addr_b (sm_scr_wr_addr[SCR_AW-1:0]),
        .din_b  (sm_scr_wr_data),
        .dout_b ()
    );

    // ================================================================
    // Phase 3 Engines
    // ================================================================

    // --- Reduce Engine ---
    logic        re_busy;
    logic        re_rd_en, re_wr_en;
    logic [SRAM0_AW-1:0] re_rd_addr, re_wr_addr;
    logic [7:0]  re_rd_data, re_wr_data;

    reduce_engine #(.SRAM0_AW(SRAM0_AW)) u_reduce (
        .clk            (clk),
        .rst_n          (rst_n),
        .cmd_valid      (re_cmd_valid),
        .cmd_opcode     (re_cmd_opcode),
        .cmd_src_base   (re_cmd_src_base),
        .cmd_dst_base   (re_cmd_dst_base),
        .cmd_reduce_dim (re_cmd_reduce_dim),
        .cmd_outer_count(re_cmd_outer_count),
        .sram_rd_en     (re_rd_en),
        .sram_rd_addr   (re_rd_addr),
        .sram_rd_data   (re_rd_data),
        .sram_wr_en     (re_wr_en),
        .sram_wr_addr   (re_wr_addr),
        .sram_wr_data   (re_wr_data),
        .busy           (re_busy),
        .done           (re_done)
    );

    // --- Math Engine ---
    logic        me_busy;
    logic        me_rd_en, me_wr_en;
    logic [SRAM0_AW-1:0] me_rd_addr, me_wr_addr;
    logic [7:0]  me_rd_data, me_wr_data;

    math_engine #(.SRAM0_AW(SRAM0_AW)) u_math (
        .clk          (clk),
        .rst_n        (rst_n),
        .cmd_valid    (me_cmd_valid),
        .cmd_opcode   (me_cmd_opcode),
        .cmd_src_base (me_cmd_src_base),
        .cmd_dst_base (me_cmd_dst_base),
        .cmd_length   (me_cmd_length),
        .cmd_dtype    (me_cmd_dtype),
        .sram_rd_en   (me_rd_en),
        .sram_rd_addr (me_rd_addr),
        .sram_rd_data (me_rd_data),
        .sram_wr_en   (me_wr_en),
        .sram_wr_addr (me_wr_addr),
        .sram_wr_data (me_wr_data),
        .busy         (me_busy),
        .done         (me_done)
    );

    // --- Gather Engine ---
    logic        ga_busy;
    logic        ga_rd_en, ga_wr_en;
    logic [SRAM0_AW-1:0] ga_rd_addr, ga_wr_addr;
    logic [7:0]  ga_rd_data, ga_wr_data;

    gather_engine #(.SRAM0_AW(SRAM0_AW)) u_gather (
        .clk            (clk),
        .rst_n          (rst_n),
        .cmd_valid      (ga_cmd_valid),
        .cmd_src_base   (ga_cmd_src_base),
        .cmd_idx_base   (ga_cmd_idx_base),
        .cmd_dst_base   (ga_cmd_dst_base),
        .cmd_num_indices(ga_cmd_num_indices),
        .cmd_row_size   (ga_cmd_row_size),
        .cmd_num_rows   (ga_cmd_num_rows),
        .sram_rd_en     (ga_rd_en),
        .sram_rd_addr   (ga_rd_addr),
        .sram_rd_data   (ga_rd_data),
        .sram_wr_en     (ga_wr_en),
        .sram_wr_addr   (ga_wr_addr),
        .sram_wr_data   (ga_wr_data),
        .busy           (ga_busy),
        .done           (ga_done)
    );

    // --- Slice Engine ---
    logic        sl_busy;
    logic        sl_rd_en, sl_wr_en;
    logic [SRAM0_AW-1:0] sl_rd_addr, sl_wr_addr;
    logic [7:0]  sl_rd_data, sl_wr_data;

    slice_engine #(.SRAM0_AW(SRAM0_AW)) u_slice (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (sl_cmd_valid),
        .cmd_src_base    (sl_cmd_src_base),
        .cmd_dst_base    (sl_cmd_dst_base),
        .cmd_src_row_len (sl_cmd_src_row_len),
        .cmd_dst_row_len (sl_cmd_dst_row_len),
        .cmd_start_offset(sl_cmd_start_offset),
        .cmd_num_rows    (sl_cmd_num_rows),
        .sram_rd_en      (sl_rd_en),
        .sram_rd_addr    (sl_rd_addr),
        .sram_rd_data    (sl_rd_data),
        .sram_wr_en      (sl_wr_en),
        .sram_wr_addr    (sl_wr_addr),
        .sram_wr_data    (sl_wr_data),
        .busy            (sl_busy),
        .done            (sl_done)
    );

    // --- Concat Engine ---
    logic        ct_busy;
    logic        ct_rd_en, ct_wr_en;
    logic [SRAM0_AW-1:0] ct_rd_addr, ct_wr_addr;
    logic [7:0]  ct_rd_data, ct_wr_data;

    concat_engine #(.SRAM0_AW(SRAM0_AW)) u_concat (
        .clk              (clk),
        .rst_n            (rst_n),
        .cmd_valid        (ct_cmd_valid),
        .cmd_src0_base    (ct_cmd_src0_base),
        .cmd_src1_base    (ct_cmd_src1_base),
        .cmd_dst_base     (ct_cmd_dst_base),
        .cmd_src0_row_len (ct_cmd_src0_row_len),
        .cmd_src1_row_len (ct_cmd_src1_row_len),
        .cmd_num_rows     (ct_cmd_num_rows),
        .sram_rd_en       (ct_rd_en),
        .sram_rd_addr     (ct_rd_addr),
        .sram_rd_data     (ct_rd_data),
        .sram_wr_en       (ct_wr_en),
        .sram_wr_addr     (ct_wr_addr),
        .sram_wr_data     (ct_wr_data),
        .busy             (ct_busy),
        .done             (ct_done)
    );

    // --- AvgPool2D Engine ---
    logic        ap_rd_en;
    logic [SRAM0_AW-1:0] ap_rd_addr;
    logic        ap_wr_en;
    logic [SRAM0_AW-1:0] ap_wr_addr;
    logic [7:0]  ap_rd_data, ap_wr_data;

    avgpool2d_engine #(.SRAM0_AW(SRAM0_AW)) u_avgpool2d (
        .clk           (clk),
        .rst_n         (rst_n),
        .cmd_valid     (ap_cmd_valid),
        .cmd_src_base  (ap_cmd_src_base),
        .cmd_dst_base  (ap_cmd_dst_base),
        .cmd_C         (ap_cmd_C),
        .cmd_H         (ap_cmd_H),
        .cmd_W         (ap_cmd_W),
        .cmd_kh        (ap_cmd_kh),
        .cmd_kw        (ap_cmd_kw),
        .cmd_sh        (ap_cmd_sh),
        .cmd_sw        (ap_cmd_sw),
        .sram_rd_en    (ap_rd_en),
        .sram_rd_addr  (ap_rd_addr),
        .sram_rd_data  (ap_rd_data),
        .sram_wr_en    (ap_wr_en),
        .sram_wr_addr  (ap_wr_addr),
        .sram_wr_data  (ap_wr_data),
        .busy          (),
        .done          (ap_done)
    );

    // ================================================================
    // Phase 4 Engines
    // ================================================================

    // --- MaxPool2D Engine ---
    logic        mp_rd_en;
    logic [SRAM0_AW-1:0] mp_rd_addr;
    logic        mp_wr_en;
    logic [SRAM0_AW-1:0] mp_wr_addr;
    logic [7:0]  mp_rd_data, mp_wr_data;

    maxpool2d_engine #(.SRAM0_AW(SRAM0_AW)) u_maxpool2d (
        .clk           (clk),
        .rst_n         (rst_n),
        .cmd_valid     (mp_cmd_valid),
        .cmd_src_base  (mp_cmd_src_base),
        .cmd_dst_base  (mp_cmd_dst_base),
        .cmd_C         (mp_cmd_C),
        .cmd_H         (mp_cmd_H),
        .cmd_W         (mp_cmd_W),
        .cmd_kh        (mp_cmd_kh),
        .cmd_kw        (mp_cmd_kw),
        .cmd_sh        (mp_cmd_sh),
        .cmd_sw        (mp_cmd_sw),
        .sram_rd_en    (mp_rd_en),
        .sram_rd_addr  (mp_rd_addr),
        .sram_rd_data  (mp_rd_data),
        .sram_wr_en    (mp_wr_en),
        .sram_wr_addr  (mp_wr_addr),
        .sram_wr_data  (mp_wr_data),
        .busy          (),
        .done          (mp_done)
    );

    // --- Pad Engine ---
    logic        pd_rd_en;
    logic [SRAM0_AW-1:0] pd_rd_addr;
    logic        pd_wr_en;
    logic [SRAM0_AW-1:0] pd_wr_addr;
    logic [7:0]  pd_rd_data, pd_wr_data;

    pad_engine #(.SRAM0_AW(SRAM0_AW)) u_pad (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (pd_cmd_valid),
        .cmd_src_base    (pd_cmd_src_base),
        .cmd_dst_base    (pd_cmd_dst_base),
        .cmd_C           (pd_cmd_C),
        .cmd_H           (pd_cmd_H),
        .cmd_W           (pd_cmd_W),
        .cmd_pad_top     (pd_cmd_pad_top),
        .cmd_pad_bottom  (pd_cmd_pad_bottom),
        .cmd_pad_left    (pd_cmd_pad_left),
        .cmd_pad_right   (pd_cmd_pad_right),
        .sram_rd_en      (pd_rd_en),
        .sram_rd_addr    (pd_rd_addr),
        .sram_rd_data    (pd_rd_data),
        .sram_wr_en      (pd_wr_en),
        .sram_wr_addr    (pd_wr_addr),
        .sram_wr_data    (pd_wr_data),
        .busy            (),
        .done            (pd_done)
    );

    // --- Resize Nearest Engine ---
    logic        rz_rd_en;
    logic [SRAM0_AW-1:0] rz_rd_addr;
    logic        rz_wr_en;
    logic [SRAM0_AW-1:0] rz_wr_addr;
    logic [7:0]  rz_rd_data, rz_wr_data;

    resize_nearest_engine #(.SRAM0_AW(SRAM0_AW)) u_resize_nearest (
        .clk           (clk),
        .rst_n         (rst_n),
        .cmd_valid     (rz_cmd_valid),
        .cmd_src_base  (rz_cmd_src_base),
        .cmd_dst_base  (rz_cmd_dst_base),
        .cmd_C         (rz_cmd_C),
        .cmd_in_H      (rz_cmd_in_H),
        .cmd_in_W      (rz_cmd_in_W),
        .cmd_out_H     (rz_cmd_out_H),
        .cmd_out_W     (rz_cmd_out_W),
        .sram_rd_en    (rz_rd_en),
        .sram_rd_addr  (rz_rd_addr),
        .sram_rd_data  (rz_rd_data),
        .sram_wr_en    (rz_wr_en),
        .sram_wr_addr  (rz_wr_addr),
        .sram_wr_data  (rz_wr_data),
        .busy          (),
        .done          (rz_done)
    );

    // --- Cast Engine ---
    logic        ca_rd_en;
    logic [SRAM0_AW-1:0] ca_rd_addr;
    logic        ca_wr_en;
    logic [SRAM0_AW-1:0] ca_wr_addr;
    logic [7:0]  ca_rd_data, ca_wr_data;

    cast_engine #(.SRAM0_AW(SRAM0_AW)) u_cast (
        .clk           (clk),
        .rst_n         (rst_n),
        .cmd_valid     (ca_cmd_valid),
        .cmd_src_base  (ca_cmd_src_base),
        .cmd_dst_base  (ca_cmd_dst_base),
        .cmd_length    (ca_cmd_length),
        .cmd_src_dtype (ca_cmd_src_dtype),
        .cmd_dst_dtype (ca_cmd_dst_dtype),
        .sram_rd_en    (ca_rd_en),
        .sram_rd_addr  (ca_rd_addr),
        .sram_rd_data  (ca_rd_data),
        .sram_wr_en    (ca_wr_en),
        .sram_wr_addr  (ca_wr_addr),
        .sram_wr_data  (ca_wr_data),
        .busy          (),
        .done          (ca_done)
    );


    // -------------------------------------------------------------------------
    // Engine activity vector for debug only.
    // bit[0]=sm, [1]=re, [2]=me, [3]=ga, [4]=sl, [5]=ct,
    // [6]=ap, [7]=mp, [8]=pd, [9]=rz, [10]=ca.
    // -------------------------------------------------------------------------
    wire [10:0] aux_active_vec = {
        {4{1'b0}}, ct_busy,
        sl_busy, ga_busy, me_busy, re_busy, sm_busy
    };

    // -------------------------------------------------------------------------
    // graph_top
    // -------------------------------------------------------------------------
    graph_top #(
        .PROG_SRAM_AW (PROG_SRAM_AW),
        .SRAM0_AW     (SRAM0_AW)
    ) u_graph_top (
        .clk            (clk),
        .rst_n          (rst_n),

        .start          (start),
        .prog_len       (prog_len),
        .scheduler_mode (scheduler_mode),

        .prog_rd_en     (prog_rd_en),
        .prog_rd_addr   (prog_rd_addr),
        .prog_rd_data   (prog_rd_data),

        .td_rd0_addr    (td_rd0_addr),
        .td_rd0_data    (td_rd0_data),
        .td_rd1_addr    (td_rd1_addr),
        .td_rd1_data    (td_rd1_data),
        .td_rd2_addr    (td_rd2_addr),
        .td_rd2_data    (td_rd2_data),

        .gm_cmd_valid   (gm_cmd_valid),
        .gm_cmd_src0    (gm_cmd_src0),
        .gm_cmd_src1    (gm_cmd_src1),
        .gm_cmd_dst     (gm_cmd_dst),
        .gm_cmd_M       (gm_cmd_M),
        .gm_cmd_N       (gm_cmd_N),
        .gm_cmd_K       (gm_cmd_K),
        .gm_cmd_flags   (gm_cmd_flags),
        .gm_cmd_imm     (gm_cmd_imm),
        .gm_cmd_dtype   (gm_cmd_dtype),
        .gm_done        (gm_done),

// Softmax
        .sm_cmd_valid  (sm_cmd_valid),
        .sm_src_base   (sm_src_base),
        .sm_dst_base   (sm_dst_base),
        .sm_length     (sm_length),
        .sm_cmd_dtype  (sm_cmd_dtype),
        .sm_done       (sm_done),
        // Reduce
        .re_cmd_valid      (re_cmd_valid),
        .re_cmd_opcode     (re_cmd_opcode),
        .re_cmd_src_base   (re_cmd_src_base),
        .re_cmd_dst_base   (re_cmd_dst_base),
        .re_cmd_reduce_dim (re_cmd_reduce_dim),
        .re_cmd_outer_count(re_cmd_outer_count),
        .re_done           (re_done),
        // Math
        .me_cmd_valid  (me_cmd_valid),
        .me_cmd_opcode (me_cmd_opcode),
        .me_cmd_src_base(me_cmd_src_base),
        .me_cmd_dst_base(me_cmd_dst_base),
        .me_cmd_length (me_cmd_length),
        .me_cmd_dtype  (me_cmd_dtype),
        .me_done       (me_done),
        // Gather
        .ga_cmd_valid      (ga_cmd_valid),
        .ga_cmd_src_base   (ga_cmd_src_base),
        .ga_cmd_idx_base   (ga_cmd_idx_base),
        .ga_cmd_dst_base   (ga_cmd_dst_base),
        .ga_cmd_num_indices(ga_cmd_num_indices),
        .ga_cmd_row_size   (ga_cmd_row_size),
        .ga_cmd_num_rows   (ga_cmd_num_rows),
        .ga_done           (ga_done),
        // Slice
        .sl_cmd_valid      (sl_cmd_valid),
        .sl_cmd_src_base   (sl_cmd_src_base),
        .sl_cmd_dst_base   (sl_cmd_dst_base),
        .sl_cmd_src_row_len(sl_cmd_src_row_len),
        .sl_cmd_dst_row_len(sl_cmd_dst_row_len),
        .sl_cmd_start_offset(sl_cmd_start_offset),
        .sl_cmd_num_rows   (sl_cmd_num_rows),
        .sl_done           (sl_done),
        // Concat
        .ct_cmd_valid      (ct_cmd_valid),
        .ct_cmd_src0_base  (ct_cmd_src0_base),
        .ct_cmd_src1_base  (ct_cmd_src1_base),
        .ct_cmd_dst_base   (ct_cmd_dst_base),
        .ct_cmd_src0_row_len(ct_cmd_src0_row_len),
        .ct_cmd_src1_row_len(ct_cmd_src1_row_len),
        .ct_cmd_num_rows   (ct_cmd_num_rows),
        .ct_done           (ct_done),
        // AvgPool2D
        .ap_cmd_valid     (ap_cmd_valid),
        .ap_cmd_src_base  (ap_cmd_src_base),
        .ap_cmd_dst_base  (ap_cmd_dst_base),
        .ap_cmd_C         (ap_cmd_C),
        .ap_cmd_H         (ap_cmd_H),
        .ap_cmd_W         (ap_cmd_W),
        .ap_cmd_kh        (ap_cmd_kh),
        .ap_cmd_kw        (ap_cmd_kw),
        .ap_cmd_sh        (ap_cmd_sh),
        .ap_cmd_sw        (ap_cmd_sw),
        .ap_done          (ap_done),
        // MaxPool2D (Phase 4)
        .mp_cmd_valid     (mp_cmd_valid),
        .mp_cmd_src_base  (mp_cmd_src_base),
        .mp_cmd_dst_base  (mp_cmd_dst_base),
        .mp_cmd_C         (mp_cmd_C),
        .mp_cmd_H         (mp_cmd_H),
        .mp_cmd_W         (mp_cmd_W),
        .mp_cmd_kh        (mp_cmd_kh),
        .mp_cmd_kw        (mp_cmd_kw),
        .mp_cmd_sh        (mp_cmd_sh),
        .mp_cmd_sw        (mp_cmd_sw),
        .mp_done          (mp_done),
        // Pad (Phase 4)
        .pd_cmd_valid     (pd_cmd_valid),
        .pd_cmd_src_base  (pd_cmd_src_base),
        .pd_cmd_dst_base  (pd_cmd_dst_base),
        .pd_cmd_C         (pd_cmd_C),
        .pd_cmd_H         (pd_cmd_H),
        .pd_cmd_W         (pd_cmd_W),
        .pd_cmd_pad_top   (pd_cmd_pad_top),
        .pd_cmd_pad_bottom(pd_cmd_pad_bottom),
        .pd_cmd_pad_left  (pd_cmd_pad_left),
        .pd_cmd_pad_right (pd_cmd_pad_right),
        .pd_done          (pd_done),
        // Resize nearest (Phase 4)
        .rz_cmd_valid     (rz_cmd_valid),
        .rz_cmd_src_base  (rz_cmd_src_base),
        .rz_cmd_dst_base  (rz_cmd_dst_base),
        .rz_cmd_C         (rz_cmd_C),
        .rz_cmd_in_H      (rz_cmd_in_H),
        .rz_cmd_in_W      (rz_cmd_in_W),
        .rz_cmd_out_H     (rz_cmd_out_H),
        .rz_cmd_out_W     (rz_cmd_out_W),
        .rz_done          (rz_done),
        // Cast (Phase 4)
        .ca_cmd_valid     (ca_cmd_valid),
        .ca_cmd_src_base  (ca_cmd_src_base),
        .ca_cmd_dst_base  (ca_cmd_dst_base),
        .ca_cmd_length    (ca_cmd_length),
        .ca_cmd_src_dtype (ca_cmd_src_dtype),
        .ca_cmd_dst_dtype (ca_cmd_dst_dtype),
        .ca_done          (ca_done),

        .dma_cmd_valid  (dma_cmd_valid),
        .dma_ddr_addr   (dma_ddr_addr),
        .dma_sram_addr  (dma_sram_addr),
        .dma_length     (dma_length),
        .dma_direction  (dma_direction),
        .dma_strided    (dma_strided),
        .dma_stride     (dma_stride),
        .dma_count      (dma_count),
        .dma_block_len  (dma_block_len),
        .dma_done       (dma_done),

        .ew_rd_en       (ew_rd_en),
        .ew_rd_addr     (ew_rd_addr),
        .ew_rd_data     (ew_rd_data),
        .ew_wr_en       (ew_wr_en),
        .ew_wr_addr     (ew_wr_addr),
        .ew_wr_data     (ew_wr_data),
        .ew_busy        (ew_busy),

        .perf_total_cycles  (),
        .perf_gemm_cycles   (),
        .perf_softmax_cycles(),
        .perf_dma_cycles    (),
        .perf_reduce_cycles (),
        .perf_math_cycles   (),
        .perf_gather_cycles (),
        .perf_slice_cycles  (),
        .perf_concat_cycles (),
        .perf_avgpool_cycles(),
        .perf_ew_cycles     (),
        .perf_overlap_cycles(),
        .perf_stall_cycles  (),

        .graph_done     (graph_done),
        .graph_busy     (graph_busy),
        .graph_status   (graph_status),
        .graph_pc       (graph_pc),
        .graph_last_op  (graph_last_op)
    );



// -------------------------------------------------------------------------
// Owned SRAM0: 8-bank BRAM version
// -------------------------------------------------------------------------
localparam int SRAM0_DEPTH      = (1 << SRAM0_AW);      // bytes
localparam int SRAM0_NUM_BANKS  = 8;
localparam int SRAM0_BANK_AW    = SRAM0_AW - 3;
localparam int SRAM0_BANK_DEPTH = (1 << SRAM0_BANK_AW);

typedef enum logic [3:0] {
  SRAM_RD_NONE = 4'd0,
  SRAM_RD_GEMM = 4'd1,
  SRAM_RD_EW   = 4'd2,
  SRAM_RD_SM   = 4'd3,
  SRAM_RD_RE   = 4'd4,
  SRAM_RD_ME   = 4'd5,
  SRAM_RD_GA   = 4'd6,
  SRAM_RD_SL   = 4'd7,
  SRAM_RD_CT   = 4'd8,
  SRAM_RD_AP   = 4'd9,
  SRAM_RD_MP   = 4'd10,
  SRAM_RD_PD   = 4'd11,
  SRAM_RD_RZ   = 4'd12,
  SRAM_RD_CA   = 4'd13
} sram0_rd_src_e;

function automatic logic [2:0] sram0_bank(input logic [SRAM0_AW-1:0] byte_addr);
  return byte_addr[2:0];
endfunction

function automatic logic [SRAM0_BANK_AW-1:0] sram0_row(input logic [SRAM0_AW-1:0] byte_addr);
  return byte_addr[SRAM0_AW-1:3];
endfunction

// Port A: byte read arbiter
logic                         sram0_a_en   [SRAM0_NUM_BANKS];
logic                         sram0_a_we   [SRAM0_NUM_BANKS];
logic [SRAM0_BANK_AW-1:0]      sram0_a_addr [SRAM0_NUM_BANKS];
logic [7:0]                   sram0_a_din  [SRAM0_NUM_BANKS];
logic [7:0]                   sram0_a_dout [SRAM0_NUM_BANKS];

// Port B: DMA read/write + byte write arbiter
logic                         sram0_b_en   [SRAM0_NUM_BANKS];
logic                         sram0_b_we   [SRAM0_NUM_BANKS];
logic [SRAM0_BANK_AW-1:0]      sram0_b_addr [SRAM0_NUM_BANKS];
logic [7:0]                   sram0_b_din  [SRAM0_NUM_BANKS];
logic [7:0]                   sram0_b_dout [SRAM0_NUM_BANKS];

sram0_rd_src_e sram0_rd_src_d, sram0_rd_src_q;
logic [2:0]    sram0_rd_bank_d, sram0_rd_bank_q;

logic dma_sram_rd_en_q;
logic [2:0] dma_rd_bank_q [8];


    // -------------------------------------------------------------------------
    // The single SRAM0 write owner.
    //
    // This is the point where all hardware engines are really connected to SRAM0.
    // Priority is deterministic and suitable for SERIAL Graph scheduling:
    // DMA_LOAD > GEMM > Softmax > Reduce > Math > Gather > Slice > Concat
    //          > AvgPool > MaxPool > Pad > Resize > Cast > EW/ReLU.
    //
    // If scheduler_mode enables overlap later, replace this priority chain with
    // a true SRAM0 arbiter / BRAM port scheduler.
    // -------------------------------------------------------------------------

    
always_comb begin
  for (int i = 0; i < SRAM0_NUM_BANKS; i++) begin
    sram0_a_en[i]   = 1'b0;
    sram0_a_we[i]   = 1'b0;
    sram0_a_addr[i] = '0;
    sram0_a_din[i]  = 8'd0;
  end

  sram0_rd_src_d  = SRAM_RD_NONE;
  sram0_rd_bank_d = 3'd0;

  if (gemm_sram_rd_en) begin
    sram0_a_en[sram0_bank(gemm_sram_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(gemm_sram_rd_addr)] = sram0_row(gemm_sram_rd_addr);
    sram0_rd_src_d  = SRAM_RD_GEMM;
    sram0_rd_bank_d = sram0_bank(gemm_sram_rd_addr);
  end else if (ew_rd_en) begin
    sram0_a_en[sram0_bank(ew_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(ew_rd_addr)] = sram0_row(ew_rd_addr);
    sram0_rd_src_d  = SRAM_RD_EW;
    sram0_rd_bank_d = sram0_bank(ew_rd_addr);
  end else if (sm_rd_en) begin
    sram0_a_en[sram0_bank(sm_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(sm_rd_addr)] = sram0_row(sm_rd_addr);
    sram0_rd_src_d  = SRAM_RD_SM;
    sram0_rd_bank_d = sram0_bank(sm_rd_addr);
  end else if (re_rd_en) begin
    sram0_a_en[sram0_bank(re_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(re_rd_addr)] = sram0_row(re_rd_addr);
    sram0_rd_src_d  = SRAM_RD_RE;
    sram0_rd_bank_d = sram0_bank(re_rd_addr);
  end else if (me_rd_en) begin
    sram0_a_en[sram0_bank(me_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(me_rd_addr)] = sram0_row(me_rd_addr);
    sram0_rd_src_d  = SRAM_RD_ME;
    sram0_rd_bank_d = sram0_bank(me_rd_addr);
  end else if (ga_rd_en) begin
    sram0_a_en[sram0_bank(ga_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(ga_rd_addr)] = sram0_row(ga_rd_addr);
    sram0_rd_src_d  = SRAM_RD_GA;
    sram0_rd_bank_d = sram0_bank(ga_rd_addr);
  end else if (sl_rd_en) begin
    sram0_a_en[sram0_bank(sl_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(sl_rd_addr)] = sram0_row(sl_rd_addr);
    sram0_rd_src_d  = SRAM_RD_SL;
    sram0_rd_bank_d = sram0_bank(sl_rd_addr);
  end else if (ct_rd_en) begin
    sram0_a_en[sram0_bank(ct_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(ct_rd_addr)] = sram0_row(ct_rd_addr);
    sram0_rd_src_d  = SRAM_RD_CT;
    sram0_rd_bank_d = sram0_bank(ct_rd_addr);
  end else if (ap_rd_en) begin
    sram0_a_en[sram0_bank(ap_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(ap_rd_addr)] = sram0_row(ap_rd_addr);
    sram0_rd_src_d  = SRAM_RD_AP;
    sram0_rd_bank_d = sram0_bank(ap_rd_addr);
  end else if (mp_rd_en) begin
    sram0_a_en[sram0_bank(mp_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(mp_rd_addr)] = sram0_row(mp_rd_addr);
    sram0_rd_src_d  = SRAM_RD_MP;
    sram0_rd_bank_d = sram0_bank(mp_rd_addr);
  end else if (pd_rd_en) begin
    sram0_a_en[sram0_bank(pd_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(pd_rd_addr)] = sram0_row(pd_rd_addr);
    sram0_rd_src_d  = SRAM_RD_PD;
    sram0_rd_bank_d = sram0_bank(pd_rd_addr);
  end else if (rz_rd_en) begin
    sram0_a_en[sram0_bank(rz_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(rz_rd_addr)] = sram0_row(rz_rd_addr);
    sram0_rd_src_d  = SRAM_RD_RZ;
    sram0_rd_bank_d = sram0_bank(rz_rd_addr);
  end else if (ca_rd_en) begin
    sram0_a_en[sram0_bank(ca_rd_addr)]   = 1'b1;
    sram0_a_addr[sram0_bank(ca_rd_addr)] = sram0_row(ca_rd_addr);
    sram0_rd_src_d  = SRAM_RD_CA;
    sram0_rd_bank_d = sram0_bank(ca_rd_addr);
  end
end

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    sram0_rd_src_q  <= SRAM_RD_NONE;
    sram0_rd_bank_q <= 3'd0;

    gemm_sram_rd_data <= 8'd0;
    ew_rd_data        <= 8'd0;
    sm_rd_data        <= 8'd0;
    re_rd_data        <= 8'd0;
    me_rd_data        <= 8'd0;
    ga_rd_data        <= 8'd0;
    sl_rd_data        <= 8'd0;
    ct_rd_data        <= 8'd0;
    ap_rd_data        <= 8'd0;
    mp_rd_data        <= 8'd0;
    pd_rd_data        <= 8'd0;
    rz_rd_data        <= 8'd0;
    ca_rd_data        <= 8'd0;
  end else begin
    sram0_rd_src_q  <= sram0_rd_src_d;
    sram0_rd_bank_q <= sram0_rd_bank_d;

    unique case (sram0_rd_src_q)
      SRAM_RD_GEMM: gemm_sram_rd_data <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_EW:   ew_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_SM:   sm_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_RE:   re_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_ME:   me_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_GA:   ga_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_SL:   sl_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_CT:   ct_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_AP:   ap_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_MP:   mp_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_PD:   pd_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_RZ:   rz_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      SRAM_RD_CA:   ca_rd_data        <= sram0_a_dout[sram0_rd_bank_q];
      default: ;
    endcase
  end
end
always_comb
  for (int i = 0; i < SRAM0_NUM_BANKS; i++) begin
    sram0_b_en[i]   = 1'b0;
    sram0_b_we[i]   = 1'b0;
    sram0_b_addr[i] = '0;
    sram0_b_din[i]  = 8'd0;
  end

  // ------------------------------------------------------------
  // DMA LOAD: 64-bit -> 8 banks, one byte per bank
  // ------------------------------------------------------------
  if (dma_sram_wr_en) begin
    for (int k = 0; k < 8; k++) begin
      logic [SRAM0_AW-1:0] byte_addr;
      logic [2:0]          bank;

      byte_addr = dma_sram_wr_addr + k[SRAM0_AW-1:0];
      bank      = sram0_bank(byte_addr);

      if (dma_sram_wr_mask[k]) begin
        sram0_b_en[bank]   = 1'b1;
        sram0_b_we[bank]   = 1'b1;
        sram0_b_addr[bank] = sram0_row(byte_addr);
        sram0_b_din[bank]  = dma_sram_wr_data[8*k +: 8];
      end
    end

  // ------------------------------------------------------------
  // Byte writes from engines
  // ------------------------------------------------------------
  end else if (gemm_sram_wr_en) begin
    sram0_b_en[sram0_bank(gemm_sram_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(gemm_sram_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(gemm_sram_wr_addr)] = sram0_row(gemm_sram_wr_addr);
    sram0_b_din[sram0_bank(gemm_sram_wr_addr)]  = gemm_sram_wr_data;

  end else if (sm_wr_en) begin
    sram0_b_en[sram0_bank(sm_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(sm_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(sm_wr_addr)] = sram0_row(sm_wr_addr);
    sram0_b_din[sram0_bank(sm_wr_addr)]  = sm_wr_data;

  end else if (re_wr_en) begin
    sram0_b_en[sram0_bank(re_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(re_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(re_wr_addr)] = sram0_row(re_wr_addr);
    sram0_b_din[sram0_bank(re_wr_addr)]  = re_wr_data;

  end else if (me_wr_en) begin
    sram0_b_en[sram0_bank(me_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(me_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(me_wr_addr)] = sram0_row(me_wr_addr);
    sram0_b_din[sram0_bank(me_wr_addr)]  = me_wr_data;

  end else if (ga_wr_en) begin
    sram0_b_en[sram0_bank(ga_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(ga_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(ga_wr_addr)] = sram0_row(ga_wr_addr);
    sram0_b_din[sram0_bank(ga_wr_addr)]  = ga_wr_data;

  end else if (sl_wr_en) begin
    sram0_b_en[sram0_bank(sl_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(sl_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(sl_wr_addr)] = sram0_row(sl_wr_addr);
    sram0_b_din[sram0_bank(sl_wr_addr)]  = sl_wr_data;

  end else if (ct_wr_en) begin
    sram0_b_en[sram0_bank(ct_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(ct_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(ct_wr_addr)] = sram0_row(ct_wr_addr);
    sram0_b_din[sram0_bank(ct_wr_addr)]  = ct_wr_data;

  end else if (ap_wr_en) begin
    sram0_b_en[sram0_bank(ap_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(ap_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(ap_wr_addr)] = sram0_row(ap_wr_addr);
    sram0_b_din[sram0_bank(ap_wr_addr)]  = ap_wr_data;

  end else if (mp_wr_en) begin
    sram0_b_en[sram0_bank(mp_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(mp_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(mp_wr_addr)] = sram0_row(mp_wr_addr);
    sram0_b_din[sram0_bank(mp_wr_addr)]  = mp_wr_data;

  end else if (pd_wr_en) begin
    sram0_b_en[sram0_bank(pd_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(pd_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(pd_wr_addr)] = sram0_row(pd_wr_addr);
    sram0_b_din[sram0_bank(pd_wr_addr)]  = pd_wr_data;

  end else if (rz_wr_en) begin
    sram0_b_en[sram0_bank(rz_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(rz_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(rz_wr_addr)] = sram0_row(rz_wr_addr);
    sram0_b_din[sram0_bank(rz_wr_addr)]  = rz_wr_data;

  end else if (ca_wr_en) begin
    sram0_b_en[sram0_bank(ca_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(ca_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(ca_wr_addr)] = sram0_row(ca_wr_addr);
    sram0_b_din[sram0_bank(ca_wr_addr)]  = ca_wr_data;

  end else if (ew_wr_en) begin
    sram0_b_en[sram0_bank(ew_wr_addr)]   = 1'b1;
    sram0_b_we[sram0_bank(ew_wr_addr)]   = 1'b1;
    sram0_b_addr[sram0_bank(ew_wr_addr)] = sram0_row(ew_wr_addr);
    sram0_b_din[sram0_bank(ew_wr_addr)]  = ew_wr_data;

  // ------------------------------------------------------------
  // DMA STORE: 8 banks -> 64-bit, synchronous read
  // ------------------------------------------------------------
  end else if (dma_sram_rd_en) begin
    for (int k = 0; k < 8; k++) begin
      logic [SRAM0_AW-1:0] byte_addr;
      logic [2:0]          bank;

      byte_addr = dma_sram_rd_addr + k[SRAM0_AW-1:0];
      bank      = sram0_bank(byte_addr);

      sram0_b_en[bank]   = 1'b1;
      sram0_b_we[bank]   = 1'b0;
      sram0_b_addr[bank] = sram0_row(byte_addr);
    end
  end

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dma_sram_rd_en_q   <= 1'b0;
    dma_sram_rd_data   <= 64'd0;
    for (int k = 0; k < 8; k++) begin
      dma_rd_bank_q[k] <= 3'd0;
    end
  end else begin
    dma_sram_rd_en_q <= dma_sram_rd_en;

    if (dma_sram_rd_en) begin
      for (int k = 0; k < 8; k++) begin
        dma_rd_bank_q[k] <= sram0_bank(dma_sram_rd_addr + k[SRAM0_AW-1:0]);
      end
    end

    if (dma_sram_rd_en_q) begin
      for (int k = 0; k < 8; k++) begin
        dma_sram_rd_data[8*k +: 8] <= sram0_b_dout[dma_rd_bank_q[k]];
      end
    end
  end
end

genvar gb;
generate
  for (gb = 0; gb < SRAM0_NUM_BANKS; gb++) begin : gen_sram0_banks
    sram_dp #(
      .DEPTH  (SRAM0_BANK_DEPTH),
      .WIDTH  (8),
      .ADDR_W (SRAM0_BANK_AW)
    ) u_sram0_bank (
      .clk    (clk),

      .en_a   (sram0_a_en[gb]),
      .we_a   (sram0_a_we[gb]),
      .addr_a (sram0_a_addr[gb]),
      .din_a  (sram0_a_din[gb]),
      .dout_a (sram0_a_dout[gb]),

      .en_b   (sram0_b_en[gb]),
      .we_b   (sram0_b_we[gb]),
      .addr_b (sram0_b_addr[gb]),
      .din_b  (sram0_b_din[gb]),
      .dout_b (sram0_b_dout[gb])
    );
  end
endgenerate


`ifdef SYNTHESIS
`ifdef NPU_SIM_DEBUG
    // -------------------------------------------------------------------------
    // Debug prints
    // -------------------------------------------------------------------------

parameter int DBG_LEVEL = 1;
parameter int DBG_HEARTBEAT_CYCLES = 100_000;

localparam logic [1:0] DBG_DUMP_NONE = 2'd0;
localparam logic [1:0] DBG_DUMP_GEMM = 2'd1;
localparam logic [1:0] DBG_DUMP_EW   = 2'd2;
localparam logic [1:0] DBG_DUMP_RELU = 2'd3;

logic [1:0] dbg_dump_req;

    logic [15:0] dbg_prev_pc;
    logic [7:0]  dbg_prev_op;
    logic [31:0] dbg_hb_cnt;

    wire any_engine_cmd_valid =
        dma_cmd_valid |
        gm_cmd_valid  | sm_cmd_valid | re_cmd_valid | me_cmd_valid |
        ga_cmd_valid  | sl_cmd_valid | ct_cmd_valid | ap_cmd_valid |
        mp_cmd_valid  | pd_cmd_valid | rz_cmd_valid | ca_cmd_valid;

    wire any_engine_done =
        dma_done |
        gm_done  | sm_done | re_done | me_done |
        ga_done  | sl_done | ct_done | ap_done |
        mp_done  | pd_done | rz_done | ca_done;


    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dbg_prev_pc <= '0;
            dbg_prev_op <= '0;
            dbg_hb_cnt  <= '0;
        end else begin
            if (!graph_busy) begin
                dbg_hb_cnt  <= '0;
                dbg_prev_pc <= graph_pc;
                dbg_prev_op <= graph_last_op;
            end else begin
                // One-cycle delayed SRAM dump.
                // This avoids dumping SRAM0 in the same clock edge as the final write.
                if (dbg_dump_req != DBG_DUMP_NONE) begin
                    unique case (dbg_dump_req)
                        DBG_DUMP_GEMM: dump_sram0_5a00("AFTER_PC2_GEMM");
                        DBG_DUMP_EW:   dump_sram0_5a00("AFTER_PC4_EW_ADD");
                        DBG_DUMP_RELU: dump_sram0_5a00("AFTER_PC5_RELU");
                        default: ;
                    endcase
                    dbg_dump_req <= DBG_DUMP_NONE;
                end
                dbg_hb_cnt <= dbg_hb_cnt + 1'b1;


                if (DBG_LEVEL >= 1) begin
                    if ((graph_pc != dbg_prev_pc) || (graph_last_op != dbg_prev_op)) begin
                        $display("[%0t] GRAPH_PC pc=%0d last_op=0x%02x status=0x%08x",
                                 $time, graph_pc, graph_last_op, graph_status);

                        // Trigger dump when the previous PC just finished.
                        // pc 2: GEMM
                        // pc 4: EW_ADD / element-wise add
                        // pc 5: ReLU
                        if ((dbg_prev_pc == 16'd2) && (dbg_prev_op == 8'h20)) begin
                            dbg_dump_req <= DBG_DUMP_GEMM;
                        end else if ((dbg_prev_pc == 16'd4) && (dbg_prev_op == 8'h30)) begin
                            dbg_dump_req <= DBG_DUMP_EW;
                        end else if ((dbg_prev_pc == 16'd5) && (dbg_prev_op == 8'h38)) begin
                            dbg_dump_req <= DBG_DUMP_RELU;
                        end

                        dbg_prev_pc <= graph_pc;
                        dbg_prev_op <= graph_last_op;
                    end

                    if (any_engine_cmd_valid) begin
                        $display("[%0t] GRAPH_CMD pc=%0d op=0x%02x dma=%0d gm=%0d sm=%0d re=%0d me=%0d ga=%0d sl=%0d ct=%0d ap=%0d mp=%0d pd=%0d rz=%0d ca=%0d ew_rd=%0d ew_wr=%0d",
                                 $time, graph_pc, graph_last_op,
                                 dma_cmd_valid,
                                 gm_cmd_valid, sm_cmd_valid, re_cmd_valid, me_cmd_valid,
                                 ga_cmd_valid, sl_cmd_valid, ct_cmd_valid,
                                 ap_cmd_valid, mp_cmd_valid, pd_cmd_valid,
                                 rz_cmd_valid, ca_cmd_valid,
                                 ew_rd_en, ew_wr_en);
                    end

                    if (any_engine_done) begin
                        $display("[%0t] GRAPH_DONE_EVT pc=%0d op=0x%02x dma_done=%0d gm=%0d sm=%0d re=%0d me=%0d ga=%0d sl=%0d ct=%0d ap=%0d mp=%0d pd=%0d rz=%0d ca=%0d stub_active=0x%03x",
                                 $time, graph_pc, graph_last_op,
                                 dma_done,
                                 gm_done, sm_done, re_done, me_done,
                                 ga_done, sl_done, ct_done,
                                 ap_done, mp_done, pd_done,
                                 rz_done, ca_done,
                                 aux_active_vec);
                    end

                    if (dbg_hb_cnt == DBG_HEARTBEAT_CYCLES-1) begin
                        dbg_hb_cnt <= '0;
                        $display("[%0t] GRAPH_HB pc=%0d op=0x%02x busy=%0d status=0x%08x stub_active=0x%03x timer=%0d dma_cmd=%0d gemm_busy=%0d",
                                 $time, graph_pc, graph_last_op, graph_busy,
                                 graph_status, aux_active_vec, 8'd0,
                                 dma_cmd_valid, gemm_busy);
                    end
                end

                if (DBG_LEVEL >= 2) begin
                    if (dma_cmd_valid) begin
                        $display("[%0t] DMA_CMD dir=%0d ddr=0x%08x sram=0x%04x len=%0d strided=%0d stride=%0d count=%0d block_len=%0d",
                                 $time, dma_direction, dma_ddr_addr, dma_sram_addr,
                                 dma_length, dma_strided, dma_stride, dma_count, dma_block_len);
                    end
                    if (gm_cmd_valid) begin
                        $display("[%0t] GM_CMD src0=0x%04x src1=0x%04x dst=0x%04x M=%0d N=%0d K=%0d flags=0x%02x imm=0x%04x dtype=%0d",
                                 $time, gm_cmd_src0, gm_cmd_src1, gm_cmd_dst,
                                 gm_cmd_M, gm_cmd_N, gm_cmd_K,
                                 gm_cmd_flags, gm_cmd_imm, gm_cmd_dtype);
                    end
                end
            end
        end
    end
    `endif
`endif

endmodule

`default_nettype wire
