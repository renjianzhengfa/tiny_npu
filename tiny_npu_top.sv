// =============================================================================
// NPU Top-Level Module
// Transformer Inference Accelerator (INT8)
// Synthesizable SystemVerilog - Xilinx FPGA Target
// =============================================================================
`default_nettype none

module tiny_npu_top
    import npu_pkg::*;
    import isa_pkg::*;
    import axi_types_pkg::*;
#(
    parameter int ARRAY_M       = 16,
    parameter int ARRAY_N       = 16,
    parameter int P_DATA_W      = 8,
    parameter int P_ACC_W       = 32,
    parameter int P_AXI_DATA_W  = 128,
    parameter int P_AXI_ADDR_W  = 32,
    parameter int P_AXI_STRB_W  = P_AXI_DATA_W / 8,
    parameter int P_AXI_ID_W    = 4,
    parameter int P_SRAM_ADDR_W = 16,
    parameter int P_UCODE_DEPTH = 1024
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // ---- AXI4-Lite Slave (Control Plane) ----
    input  wire  [31:0]                 s_axil_awaddr,
    input  wire                         s_axil_awvalid,
    output wire                         s_axil_awready,
    input  wire  [31:0]                 s_axil_wdata,
    input  wire  [3:0]                  s_axil_wstrb,
    input  wire                         s_axil_wvalid,
    output wire                         s_axil_wready,
    output wire  [1:0]                  s_axil_bresp,
    output wire                         s_axil_bvalid,
    input  wire                         s_axil_bready,
    input  wire  [31:0]                 s_axil_araddr,
    input  wire                         s_axil_arvalid,
    output wire                         s_axil_arready,
    output wire  [31:0]                 s_axil_rdata,
    output wire  [1:0]                  s_axil_rresp,
    output wire                         s_axil_rvalid,
    input  wire                         s_axil_rready,

    // ---- AXI4 Master (Data Plane - DMA) ----
    output wire  [P_AXI_ID_W-1:0]      m_axi_arid,
    output wire  [P_AXI_ADDR_W-1:0]    m_axi_araddr,
    output wire  [7:0]                  m_axi_arlen,
    output wire  [2:0]                  m_axi_arsize,
    output wire  [1:0]                  m_axi_arburst,
    output wire                         m_axi_arvalid,
    input  wire                         m_axi_arready,
    input  wire  [P_AXI_ID_W-1:0]      m_axi_rid,
    input  wire  [P_AXI_DATA_W-1:0]    m_axi_rdata,
    input  wire  [1:0]                  m_axi_rresp,
    input  wire                         m_axi_rlast,
    input  wire                         m_axi_rvalid,
    output wire                         m_axi_rready,
    output wire  [P_AXI_ID_W-1:0]      m_axi_awid,
    output wire  [P_AXI_ADDR_W-1:0]    m_axi_awaddr,
    output wire  [7:0]                  m_axi_awlen,
    output wire  [2:0]                  m_axi_awsize,
    output wire  [1:0]                  m_axi_awburst,
    output wire                         m_axi_awvalid,
    input  wire                         m_axi_awready,
    output wire  [P_AXI_DATA_W-1:0]    m_axi_wdata,
    output wire  [P_AXI_STRB_W-1:0]    m_axi_wstrb,
    output wire                         m_axi_wlast,
    output wire                         m_axi_wvalid,
    input  wire                         m_axi_wready,
    input  wire  [P_AXI_ID_W-1:0]      m_axi_bid,
    input  wire  [1:0]                  m_axi_bresp,
    input  wire                         m_axi_bvalid,
    output wire                         m_axi_bready
);

localparam int GRAPH_PROG_AW = 9;

logic                         graph_prog_we;
logic [GRAPH_PROG_AW-1:0]     graph_prog_waddr;
logic [127:0]                 graph_prog_wdata;

// // Graph program SRAM moved into graph_compute_core_stage0.
//   if (graph_prog_rd_en) begin
//     graph_prog_rd_data <= graph_prog_mem[graph_prog_rd_addr];
//   end
// end


// axi_dma_wr.sv    logic

    localparam int UCODE_AW = $clog2(P_UCODE_DEPTH);
    
localparam int DMA_SMOKE_BUF_DEPTH = 16;
localparam int DMA_SMOKE_BUF_AW    = $clog2(DMA_SMOKE_BUF_DEPTH);

logic [P_AXI_DATA_W-1:0] dma_smoke_buf [0:DMA_SMOKE_BUF_DEPTH-1];

logic [DMA_SMOKE_BUF_AW-1:0] dma_smoke_wr_ptr;
logic [DMA_SMOKE_BUF_AW-1:0] dma_smoke_rd_ptr;
logic [DMA_SMOKE_BUF_AW:0]   dma_smoke_count;

logic dma_smoke_push;
logic dma_smoke_pop;

logic dma_wr_data_valid;
logic dma_wr_data_ready;
logic [P_AXI_DATA_W-1:0] dma_wr_data_in;
logic dma_wr_data_last;

    // =========================================================================
    // Internal signals
    // =========================================================================
    // Control registers
    wire        start_pulse;
    wire        soft_reset;
    wire [31:0] reg_ctrl;
    wire [31:0] reg_status;
    wire [31:0] reg_ucode_base;
    wire [31:0] reg_ucode_len;
    wire [31:0] reg_ddr_base_act;
    wire [31:0] reg_ddr_base_wgt;
    wire [31:0] reg_ddr_base_kv;
    wire [31:0] reg_ddr_base_out;
    wire [31:0] reg_model_hidden;
    wire [31:0] reg_model_heads;
    wire [31:0] reg_model_head_dim;
    wire [31:0] reg_seq_len;
    wire [31:0] reg_token_idx;
    wire [31:0] reg_debug_ctrl;
    wire [31:0] reg_exec_mode;

    // -------------------------------------------------------------------------
    // Graph-mode signals
    // IMPORTANT:
    // Put ALL graph declarations here before any use, because this file uses
    // `default_nettype none. Do not redeclare these signals later.
    // -------------------------------------------------------------------------
    logic [31:0] graph_status_w;
    logic [31:0] graph_pc_w;
    logic [31:0] graph_last_op_w;

    // Exec mode convenience
    wire graph_mode = (reg_exec_mode[0] == 1'b1);

    logic        graph_start;
    logic        graph_done;
    logic        graph_busy;
    logic [31:0] graph_status;
    logic [15:0] graph_pc;
    logic [7:0]  graph_last_op;

    logic        graph_dma_cmd_valid;
    logic [31:0] graph_dma_ddr_addr;
    logic [15:0] graph_dma_sram_addr;
    logic [15:0] graph_dma_length;
    logic        graph_dma_direction;
    logic        graph_dma_strided;
    logic [31:0] graph_dma_stride;
    logic [15:0] graph_dma_count;
    logic [15:0] graph_dma_block_len;
    logic        graph_dma_done;

    logic        graph_dma_direction_q;
    logic        graph_dma_cmd_seen;

    // Graph DMA command handshake signals.
    // Only declarations are placed here. Continuous assignments are placed later,
    // after dma_rd_int_cmd_ready / dma_wr_int_cmd_ready have been declared.
    logic        graph_dma_cmd_pending;
    logic        graph_dma_artifact_accept;
    logic        graph_dma_rd_req;
    logic        graph_dma_wr_req;
    logic        graph_dma_rd_accept;
    logic        graph_dma_wr_accept;
    logic        graph_dma_cmd_accept;
    logic        graph_dma_cmd_fire;

    // Simulation-only fast artifact mode enable.
    // TB can enable it by hierarchical assignment.
    logic        artifact_fast_en;
    initial artifact_fast_en = 1'b0;

    logic         tdesc_we;
    logic [7:0]   tdesc_waddr;
    logic [255:0] tdesc_wdata;

    logic [7:0]   td_rd0_addr, td_rd1_addr, td_rd2_addr;
    logic [255:0] td_rd0_data, td_rd1_data, td_rd2_data;

    logic npu_done;
    logic npu_busy;
    logic npu_error;
    logic npu_done_latch;
    logic graph_done_latch;

logic [31:0] graph_dma_abs_addr;

always_comb begin
    graph_dma_abs_addr = reg_ddr_base_act + graph_dma_ddr_addr;
end


    // -------------------------------------------------------------------------
    // DMA engine stream/control declarations
    // Keep these BEFORE graph real-DDR bridge logic. This file uses
    // `default_nettype none`, so the bridge cannot reference these before
    // declaration.
    // -------------------------------------------------------------------------
    logic        dma_rd_int_cmd_valid, dma_rd_int_cmd_ready;
    logic [31:0] dma_rd_int_cmd_addr;
    logic [23:0] dma_rd_int_cmd_len;
    logic [3:0]  dma_rd_int_cmd_tag;
    logic        dma_rd_data_valid, dma_rd_data_ready;
    logic [P_AXI_DATA_W-1:0] dma_rd_data;
    logic        dma_rd_data_last;
    logic        dma_rd_done, dma_rd_busy;

    logic        dma_wr_int_cmd_valid, dma_wr_int_cmd_ready;
    logic [31:0] dma_wr_int_cmd_addr;
    logic [23:0] dma_wr_int_cmd_len;
    logic [3:0]  dma_wr_int_cmd_tag;
    logic        dma_wr_done, dma_wr_busy;

    // =========================================================================
    // Graph artifact fast-simulation support
    // -------------------------------------------------------------------------
    // This is a simulation acceleration path for ONNX/Graph artifacts.
    // It bypasses Cheshire DRAM/DRAMSys for graph DMA only:
    //   - TB backdoor-loads ddr_64.hex into artifact_ddr_mem[].
    //   - Graph DMA_LOAD copies artifact_ddr_mem -> graph_sram.
    //   - Graph DMA_STORE copies graph_sram -> artifact_ddr_mem.
    //
    // Keep artifact_fast_en=0 for normal UART/AXI Graph DMA smoke.
    // TB can enable it by hierarchical assignment:
    //   i_fix.i_npu_wrap.i_tiny_npu.artifact_fast_en = 1'b1;
    // =========================================================================
    localparam int GRAPH_SRAM_BYTES      = 65536;

`ifdef NPU_KEEP_ARTIFACT_DDR
    // Optional legacy artifact-fast virtual DDR.
    // Keep this only when you explicitly want the old fast artifact path.
    localparam int ARTIFACT_DDR_WORDS    = 1048576; // 8 MiB / 8B
    localparam int ARTIFACT_DDR_WORD_AW  = 20;
`else
    // Real-DRAMSys flow is now the default.  Keep a 1-word dummy so old
    // artifact-fast code still compiles, but it is effectively disabled.
    localparam int ARTIFACT_DDR_WORDS    = 1;
    localparam int ARTIFACT_DDR_WORD_AW  = 1;
`endif

    // artifact_fast_en is declared in the top graph declaration area.
    // SRAM0/graph_sram has moved into u_graph_core.
    // In the real-DRAMSys flow, do not preload this memory.  It is a tiny dummy
    // by default.  Define NPU_KEEP_ARTIFACT_DDR to restore the old 8 MiB
    // artifact-fast virtual DDR for legacy tests.
    logic [63:0] artifact_ddr_mem [0:ARTIFACT_DDR_WORDS-1];

    logic        artifact_dma_done_pulse;


    // artifact_fast DMA streaming state.
    logic        artifact_dma_busy;
    logic        artifact_dma_dir_q;
    logic [31:0] artifact_dma_ddr_addr_q;
    logic [15:0] artifact_dma_sram_addr_q;
    logic [15:0] artifact_dma_len_q;
    logic [15:0] artifact_dma_pos_q;

    // Real external DDR DMA state.
    // When artifact_fast_en=0, axi_dma_rd/axi_dma_wr are the real DDR path.
    // These state registers bridge AXI streaming data and u_graph_core's owned SRAM0.
    logic [63:0] graph_sram_dma_rd_data;
    logic        real_dma_load_active;
    logic [15:0] real_dma_load_sram_addr_q;
    logic [15:0] real_dma_load_len_q;
    logic [15:0] real_dma_load_pos_q;

    logic        real_dma_store_active;
    logic [15:0] real_dma_store_sram_addr_q;
    logic [15:0] real_dma_store_len_q;
    logic [15:0] real_dma_store_pos_q;


    // EW/ReLU SRAM port from graph_top.
    logic        ew_rd_en;
    logic [15:0] ew_rd_addr;
    logic [7:0]  ew_rd_data;
    logic        ew_wr_en;
    logic [15:0] ew_wr_addr;
    logic [7:0]  ew_wr_data;
    logic        ew_busy;

    // Debug counter: only print the first EW/ReLU writes to avoid log flood.
    logic [31:0] ew_dbg_wr_count;

    // NOTE:
    // ew_busy is an output from graph_top in this repository version, so do not
    // drive it here.
    //
    // graph_sram has exactly ONE procedural write owner below:
    //   p_artifact_fast_dma
    // That block handles both:
    //   1) artifact_fast DMA LOAD into graph_sram
    //   2) EW/ReLU writeback into graph_sram
    // This avoids VCS multiple procedural drivers on the same memory.
    // EW/ReLU SRAM0 byte port is now internal to u_graph_core.

    // Graph GEMM/Pool command stubs for schedule-level artifact demo.
    // These let graph_top advance through compiled Graph programs, but they do
    // NOT produce numerically correct Conv/GEMM/Pool output. Golden alignment
    // requires replacing these stubs with real engines.
    logic        graph_gm_cmd_valid;
    logic [15:0] graph_gm_cmd_src0, graph_gm_cmd_src1, graph_gm_cmd_dst;
    logic [15:0] graph_gm_cmd_M, graph_gm_cmd_N, graph_gm_cmd_K;
    logic [7:0]  graph_gm_cmd_flags;
    logic [31:0] graph_gm_cmd_imm;
    logic [7:0]  graph_gm_cmd_dtype;
    logic        graph_gm_done;

    logic        graph_ap_cmd_valid, graph_ap_done;
    logic        graph_mp_cmd_valid, graph_mp_done;
    logic [15:0] graph_ap_cmd_src_base, graph_ap_cmd_dst_base;
    logic [15:0] graph_mp_cmd_src_base, graph_mp_cmd_dst_base;
    logic [15:0] graph_ap_cmd_C, graph_ap_cmd_H, graph_ap_cmd_W;
    logic [15:0] graph_mp_cmd_C, graph_mp_cmd_H, graph_mp_cmd_W;
    logic [7:0]  graph_ap_cmd_kh, graph_ap_cmd_kw, graph_ap_cmd_sh, graph_ap_cmd_sw;
    logic [7:0]  graph_mp_cmd_kh, graph_mp_cmd_kw, graph_mp_cmd_sh, graph_mp_cmd_sw;

`ifndef SYNTHESIS
    // one-shot done pulses for schedule-level demo
    logic [7:0] graph_gm_timer, graph_ap_timer, graph_mp_timer;
    logic       graph_gm_busy,  graph_ap_busy,  graph_mp_busy;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            graph_gm_done  <= 1'b0;
            graph_ap_done  <= 1'b0;
            graph_mp_done  <= 1'b0;
            graph_gm_busy  <= 1'b0;
            graph_ap_busy  <= 1'b0;
            graph_mp_busy  <= 1'b0;
            graph_gm_timer <= '0;
            graph_ap_timer <= '0;
            graph_mp_timer <= '0;
        end else begin
            graph_gm_done <= 1'b0;
            graph_ap_done <= 1'b0;
            graph_mp_done <= 1'b0;

            if (graph_gm_cmd_valid && !graph_gm_busy) begin
                graph_gm_busy  <= 1'b1;
                graph_gm_timer <= 8'd8;
                $display("[%0t] GRAPH_FAST GEMM_STUB src0=0x%04x src1=0x%04x dst=0x%04x M=%0d N=%0d K=%0d",
                         $time, graph_gm_cmd_src0, graph_gm_cmd_src1, graph_gm_cmd_dst,
                         graph_gm_cmd_M, graph_gm_cmd_N, graph_gm_cmd_K);
            end else if (graph_gm_busy) begin
                if (graph_gm_timer == 0) begin
                    graph_gm_busy <= 1'b0;
                    graph_gm_done <= 1'b1;
                end else begin
                    graph_gm_timer <= graph_gm_timer - 1'b1;
                end
            end

            if (graph_ap_cmd_valid && !graph_ap_busy) begin
                graph_ap_busy  <= 1'b1;
                graph_ap_timer <= 8'd8;
                $display("[%0t] GRAPH_FAST AVGPOOL_STUB src=0x%04x dst=0x%04x",
                         $time, graph_ap_cmd_src_base, graph_ap_cmd_dst_base);
            end else if (graph_ap_busy) begin
                if (graph_ap_timer == 0) begin
                    graph_ap_busy <= 1'b0;
                    graph_ap_done <= 1'b1;
                end else begin
                    graph_ap_timer <= graph_ap_timer - 1'b1;
                end
            end

            if (graph_mp_cmd_valid && !graph_mp_busy) begin
                graph_mp_busy  <= 1'b1;
                graph_mp_timer <= 8'd8;
                $display("[%0t] GRAPH_FAST MAXPOOL_STUB src=0x%04x dst=0x%04x",
                         $time, graph_mp_cmd_src_base, graph_mp_cmd_dst_base);
            end else if (graph_mp_busy) begin
                if (graph_mp_timer == 0) begin
                    graph_mp_busy <= 1'b0;
                    graph_mp_done <= 1'b1;
                end else begin
                    graph_mp_timer <= graph_mp_timer - 1'b1;
                end
            end
        end
    end

    
    // Fast graph DMA copy.
    // This version streams up to 8 bytes/cycle between top-level artifact_ddr_mem
    // and the SRAM0 owned by u_graph_core.
    //
    // It replaces the old zero-time for-loop copy into top-level graph_sram.
    // SRAM0 is now inside graph_compute, so this adapter talks to the compute
    // core through graph_sram_dma_* ports.
    // Unified SRAM0 DMA adapter into u_graph_core.
    // - artifact_fast_en=1 : artifact_ddr_mem <-> compute SRAM0.
    // - artifact_fast_en=0 : AXI DMA R/W streams <-> compute SRAM0.
    //
    // This is the key bridge after moving SRAM0 inside graph_compute:
    // real DDR data arrives in tiny_npu_top via axi_dma_rd, then this block
    // drives graph_sram_dma_wr_* so u_graph_core can write its internal SRAM0.
    // STORE reads u_graph_core SRAM0 through graph_sram_dma_rd_* and feeds
    // axi_dma_wr through dma_wr_data_in.
        // 64-bit/cycle access port into u_graph_core's owned SRAM0.
    logic        graph_sram_dma_wr_en;
    logic [15:0] graph_sram_dma_wr_addr;
    logic [63:0] graph_sram_dma_wr_data;
    logic [7:0]  graph_sram_dma_wr_mask;
    logic        graph_sram_dma_rd_en;
    logic [15:0] graph_sram_dma_rd_addr;
  

    always_comb begin : p_graph_sram_dma_comb
        integer lane;
        integer ddr_byte_addr;
        integer ddr_word_idx;
        integer ddr_lane;

        graph_sram_dma_wr_en   = 1'b0;
        graph_sram_dma_wr_addr = 16'd0;
        graph_sram_dma_wr_data = 64'd0;
        graph_sram_dma_wr_mask = 8'd0;

        graph_sram_dma_rd_en   = 1'b0;
        graph_sram_dma_rd_addr = 16'd0;

        // -------------------------------------------------------------
        // Fast artifact LOAD: artifact_ddr_mem -> compute SRAM0
        // -------------------------------------------------------------
        if (artifact_dma_busy && !artifact_dma_dir_q) begin
            graph_sram_dma_wr_en   = 1'b1;
            graph_sram_dma_wr_addr = artifact_dma_sram_addr_q + artifact_dma_pos_q;
            for (lane = 0; lane < 8; lane = lane + 1) begin
                if ((artifact_dma_pos_q + lane[15:0]) < artifact_dma_len_q) begin
                    ddr_byte_addr = artifact_dma_ddr_addr_q + artifact_dma_pos_q + lane;
                    ddr_word_idx  = ddr_byte_addr >> 3;
                    ddr_lane      = ddr_byte_addr & 7;
                    if (ddr_word_idx < ARTIFACT_DDR_WORDS) begin
                        graph_sram_dma_wr_data[8*lane +: 8] =
                            artifact_ddr_mem[ddr_word_idx][8*ddr_lane +: 8];
                        graph_sram_dma_wr_mask[lane] = 1'b1;
                    end
                end
            end
        end

        // -------------------------------------------------------------
        // Fast artifact STORE: compute SRAM0 -> artifact_ddr_mem
        // -------------------------------------------------------------
        if (artifact_dma_busy && artifact_dma_dir_q) begin
            graph_sram_dma_rd_en   = 1'b1;
            graph_sram_dma_rd_addr = artifact_dma_sram_addr_q + artifact_dma_pos_q;
        end

        // -------------------------------------------------------------
        // Real DDR LOAD: axi_dma_rd data_out -> compute SRAM0
        // The current AXI monitor shows size=3, so this bridge consumes one
        // 64-bit lower-lane beat at a time. Upper bits are ignored for now.
        // -------------------------------------------------------------
        if (!artifact_fast_en && graph_mode && real_dma_load_active &&
            dma_rd_data_valid && dma_rd_data_ready) begin
            graph_sram_dma_wr_en   = 1'b1;
            graph_sram_dma_wr_addr = real_dma_load_sram_addr_q + real_dma_load_pos_q;
            graph_sram_dma_wr_data = dma_rd_data[63:0];
            graph_sram_dma_wr_mask = 8'd0;
            for (lane = 0; lane < 8; lane = lane + 1) begin
                if ((real_dma_load_pos_q + lane[15:0]) < real_dma_load_len_q) begin
                    graph_sram_dma_wr_mask[lane] = 1'b1;
                end
            end
        end

        // -------------------------------------------------------------
        // Real DDR STORE: compute SRAM0 -> axi_dma_wr data_in
        // -------------------------------------------------------------
        if (!artifact_fast_en && graph_mode && real_dma_store_active) begin
            graph_sram_dma_rd_en   = 1'b1;
            graph_sram_dma_rd_addr = real_dma_store_sram_addr_q + real_dma_store_pos_q;
        end
    end

    always @(posedge clk or negedge rst_n) begin : p_artifact_fast_dma
        integer lane;
        integer ddr_byte_addr;
        integer ddr_word_idx;
        integer ddr_lane;

        if (!rst_n) begin
            artifact_dma_done_pulse <= 1'b0;
            artifact_dma_busy       <= 1'b0;
            artifact_dma_dir_q      <= 1'b0;
            artifact_dma_ddr_addr_q <= 32'd0;
            artifact_dma_sram_addr_q<= 16'd0;
            artifact_dma_len_q      <= 16'd0;
            artifact_dma_pos_q      <= 16'd0;
        end else begin
            artifact_dma_done_pulse <= 1'b0;

            if (!artifact_dma_busy) begin
                if (artifact_fast_en && graph_mode && graph_dma_cmd_fire) begin
                    $display("[%0t] GRAPH_FAST_DMA_START %s ddr_off=0x%08x sram=0x%04x len=%0d",
                             $time,
                             graph_dma_direction ? "STORE" : "LOAD ",
                             graph_dma_ddr_addr,
                             graph_dma_sram_addr,
                             graph_dma_length);

                    artifact_dma_busy        <= (graph_dma_length != 16'd0);
                    artifact_dma_dir_q       <= graph_dma_direction;
                    artifact_dma_ddr_addr_q  <= graph_dma_ddr_addr;
                    artifact_dma_sram_addr_q <= graph_dma_sram_addr;
                    artifact_dma_len_q       <= graph_dma_length;
                    artifact_dma_pos_q       <= 16'd0;

                    if (graph_dma_length == 16'd0) begin
                        artifact_dma_done_pulse <= 1'b1;
                    end
                end
            end else begin
                // STORE path: copy current 8-byte SRAM beat into artifact DDR.
                if (artifact_dma_dir_q) begin
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        if ((artifact_dma_pos_q + lane[15:0]) < artifact_dma_len_q) begin
                            ddr_byte_addr = artifact_dma_ddr_addr_q + artifact_dma_pos_q + lane;
                            ddr_word_idx  = ddr_byte_addr >> 3;
                            ddr_lane      = ddr_byte_addr & 7;
                            if (ddr_word_idx < ARTIFACT_DDR_WORDS) begin
                                artifact_ddr_mem[ddr_word_idx][8*ddr_lane +: 8] <=
                                    graph_sram_dma_rd_data[8*lane +: 8];
                            end else begin
                                $error("[%0t] GRAPH_FAST_DMA STORE address out of range: byte=0x%08x word=%0d",
                                       $time, ddr_byte_addr, ddr_word_idx);
                            end
                        end
                    end
                end

                if (artifact_dma_pos_q + 16'd8 >= artifact_dma_len_q) begin
                    artifact_dma_busy       <= 1'b0;
                    artifact_dma_done_pulse <= 1'b1;
                    $display("[%0t] GRAPH_FAST_DMA_DONE %s ddr_off=0x%08x sram=0x%04x len=%0d",
                             $time,
                             artifact_dma_dir_q ? "STORE" : "LOAD ",
                             artifact_dma_ddr_addr_q,
                             artifact_dma_sram_addr_q,
                             artifact_dma_len_q);
                end else begin
                    artifact_dma_pos_q <= artifact_dma_pos_q + 16'd8;
                end
            end
        end
    end
`endif

    // Real external-DDR Graph DMA stream control.
    // This block replaces the old dma_smoke_buf path for graph_mode.
    // It does not create AXI transactions; axi_dma_rd/axi_dma_wr still do that.
    // It only tracks where each returned/read-out beat belongs in compute SRAM0.
    always_ff @(posedge clk or negedge rst_n) begin : p_real_graph_dma_stream
        if (!rst_n) begin
            real_dma_load_active       <= 1'b0;
            real_dma_load_sram_addr_q  <= 16'd0;
            real_dma_load_len_q        <= 16'd0;
            real_dma_load_pos_q        <= 16'd0;

            real_dma_store_active      <= 1'b0;
            real_dma_store_sram_addr_q <= 16'd0;
            real_dma_store_len_q       <= 16'd0;
            real_dma_store_pos_q       <= 16'd0;
        end else begin
            // Start a real DDR LOAD after the graph DMA read command is accepted.
            if (graph_dma_rd_accept) begin
                real_dma_load_active      <= (graph_dma_length != 16'd0);
                real_dma_load_sram_addr_q <= graph_dma_sram_addr;
                real_dma_load_len_q       <= graph_dma_length;
                real_dma_load_pos_q       <= 16'd0;
                // $display("[%0t] REAL_DMA_LOAD_START sram=0x%04x len=%0d",
                         $time, graph_dma_sram_addr, graph_dma_length);
            end else if (real_dma_load_active && dma_rd_data_valid && dma_rd_data_ready) begin
                if (real_dma_load_pos_q + 16'd8 >= real_dma_load_len_q) begin
                    real_dma_load_pos_q <= real_dma_load_len_q;
                end else begin
                    real_dma_load_pos_q <= real_dma_load_pos_q + 16'd8;
                end

                if (real_dma_load_pos_q < 16'd64) begin
                    // $display("[%0t] REAL_DMA_LOAD_WR sram=0x%04x data=0x%016x pos=%0d last=%0d",
                             $time,
                             real_dma_load_sram_addr_q + real_dma_load_pos_q,
                             dma_rd_data[63:0],
                             real_dma_load_pos_q,
                             dma_rd_data_last);
                end
            end

            if (dma_rd_done) begin
                if (real_dma_load_active) begin
                    // $display("[%0t] REAL_DMA_LOAD_DONE sram=0x%04x len=%0d pos=%0d",
                             $time,
                             real_dma_load_sram_addr_q,
                             real_dma_load_len_q,
                             real_dma_load_pos_q);
                end
                real_dma_load_active <= 1'b0;
            end

            // Start a real DDR STORE after the graph DMA write command is accepted.
            if (graph_dma_wr_accept) begin
                real_dma_store_active      <= (graph_dma_length != 16'd0);
                real_dma_store_sram_addr_q <= graph_dma_sram_addr;
                real_dma_store_len_q       <= graph_dma_length;
                real_dma_store_pos_q       <= 16'd0;
                $display("[%0t] REAL_DMA_STORE_START sram=0x%04x len=%0d",
                         $time, graph_dma_sram_addr, graph_dma_length);
            end else if (real_dma_store_active && dma_wr_data_valid && dma_wr_data_ready) begin
                if (real_dma_store_pos_q + 16'd8 >= real_dma_store_len_q) begin
                    real_dma_store_pos_q  <= real_dma_store_len_q;
                    real_dma_store_active <= 1'b0;
                end else begin
                    real_dma_store_pos_q <= real_dma_store_pos_q + 16'd8;
                end

                if (real_dma_store_pos_q < 16'd64) begin
                    $display("[%0t] REAL_DMA_STORE_RD sram=0x%04x data=0x%016x pos=%0d last=%0d",
                             $time,
                             real_dma_store_sram_addr_q + real_dma_store_pos_q,
                             graph_sram_dma_rd_data,
                             real_dma_store_pos_q,
                             dma_wr_data_last);
                end
            end

            if (dma_wr_done) begin
                if (real_dma_store_active) begin
                    $display("[%0t] REAL_DMA_STORE_DONE sram=0x%04x len=%0d pos=%0d",
                             $time,
                             real_dma_store_sram_addr_q,
                             real_dma_store_len_q,
                             real_dma_store_pos_q);
                end
                real_dma_store_active <= 1'b0;
            end
        end
    end



    wire rst_int_n;
    assign rst_int_n = rst_n & ~soft_reset;

    assign graph_start     = start_pulse && graph_mode;
    assign graph_status_w  = graph_status;
    assign graph_pc_w      = {16'd0, graph_pc};
    assign graph_last_op_w = {24'd0, graph_last_op};

    // Scoreboard
    logic                        issue_valid;
    logic [2:0]                  issue_engine_id;
    logic [NUM_ENGINES-1:0]      engine_busy;
    logic [NUM_ENGINES-1:0]      can_issue;
    logic [7:0]                  can_issue_8;  // padded for ucode_decode (8-wide)
    logic [NUM_ENGINES-1:0]      engine_done_vec;
    logic                        all_idle;

    // Barrier
    logic barrier_trigger;
    logic barrier_stall;

    // Fetch -> Decode
    logic              fetch_instr_valid;
    logic              fetch_instr_ready;
    logic [127:0]      fetch_instr_data;
    logic [P_SRAM_ADDR_W-1:0] fetch_pc;
    logic              fetch_done;

    // UCODE SRAM
    logic                       uc_rd_en;
    logic [P_SRAM_ADDR_W-1:0]  uc_rd_addr;
    logic [127:0]               uc_rd_data;
    logic                       uc_rd_valid;

    // Engine command signals from decode
    logic       gemm_cmd_valid, softmax_cmd_valid, layernorm_cmd_valid;
    logic       gelu_cmd_valid, vec_cmd_valid, dma_rd_cmd_valid, dma_wr_cmd_valid;
    logic       kv_cmd_valid;

    // Engine status
    logic gemm_done, softmax_done, layernorm_done, gelu_done, vec_done;

    // -------------------------------------------------------------------------
    // Graph DMA command accept handshake
    // -------------------------------------------------------------------------
    assign graph_dma_cmd_pending =
        graph_mode && graph_dma_cmd_valid && !graph_dma_cmd_seen;

    assign graph_dma_artifact_accept =
        artifact_fast_en && graph_dma_cmd_pending;

    assign graph_dma_rd_req =
        !artifact_fast_en && graph_dma_cmd_pending && !graph_dma_direction;

    assign graph_dma_wr_req =
        !artifact_fast_en && graph_dma_cmd_pending && graph_dma_direction;

    assign graph_dma_rd_accept =
        graph_dma_rd_req && dma_rd_int_cmd_ready;

    assign graph_dma_wr_accept =
        graph_dma_wr_req && dma_wr_int_cmd_ready;

    assign graph_dma_cmd_accept =
        graph_dma_artifact_accept || graph_dma_rd_accept || graph_dma_wr_accept;

    // Only artifact-fast copy block uses this old name.
    assign graph_dma_cmd_fire =
        graph_dma_artifact_accept;

    // Decoded fields from decoder
    logic [15:0] dec_gemm_src0, dec_gemm_src1, dec_gemm_dst;
    logic [15:0] dec_gemm_m, dec_gemm_n, dec_gemm_k;
    logic [7:0]  dec_gemm_flags;
    logic [15:0] dec_dma_rd_src, dec_dma_rd_dst, dec_dma_rd_len;
    logic [7:0]  dec_dma_rd_flags;
    logic [15:0] dec_dma_wr_src, dec_dma_wr_dst, dec_dma_wr_len;
    logic [7:0]  dec_dma_wr_flags;
    ucode_instr_t decoded_instr;
    logic program_end;

    // Compose engine_done vector
    assign engine_done_vec[0] = gemm_done;
    assign engine_done_vec[1] = softmax_done;
    assign engine_done_vec[2] = layernorm_done;
    assign engine_done_vec[3] = gelu_done;
    assign engine_done_vec[4] = vec_done;
    assign engine_done_vec[5] = dma_rd_done | dma_wr_done;

    // In graph mode, legacy pipeline is idle; graph_top drives done/busy/error
    // In legacy mode, legacy pipeline is active
    wire legacy_start_pulse = start_pulse && !graph_mode;


always_ff @(posedge clk or negedge rst_int_n) begin
    if (!rst_int_n)
        graph_done_latch <= 1'b0;
    else if (graph_start)
        graph_done_latch <= 1'b0;
    else if (graph_done)
        graph_done_latch <= 1'b1;
end

assign npu_done  = graph_mode ? graph_done_latch : npu_done_latch;
assign npu_busy  = graph_mode ? graph_busy       : |engine_busy;
assign npu_error = 1'b0;


    // Latch done signal so it persists until next START
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n)
            npu_done_latch <= 1'b0;
        else if (start_pulse)
            npu_done_latch <= 1'b0;
        else if (!graph_mode && (program_end || fetch_done))
            npu_done_latch <= 1'b1;
    end

    logic                  ucode_axil_we;
    logic [UCODE_AW-1:0]   ucode_axil_waddr;
    logic [127:0]          ucode_axil_wdata;

    // =========================================================================
    // AXI4-Lite Register Bank
    // =========================================================================
    axi_lite_regs  #(
            .UCODE_AW      (UCODE_AW),
            .GRAPH_PROG_AW (GRAPH_PROG_AW),
            .TDESC_AW      (8)
    )
    u_regs (
        .clk              (clk),
        .rst_n            (rst_n),
        .s_axil_awaddr    (s_axil_awaddr),
        .s_axil_awvalid   (s_axil_awvalid),
        .s_axil_awready   (s_axil_awready),
        .s_axil_wdata     (s_axil_wdata),
        .s_axil_wstrb     (s_axil_wstrb),
        .s_axil_wvalid    (s_axil_wvalid),
        .s_axil_wready    (s_axil_wready),
        .s_axil_bresp     (s_axil_bresp),
        .s_axil_bvalid    (s_axil_bvalid),
        .s_axil_bready    (s_axil_bready),
        .s_axil_araddr    (s_axil_araddr),
        .s_axil_arvalid   (s_axil_arvalid),
        .s_axil_arready   (s_axil_arready),
        .s_axil_rdata     (s_axil_rdata),
        .s_axil_rresp     (s_axil_rresp),
        .s_axil_rvalid    (s_axil_rvalid),
        .s_axil_rready    (s_axil_rready),
        .done_i           (npu_done),
        .busy_i           (npu_busy),
        .error_i          (npu_error),
        .ctrl_o           (reg_ctrl),
        .status_o         (reg_status),
        .ucode_base_o     (reg_ucode_base),
        .ucode_len_o      (reg_ucode_len),
        .ddr_base_act_o   (reg_ddr_base_act),
        .ddr_base_wgt_o   (reg_ddr_base_wgt),
        .ddr_base_kv_o    (reg_ddr_base_kv),
        .ddr_base_out_o   (reg_ddr_base_out),
        .model_hidden_o   (reg_model_hidden),
        .model_heads_o    (reg_model_heads),
        .model_head_dim_o (reg_model_head_dim),
        .seq_len_o        (reg_seq_len),
        .token_idx_o      (reg_token_idx),
        .debug_ctrl_o     (reg_debug_ctrl),
        .exec_mode_o      (reg_exec_mode),
        .graph_status_i   (graph_status_w),
        .graph_pc_i       (graph_pc_w),
        .graph_last_op_i  (graph_last_op_w),
        .start_pulse_o    (start_pulse),
        .soft_reset_o     (soft_reset),

        .ucode_we_o         (ucode_axil_we),
        .ucode_waddr_o      (ucode_axil_waddr),
        .ucode_wdata_o      (ucode_axil_wdata),

        .graph_prog_we_o   (graph_prog_we),
        .graph_prog_waddr_o(graph_prog_waddr),
        .graph_prog_wdata_o(graph_prog_wdata),

        .tdesc_we_o        (tdesc_we),
        .tdesc_waddr_o     (tdesc_waddr),
        .tdesc_wdata_o     (tdesc_wdata)
    );

    // =========================================================================
    // AXI4 DMA Read Master
    // =========================================================================
    axi_dma_rd #(
        .AXI_DATA_W    (P_AXI_DATA_W),
        .AXI_ADDR_W    (P_AXI_ADDR_W)
    ) u_dma_rd (
        .clk            (clk),
        .rst_n          (rst_int_n),
        .cmd_valid      (dma_rd_int_cmd_valid),
        .cmd_ready      (dma_rd_int_cmd_ready),
        .cmd_addr       (dma_rd_int_cmd_addr),
        .cmd_len        (dma_rd_int_cmd_len),
        .cmd_tag        (dma_rd_int_cmd_tag),
        .m_axi_arid     (m_axi_arid),
        .m_axi_araddr   (m_axi_araddr),
        .m_axi_arlen    (m_axi_arlen),
        .m_axi_arsize   (m_axi_arsize),
        .m_axi_arburst  (m_axi_arburst),
        .m_axi_arvalid  (m_axi_arvalid),
        .m_axi_arready  (m_axi_arready),
        .m_axi_rid      (m_axi_rid),
        .m_axi_rdata    (m_axi_rdata),
        .m_axi_rresp    (m_axi_rresp),
        .m_axi_rlast    (m_axi_rlast),
        .m_axi_rvalid   (m_axi_rvalid),
        .m_axi_rready   (m_axi_rready),
        .data_valid     (dma_rd_data_valid),
        .data_ready     (dma_rd_data_ready),
        .data_out       (dma_rd_data),
        .data_last      (dma_rd_data_last),
        .data_tag       (),
        .busy           (dma_rd_busy),
        .done           (dma_rd_done),
        .error          ()
    );

    // =========================================================================
    // AXI4 DMA Write Master
    // =========================================================================

    logic [P_AXI_DATA_W-1:0] dma_wr_smoke_data;
    assign dma_wr_smoke_data = {{(P_AXI_DATA_W-64){1'b0}}, 64'hCAFE_BABE_1234_5678};

    axi_dma_wr #(
        .AXI_DATA_W    (P_AXI_DATA_W),
        .AXI_ADDR_W    (P_AXI_ADDR_W)
    ) u_dma_wr (
        .clk            (clk),
        .rst_n          (rst_int_n),
        .cmd_valid      (dma_wr_int_cmd_valid),
        .cmd_ready      (dma_wr_int_cmd_ready),
        .cmd_addr       (dma_wr_int_cmd_addr),
        .cmd_len        (dma_wr_int_cmd_len),
        .cmd_tag        (dma_wr_int_cmd_tag),
        .m_axi_awid     (m_axi_awid),
        .m_axi_awaddr   (m_axi_awaddr),
        .m_axi_awlen    (m_axi_awlen),
        .m_axi_awsize   (m_axi_awsize),
        .m_axi_awburst  (m_axi_awburst),
        .m_axi_awvalid  (m_axi_awvalid),
        .m_axi_awready  (m_axi_awready),
        .m_axi_wdata    (m_axi_wdata),
        .m_axi_wstrb    (m_axi_wstrb),
        .m_axi_wlast    (m_axi_wlast),
        .m_axi_wvalid   (m_axi_wvalid),
        .m_axi_wready   (m_axi_wready),
        .m_axi_bid      (m_axi_bid),
        .m_axi_bresp    (m_axi_bresp),
        .m_axi_bvalid   (m_axi_bvalid),
        .m_axi_bready   (m_axi_bready),

        .data_valid     (dma_wr_data_valid),
        .data_ready     (dma_wr_data_ready),
        .data_in        (dma_wr_data_in),
        .data_last      (dma_wr_data_last),

        .busy           (dma_wr_busy),
        .done           (dma_wr_done),
        .error          ()
    );


    // =========================================================================
    // UCODE SRAM (128-bit wide, single bank)
    // =========================================================================
    sram_dp #(
        .DEPTH (P_UCODE_DEPTH),
        .WIDTH (128)
    ) u_ucode_sram (
        .clk    (clk),
        .en_a   (uc_rd_en),
        .we_a   (1'b0),
        .addr_a (uc_rd_addr[$clog2(P_UCODE_DEPTH)-1:0]),
        .din_a  (128'b0),
        .dout_a (uc_rd_data),

        
        .en_b   (ucode_axil_we),
        .we_b   (ucode_axil_we),
        .addr_b (ucode_axil_waddr),
        .din_b  (ucode_axil_wdata),
        .dout_b ()
    );

    // SRAM read valid (1-cycle latency)
    logic uc_rd_en_d;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) uc_rd_en_d <= 1'b0;
        else            uc_rd_en_d <= uc_rd_en;
    end
    assign uc_rd_valid = uc_rd_en_d;

    // =========================================================================
    // Microcode Fetch
    // =========================================================================
    ucode_fetch #(
        .INSTR_W     (128),
        .SRAM_ADDR_W (P_SRAM_ADDR_W)
    ) u_fetch (
        .clk            (clk),
        .rst_n          (rst_int_n),
        .start          (legacy_start_pulse),
        .stop           (1'b0),
        .ucode_base_addr({P_SRAM_ADDR_W{1'b0}}),
        .ucode_len      (reg_ucode_len[P_SRAM_ADDR_W-1:0]),
        .rd_en          (uc_rd_en),
        .rd_addr        (uc_rd_addr),
        .rd_data        (uc_rd_data),
        .rd_valid       (uc_rd_valid),
        .instr_valid    (fetch_instr_valid),
        .instr_ready    (fetch_instr_ready),
        .instr_data     (fetch_instr_data),
        .pc             (fetch_pc),
        .done           (fetch_done),
        .busy           ()
    );

    // =========================================================================
    // Microcode Decode & Dispatch
    // =========================================================================
    // Pad can_issue to 8 bits: engines 6,7 not present, mark always free
    assign can_issue_8 = {2'b11, can_issue};

    ucode_decode u_decode (
        .clk                (clk),
        .rst_n              (rst_int_n),
        .instr_valid        (fetch_instr_valid),
        .instr_data         (fetch_instr_data),
        .instr_ready        (fetch_instr_ready),
        // Scoreboard
        .can_issue          (can_issue_8),
        .all_idle           (all_idle),
        // GEMM
        .gemm_cmd_valid     (gemm_cmd_valid),
        .gemm_cmd_src0      (dec_gemm_src0),
        .gemm_cmd_src1      (dec_gemm_src1),
        .gemm_cmd_dst       (dec_gemm_dst),
        .gemm_cmd_m         (dec_gemm_m),
        .gemm_cmd_n         (dec_gemm_n),
        .gemm_cmd_k         (dec_gemm_k),
        .gemm_cmd_flags     (dec_gemm_flags),
        // Softmax
        .softmax_cmd_valid  (softmax_cmd_valid),
        .softmax_cmd_src0   (),
        .softmax_cmd_dst    (),
        .softmax_cmd_len    (),
        .softmax_cmd_flags  (),
        // LayerNorm
        .layernorm_cmd_valid(layernorm_cmd_valid),
        .layernorm_cmd_src0 (),
        .layernorm_cmd_dst  (),
        .layernorm_cmd_len  (),
        .layernorm_cmd_flags(),
        // GELU
        .gelu_cmd_valid     (gelu_cmd_valid),
        .gelu_cmd_src0      (),
        .gelu_cmd_dst       (),
        .gelu_cmd_len       (),
        .gelu_cmd_flags     (),
        // Vec
        .vec_cmd_valid      (vec_cmd_valid),
        .vec_cmd_src0       (),
        .vec_cmd_src1       (),
        .vec_cmd_dst        (),
        .vec_cmd_len        (),
        .vec_cmd_flags      (),
        .vec_cmd_imm        (),
        // DMA Read
        .dma_rd_cmd_valid   (dma_rd_cmd_valid),
        .dma_rd_cmd_src     (dec_dma_rd_src),
        .dma_rd_cmd_dst     (dec_dma_rd_dst),
        .dma_rd_cmd_len     (dec_dma_rd_len),
        .dma_rd_cmd_flags   (dec_dma_rd_flags),
        // DMA Write
        .dma_wr_cmd_valid   (dma_wr_cmd_valid),
        .dma_wr_cmd_src     (dec_dma_wr_src),
        .dma_wr_cmd_dst     (dec_dma_wr_dst),
        .dma_wr_cmd_len     (dec_dma_wr_len),
        .dma_wr_cmd_flags   (dec_dma_wr_flags),
        // KV
        .kv_cmd_valid       (kv_cmd_valid),
        .kv_cmd_opcode      (),
        .kv_cmd_src0        (),
        .kv_cmd_dst         (),
        .kv_cmd_len         (),
        .kv_cmd_flags       (),
        .kv_cmd_imm         (),
        // RMSNorm (unused in full top)
        .rmsnorm_cmd_valid  (),
        .rmsnorm_cmd_src0   (),
        .rmsnorm_cmd_dst    (),
        .rmsnorm_cmd_len    (),
        .rmsnorm_cmd_gamma  (),
        // RoPE (unused in full top)
        .rope_cmd_valid     (),
        .rope_cmd_src0      (),
        .rope_cmd_dst       (),
        .rope_cmd_num_rows  (),
        .rope_cmd_head_dim  (),
        .rope_cmd_pos_offset(),
        .rope_cmd_sin_base  (),
        .rope_cmd_cos_base  (),
        // SiLU mode (unused in full top)
        .silu_mode          (),
        // Barrier
        .barrier_trigger    (barrier_trigger),
        // Scoreboard issue
        .issue_valid        (issue_valid),
        .issue_engine_id    (issue_engine_id),
        // Debug
        .decoded_instr      (decoded_instr),
        .program_end        (program_end)
    );

    // =========================================================================
    // Scoreboard
    // =========================================================================
    scoreboard_npu #(
        .NUM_ENGINES (NUM_ENGINES)
    ) u_scoreboard (
        .clk              (clk),
        .rst_n            (rst_int_n),
        .issue_valid      (issue_valid),
        .issue_engine_id  (issue_engine_id),
        .engine_done      (engine_done_vec),
        .engine_busy      (engine_busy),
        .can_issue        (can_issue),
        .all_idle         (all_idle)
    );

    // =========================================================================
    // Barrier
    // =========================================================================
    barrier u_barrier (
        .clk      (clk),
        .rst_n    (rst_int_n),
        .trigger  (barrier_trigger),
        .all_idle (all_idle),
        .stall    (barrier_stall),
        .done     ()
    );



    // =========================================================================
    // DMA Command Translation (decode/graph -> DMA engine)
    // =========================================================================
always_comb begin
    dma_rd_int_cmd_valid = 1'b0;
    dma_rd_int_cmd_addr  = 32'd0;
    dma_rd_int_cmd_len   = 24'd0;
    dma_rd_int_cmd_tag   = 4'd0;

    if (graph_mode) begin
        if (graph_dma_rd_req) begin
            dma_rd_int_cmd_valid = 1'b1;
            dma_rd_int_cmd_addr  = graph_dma_abs_addr;
            dma_rd_int_cmd_len   = {8'd0, graph_dma_length};
            dma_rd_int_cmd_tag   = 4'd0;
        end
    end else begin
        dma_rd_int_cmd_valid = dma_rd_cmd_valid;

        case (dec_dma_rd_flags[2:0])
            3'd0:    dma_rd_int_cmd_addr = reg_ddr_base_act + {16'b0, dec_dma_rd_src};
            3'd1:    dma_rd_int_cmd_addr = reg_ddr_base_wgt + {16'b0, dec_dma_rd_src};
            3'd2:    dma_rd_int_cmd_addr = reg_ddr_base_kv  + {16'b0, dec_dma_rd_src};
            3'd4:    dma_rd_int_cmd_addr = reg_ucode_base   + {16'b0, dec_dma_rd_src};
            default: dma_rd_int_cmd_addr = reg_ddr_base_act + {16'b0, dec_dma_rd_src};
        endcase

        dma_rd_int_cmd_len = {8'b0, dec_dma_rd_len};
        dma_rd_int_cmd_tag = dec_dma_rd_flags[6:3];
    end
end

always_comb begin
    dma_wr_int_cmd_valid = 1'b0;
    dma_wr_int_cmd_addr  = 32'd0;
    dma_wr_int_cmd_len   = 24'd0;
    dma_wr_int_cmd_tag   = 4'd0;

    if (graph_mode) begin
        if (graph_dma_wr_req) begin
            dma_wr_int_cmd_valid = 1'b1;
            dma_wr_int_cmd_addr  = graph_dma_abs_addr;
            dma_wr_int_cmd_len   = {8'd0, graph_dma_length};
            dma_wr_int_cmd_tag   = 4'd0;
        end
    end else begin
        dma_wr_int_cmd_valid = dma_wr_cmd_valid;

        case (dec_dma_wr_flags[2:0])
            3'd0:    dma_wr_int_cmd_addr = reg_ddr_base_act + {16'b0, dec_dma_wr_dst};
            3'd3:    dma_wr_int_cmd_addr = reg_ddr_base_out + {16'b0, dec_dma_wr_dst};
            default: dma_wr_int_cmd_addr = reg_ddr_base_out + {16'b0, dec_dma_wr_dst};
        endcase

        dma_wr_int_cmd_len = {8'b0, dec_dma_wr_len};
        dma_wr_int_cmd_tag = dec_dma_wr_flags[6:3];
    end
end
    // assign dma_rd_data_ready = 1'b1; // Always accept DMA data (placeholder)

    // =========================================================================
    // Engine stubs: placeholder done signals for engines not yet fully wired
    // Full wiring requires SRAM arbitration (future enhancement)
    // =========================================================================

    // GEMM: generate done pulse one cycle after cmd_valid
    // In full implementation, gemm_ctrl drives the systolic array
    logic gemm_busy_r;
    logic [15:0] gemm_timer;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) begin
            gemm_busy_r <= 1'b0;
            gemm_timer  <= '0;
            gemm_done   <= 1'b0;
        end else begin
            gemm_done <= 1'b0;
            if (gemm_cmd_valid && !gemm_busy_r) begin
                gemm_busy_r <= 1'b1;
                gemm_timer  <= 16'd10; // Simulated latency
            end else if (gemm_busy_r) begin
                if (gemm_timer == 0) begin
                    gemm_busy_r <= 1'b0;
                    gemm_done   <= 1'b1;
                end else begin
                    gemm_timer <= gemm_timer - 1;
                end
            end
        end
    end

    // Softmax engine done stub
    logic softmax_busy_r;
    logic [15:0] softmax_timer;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) begin
            softmax_busy_r <= 1'b0;
            softmax_timer  <= '0;
            softmax_done   <= 1'b0;
        end else begin
            softmax_done <= 1'b0;
            if (softmax_cmd_valid && !softmax_busy_r) begin
                softmax_busy_r <= 1'b1;
                softmax_timer  <= 16'd10;
            end else if (softmax_busy_r) begin
                if (softmax_timer == 0) begin
                    softmax_busy_r <= 1'b0;
                    softmax_done   <= 1'b1;
                end else begin
                    softmax_timer <= softmax_timer - 1;
                end
            end
        end
    end

    // LayerNorm engine done stub
    logic layernorm_busy_r;
    logic [15:0] layernorm_timer;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) begin
            layernorm_busy_r <= 1'b0;
            layernorm_timer  <= '0;
            layernorm_done   <= 1'b0;
        end else begin
            layernorm_done <= 1'b0;
            if (layernorm_cmd_valid && !layernorm_busy_r) begin
                layernorm_busy_r <= 1'b1;
                layernorm_timer  <= 16'd10;
            end else if (layernorm_busy_r) begin
                if (layernorm_timer == 0) begin
                    layernorm_busy_r <= 1'b0;
                    layernorm_done   <= 1'b1;
                end else begin
                    layernorm_timer <= layernorm_timer - 1;
                end
            end
        end
    end

    // GELU engine done stub
    logic gelu_busy_r;
    logic [15:0] gelu_timer;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) begin
            gelu_busy_r <= 1'b0;
            gelu_timer  <= '0;
            gelu_done   <= 1'b0;
        end else begin
            gelu_done <= 1'b0;
            if (gelu_cmd_valid && !gelu_busy_r) begin
                gelu_busy_r <= 1'b1;
                gelu_timer  <= 16'd5;
            end else if (gelu_busy_r) begin
                if (gelu_timer == 0) begin
                    gelu_busy_r <= 1'b0;
                    gelu_done   <= 1'b1;
                end else begin
                    gelu_timer <= gelu_timer - 1;
                end
            end
        end
    end

    // Vec engine done stub
    logic vec_busy_r;
    logic [15:0] vec_timer;
    always_ff @(posedge clk or negedge rst_int_n) begin
        if (!rst_int_n) begin
            vec_busy_r <= 1'b0;
            vec_timer  <= '0;
            vec_done   <= 1'b0;
        end else begin
            vec_done <= 1'b0;
            if (vec_cmd_valid && !vec_busy_r) begin
                vec_busy_r <= 1'b1;
                vec_timer  <= 16'd5;
            end else if (vec_busy_r) begin
                if (vec_timer == 0) begin
                    vec_busy_r <= 1'b0;
                    vec_done   <= 1'b1;
                end else begin
                    vec_timer <= vec_timer - 1;
                end
            end
        end
    end

    // =========================================================================
    // Simulation-only assertions
    // =========================================================================
`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_int_n && issue_valid) begin
            assert (issue_engine_id < NUM_ENGINES[2:0])
                else $error("Invalid engine ID: %0d", issue_engine_id);
        end
    end
`endif

// -------------------------------------------------------------------------
// Legacy DMA smoke buffer and real Graph DMA stream selection.
//
// In legacy/non-graph mode, keep the old smoke behavior:
//   axi_dma_rd -> dma_smoke_buf -> axi_dma_wr.
//
// In graph mode with artifact_fast_en=0, bypass dma_smoke_buf completely:
//   axi_dma_rd.data_out -> u_graph_core.SRAM0 through graph_sram_dma_wr_*
//   u_graph_core.SRAM0 -> axi_dma_wr.data_in through graph_sram_dma_rd_*
// -------------------------------------------------------------------------
assign dma_smoke_push = (!graph_mode) && dma_rd_data_valid && dma_rd_data_ready;
assign dma_smoke_pop  = (!graph_mode) && dma_wr_data_valid && dma_wr_data_ready;

assign dma_rd_data_ready = graph_mode ? real_dma_load_active
                                      : (dma_smoke_count < DMA_SMOKE_BUF_DEPTH);

always_ff @(posedge clk or negedge rst_int_n) begin
    if (!rst_int_n) begin
        dma_smoke_wr_ptr <= '0;
        dma_smoke_rd_ptr <= '0;
        dma_smoke_count  <= '0;
    end else begin
        if (dma_smoke_push) begin
            dma_smoke_buf[dma_smoke_wr_ptr] <= dma_rd_data;
            dma_smoke_wr_ptr <= dma_smoke_wr_ptr + 1'b1;
        end

        if (dma_smoke_pop) begin
            dma_smoke_rd_ptr <= dma_smoke_rd_ptr + 1'b1;
        end

        case ({dma_smoke_push, dma_smoke_pop})
            2'b10: dma_smoke_count <= dma_smoke_count + 1'b1;
            2'b01: dma_smoke_count <= dma_smoke_count - 1'b1;
            default: ;
        endcase
    end
end

////
//// GRAPH
////
logic graph_sram_dma_rd_valid_q;

always_ff @(posedge clk or negedge rst_int_n) begin
  if (!rst_int_n) begin
    graph_sram_dma_rd_valid_q <= 1'b0;
  end else begin
    graph_sram_dma_rd_valid_q <= graph_sram_dma_rd_en;
  end
end

assign dma_wr_data_valid = graph_mode ? graph_sram_dma_rd_valid_q
                                      : (dma_smoke_count != 0);
                                      
assign dma_wr_data_in    = graph_mode ? {{(P_AXI_DATA_W-64){1'b0}}, graph_sram_dma_rd_data}
                                      : dma_smoke_buf[dma_smoke_rd_ptr];
assign dma_wr_data_last  = graph_mode ? (real_dma_store_active &&
                                         (real_dma_store_pos_q + 16'd8 >= real_dma_store_len_q))
                                      : 1'b0;

// Graph declarations are placed near the top of this module.
// Only sequential/control logic and instances remain below.

always_ff @(posedge clk or negedge rst_int_n) begin
    if (!rst_int_n) begin
        graph_dma_direction_q <= 1'b0;
    end else if (graph_dma_cmd_accept) begin
        graph_dma_direction_q <= graph_dma_direction;
    end
end
assign graph_dma_done =
    artifact_fast_en ? artifact_dma_done_pulse :
    (graph_dma_direction_q ? dma_wr_done : dma_rd_done);


always_ff @(posedge clk or negedge rst_int_n) begin
    if (!rst_int_n) begin
        graph_dma_cmd_seen <= 1'b0;
    end else if (!graph_mode || graph_dma_done) begin
        graph_dma_cmd_seen <= 1'b0;
    end else if (graph_dma_cmd_accept) begin
        graph_dma_cmd_seen <= 1'b1;
    end
end


`ifndef SYNTHESIS
// -------------------------------------------------------------------------
// Quiet DMA debug
// -------------------------------------------------------------------------
// Important: do NOT print every AXI R/W beat here. Block1 has tens of thousands
// of DMA beats, and per-beat printing hides the useful early log. Keep only
// graph-level DMA commands and non-OKAY AXI responses.
always @(posedge clk) begin
    if (rst_int_n && graph_dma_rd_accept) begin
        $display("[%0t] NPU_GRAPH_DMA_RD_ACCEPT graph_off=0x%08x base=0x%08x abs=0x%08x len=%0d",
                 $time,
                 graph_dma_ddr_addr,
                 reg_ddr_base_act,
                 graph_dma_abs_addr,
                 graph_dma_length);
    end

    if (rst_int_n && graph_dma_wr_accept) begin
        $display("[%0t] NPU_GRAPH_DMA_WR_ACCEPT graph_off=0x%08x base=0x%08x abs=0x%08x len=%0d",
                 $time,
                 graph_dma_ddr_addr,
                 reg_ddr_base_act,
                 graph_dma_abs_addr,
                 graph_dma_length);
    end

    if (rst_int_n && m_axi_rvalid && m_axi_rready && (m_axi_rresp != 2'b00)) begin
        $display("[%0t] NPU_AXI_R_ERROR data=0x%032x resp=0x%0x last=%0d",
                 $time, m_axi_rdata, m_axi_rresp, m_axi_rlast);
    end

    if (rst_int_n && m_axi_bvalid && m_axi_bready && (m_axi_bresp != 2'b00)) begin
        $display("[%0t] NPU_AXI_B_ERROR resp=0x%0x", $time, m_axi_bresp);
    end
end
`endif

// =========================================================================
// Graph Compute Core Stage0 (owned SRAM0 inside compute)
// -------------------------------------------------------------------------
// This replaces the old in-top tensor_table + graph_top instance.
// The outer tiny_npu_top still owns:
//   - AXI-Lite register bank
//   - Graph program / descriptor write ports
//   - AXI DMA translation
//   - dma_smoke_buf smoke path
//   - artifact_fast_en simulation path; artifact_ddr_mem remains here
//
// Stage0 core owns SRAM0, handles EW/ReLU internally, and has GEMM-lite writeback.
// It is for integration bring-up, not final Conv/GEMM golden correctness.
// =========================================================================
graph_compute_core_stage0 #(
    .PROG_SRAM_AW (GRAPH_PROG_AW),
    .SRAM0_AW     (16),
    .TDESC_AW     (8)
) u_graph_core (
    .clk             (clk),
    .rst_n           (rst_int_n),

    .start           (graph_start),
    .prog_len        (reg_ucode_len[15:0]),
    .scheduler_mode  (1'b0),

    .graph_prog_we    (graph_prog_we),
    .graph_prog_waddr (graph_prog_waddr),
    .graph_prog_wdata (graph_prog_wdata),

    .tdesc_we       (tdesc_we),
    .tdesc_waddr    (tdesc_waddr),
    .tdesc_wdata    (tdesc_wdata),

    .dma_cmd_valid  (graph_dma_cmd_valid),
    .dma_ddr_addr   (graph_dma_ddr_addr),
    .dma_sram_addr  (graph_dma_sram_addr),
    .dma_length     (graph_dma_length),
    .dma_direction  (graph_dma_direction),
    .dma_strided    (graph_dma_strided),
    .dma_stride     (graph_dma_stride),
    .dma_count      (graph_dma_count),
    .dma_block_len  (graph_dma_block_len),
    .dma_done       (graph_dma_done),

    .dma_sram_wr_en   (graph_sram_dma_wr_en),
    .dma_sram_wr_addr (graph_sram_dma_wr_addr),
    .dma_sram_wr_data (graph_sram_dma_wr_data),
    .dma_sram_wr_mask (graph_sram_dma_wr_mask),
    .dma_sram_rd_en   (graph_sram_dma_rd_en),
    .dma_sram_rd_addr (graph_sram_dma_rd_addr),
    .dma_sram_rd_data (graph_sram_dma_rd_data),

    .graph_done     (graph_done),
    .graph_busy     (graph_busy),
    .graph_status   (graph_status),
    .graph_pc       (graph_pc),
    .graph_last_op  (graph_last_op)
);



endmodule
