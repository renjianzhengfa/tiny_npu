module axi_lite_regs
    import npu_pkg::*;
    import axi_types_pkg::*;
    #(
        parameter       UCODE_AW        =   10,
        parameter       GRAPH_PROG_AW   =   9,   // 
        parameter       TDESC_AW        =   8    //
    )(
    input  logic                      clk,
    input  logic                      rst_n,

    // -----------------------------------------------------------------------
    //  AXI4-Lite slave interface
    // -----------------------------------------------------------------------
    // Write-address channel
    input  logic [LITE_ADDR_W-1:0]    s_axil_awaddr,
    input  logic                      s_axil_awvalid,
    output logic                      s_axil_awready,

    // Write-data channel
    input  logic [LITE_DATA_W-1:0]    s_axil_wdata,
    input  logic [LITE_STRB_W-1:0]    s_axil_wstrb,
    input  logic                      s_axil_wvalid,
    output logic                      s_axil_wready,

    // Write-response channel
    output logic [1:0]                s_axil_bresp,
    output logic                      s_axil_bvalid,
    input  logic                      s_axil_bready,

    // Read-address channel
    input  logic [LITE_ADDR_W-1:0]    s_axil_araddr,
    input  logic                      s_axil_arvalid,
    output logic                      s_axil_arready,

    // Read-data channel
    output logic [LITE_DATA_W-1:0]    s_axil_rdata,
    output logic [1:0]                s_axil_rresp,
    output logic                      s_axil_rvalid,
    input  logic                      s_axil_rready,

    // -----------------------------------------------------------------------
    //  Status inputs (directly drive the read-only STATUS register)
    // -----------------------------------------------------------------------
    input  logic                      done_i,
    input  logic                      busy_i,
    input  logic                      error_i,

    // -----------------------------------------------------------------------
    //  Register value outputs
    // -----------------------------------------------------------------------
    output logic [31:0]               ctrl_o,
    output logic [31:0]               status_o,
    output logic [31:0]               ucode_base_o,
    output logic [31:0]               ucode_len_o,
    output logic [31:0]               ddr_base_act_o,
    output logic [31:0]               ddr_base_wgt_o,
    output logic [31:0]               ddr_base_kv_o,
    output logic [31:0]               ddr_base_out_o,
    output logic [31:0]               model_hidden_o,
    output logic [31:0]               model_heads_o,
    output logic [31:0]               model_head_dim_o,
    output logic [31:0]               seq_len_o,
    output logic [31:0]               token_idx_o,
    output logic [31:0]               debug_ctrl_o,
    output logic [31:0]               exec_mode_o,

    // -----------------------------------------------------------------------
    //  Graph Mode status inputs (read-only)
    // -----------------------------------------------------------------------
    input  logic [31:0]               graph_status_i,
    input  logic [31:0]               graph_pc_i,
    input  logic [31:0]               graph_last_op_i,

    // -----------------------------------------------------------------------
    //  Derived control outputs
    // -----------------------------------------------------------------------
    output logic                      start_pulse_o,
    output logic                      soft_reset_o,
    output logic                        ucode_we_o,
    output logic [UCODE_AW-1:0]         ucode_waddr_o,
    output logic [127:0]                ucode_wdata_o,

    //  Graph program SRAM 
    output logic                      graph_prog_we_o,
    output logic [GRAPH_PROG_AW-1:0]  graph_prog_waddr_o,
    output logic [127:0]              graph_prog_wdata_o,

    //  Tensor descriptor table 
    output logic                      tdesc_we_o,
    output logic [TDESC_AW-1:0]       tdesc_waddr_o,
    output logic [255:0]              tdesc_wdata_o,

    input logic [31:0]             perf_total_cycles,
    input logic [31:0]             perf_gemm_cycles,
    input logic [31:0]             perf_dma_cycles,
    input logic [31:0]             perf_ew_cycles,
    input logic [31:0]             perf_stall_cycles
);
///////
//performance
///////
logic [31:0]             i_perf_total_cycles  ;
logic [31:0]             i_perf_gemm_cycles   ;
logic [31:0]             i_perf_dma_cycles    ;
logic [31:0]             i_perf_ew_cycles     ;
logic [31:0]             i_perf_stall_cycles  ;

always_ff @(posedge clk or negedge rst_n) begin
    if(~rst_n)  begin
        i_perf_total_cycles <=  0;
        i_perf_gemm_cycles  <=  0;
        i_perf_dma_cycles   <=  0;
        i_perf_ew_cycles    <=  0;
        i_perf_stall_cycles <=  0;
    end
    else    begin
        i_perf_total_cycles <=  perf_total_cycles;
        i_perf_gemm_cycles  <=  perf_gemm_cycles ;
        i_perf_dma_cycles   <=  perf_dma_cycles  ;
        i_perf_ew_cycles    <=  perf_ew_cycles   ;
        i_perf_stall_cycles <=  perf_stall_cycles;
    end
end

    // =======================================================================
    //  Internal register storage
    // =======================================================================
    logic [31:0] reg_ctrl;
    logic [31:0] reg_ucode_base;
    logic [31:0] reg_ucode_len;
    logic [31:0] reg_ddr_base_act;
    logic [31:0] reg_ddr_base_wgt;
    logic [31:0] reg_ddr_base_kv;
    logic [31:0] reg_ddr_base_out;
    logic [31:0] reg_model_hidden;
    logic [31:0] reg_model_heads;
    logic [31:0] reg_model_head_dim;
    logic [31:0] reg_seq_len;
    logic [31:0] reg_token_idx;
    logic [31:0] reg_debug_ctrl;
    logic [31:0] reg_exec_mode;

    // Soft-reset hold counter (holds reset for 4 cycles)
    logic [2:0]  soft_reset_cnt;

    // =======================================================================
    //  AXI4-Lite Write FSM
    // =======================================================================
    //  We accept AW and W simultaneously (both ready asserted together).
    //  When both handshakes complete, we perform the register write and
    //  present the B response.
    // =======================================================================
    logic        aw_latched;
    logic [31:0] aw_addr_r;
    logic        w_latched;
    logic [31:0] w_data_r;
    logic [3:0]  w_strb_r;

    // AW ready: accept when we have no pending write and B channel is clear
    assign s_axil_awready = !aw_latched && !s_axil_bvalid;
    // W ready: accept when we have no pending write data and B channel is clear
    assign s_axil_wready  = !w_latched  && !s_axil_bvalid;

    // Write response is always OKAY (no decode errors for valid addresses)
    assign s_axil_bresp = AXI_RESP_OKAY;

    // Latch AW and W channels; perform write when both are captured
    logic wr_en;
    logic [31:0] wr_addr;
    logic [31:0] wr_data;
    logic [3:0]  wr_strb;



// =======================================================================
// UCODE AXI-Lite write window
// External address: NPU_BASE + 0x1000 ~ NPU_BASE + 0x1FFF
// Internal offset : 0x1000 ~ 0x1FFF
//
// One 128-bit ucode instruction is written by four 32-bit writes:
//   +0x0 -> instr[31:0]
//   +0x4 -> instr[63:32]
//   +0x8 -> instr[95:64]
//   +0xC -> instr[127:96], then commit to ucode SRAM
// =======================================================================
// 
localparam logic [15:0] REG_UCODE_WIN_BASE   = 16'h1000;
localparam logic [15:0] REG_UCODE_WIN_END    = 16'h2000;

localparam logic [15:0] GRAPH_PROG_WIN_BASE  = 16'h2000;
localparam logic [15:0] GRAPH_PROG_WIN_END   = 16'h4000;

localparam logic [15:0] TDESC_WIN_BASE       = 16'h4000;
localparam logic [15:0] TDESC_WIN_END        = 16'h8000;

wire [15:0] graph_prog_win_off =
    wr_addr[15:0] - GRAPH_PROG_WIN_BASE;

wire [GRAPH_PROG_AW-1:0] graph_prog_win_idx =
    graph_prog_win_off[GRAPH_PROG_AW+3:4];

wire [1:0] graph_prog_word_sel =
    graph_prog_win_off[3:2];

wire [15:0] tdesc_win_off =
    wr_addr[15:0] - TDESC_WIN_BASE;

wire [TDESC_AW-1:0] tdesc_win_idx =
    tdesc_win_off[TDESC_AW+4:5];   //

wire [2:0] tdesc_word_sel =
    tdesc_win_off[4:2];            // 

logic [127:0]              graph_prog_stage;
logic [GRAPH_PROG_AW-1:0]  graph_prog_stage_idx;
logic [3:0]                graph_prog_stage_valid;

wire is_ucode_win_wr_addr =
    (wr_addr[15:0] >= REG_UCODE_WIN_BASE) &&
    (wr_addr[15:0] <  REG_UCODE_WIN_END);

wire is_graph_prog_win_wr_addr =
    (wr_addr[15:0] >= GRAPH_PROG_WIN_BASE) &&
    (wr_addr[15:0] <  GRAPH_PROG_WIN_END);

wire is_tdesc_win_wr_addr =
    (wr_addr[15:0] >= TDESC_WIN_BASE) &&
    (wr_addr[15:0] <  TDESC_WIN_END);

wire ucode_win_wr      = wr_en && is_ucode_win_wr_addr;
wire graph_prog_win_wr = wr_en && is_graph_prog_win_wr_addr;
wire tdesc_win_wr      = wr_en && is_tdesc_win_wr_addr;


wire [15:0] ucode_win_off = wr_addr[15:0] - REG_UCODE_WIN_BASE;
wire [UCODE_AW-1:0] ucode_win_idx = ucode_win_off[UCODE_AW+3:4];
wire [1:0] ucode_word_sel = ucode_win_off[3:2];

logic [127:0]        ucode_stage;
logic [UCODE_AW-1:0] ucode_stage_idx;
logic [3:0]          ucode_stage_valid;


// Debug read-only register offsets for write-window commits.
// These are addresses, not counters.
localparam logic [7:0] REG_GRAPH_PROG_COMMIT_COUNT = 8'h50;
localparam logic [7:0] REG_GRAPH_PROG_LAST_WADDR   = 8'h54;
localparam logic [7:0] REG_GRAPH_PROG_LAST_LOW32   = 8'h58;
localparam logic [7:0] REG_GRAPH_PROG_LAST_HIGH32  = 8'h5C;

localparam logic [7:0] REG_TDESC_COMMIT_COUNT      = 8'h60;
localparam logic [7:0] REG_TDESC_LAST_WADDR        = 8'h64;
localparam logic [7:0] REG_TDESC_LAST_LOW32        = 8'h68;
localparam logic [7:0] REG_TDESC_LAST_HIGH32       = 8'h6C;

localparam logic [7:0] REG_PERF_TOTAL_CYCLES       = 8'h70;
localparam logic [7:0] REG_PERF_GEMM_CYCLES        = 8'h74;
localparam logic [7:0] REG_PERF_DMA_CYCLES         = 8'h78;
localparam logic [7:0] REG_PERF_EW_CYCLES          = 8'h7C;
localparam logic [7:0] REG_PERF_STALL_CYCLES       = 8'h80;

// Actual debug state registers.
logic [31:0] dbg_graph_prog_commit_count;
logic [31:0] dbg_graph_prog_last_waddr;
logic [31:0] dbg_graph_prog_last_low32;
logic [31:0] dbg_graph_prog_last_high32;

logic [31:0] dbg_tdesc_commit_count;
logic [31:0] dbg_tdesc_last_waddr;
logic [31:0] dbg_tdesc_last_low32;
logic [31:0] dbg_tdesc_last_high32;

// Optional AXI-Lite-side shadow memories for readback of write windows.
// These mirror what is committed through graph_prog_we_o/tdesc_we_o.
// They are intended for board bring-up/readback verification.
logic [127:0] graph_prog_shadow [0:(1 << GRAPH_PROG_AW)-1];
logic [255:0] tdesc_shadow      [0:(1 << TDESC_AW)-1];


    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_latched <= 1'b0;
            aw_addr_r  <= 32'd0;
            w_latched  <= 1'b0;
            w_data_r   <= 32'd0;
            w_strb_r   <= 4'd0;
        end else begin
            // wr_en clearing has priority over new latch-set to prevent
            // latches getting stuck when AW+W arrive simultaneously
            if (wr_en) begin
                aw_latched <= 1'b0;
                w_latched  <= 1'b0;
            end else begin
                if (s_axil_awvalid && s_axil_awready)
                    aw_latched <= 1'b1;
                if (s_axil_wvalid && s_axil_wready)
                    w_latched <= 1'b1;
            end

            if (s_axil_awvalid && s_axil_awready)
                aw_addr_r <= s_axil_awaddr;

            if (s_axil_wvalid && s_axil_wready) begin
                w_data_r <= s_axil_wdata;
                w_strb_r <= s_axil_wstrb;
            end
        end
    end

    // Determine when both AW and W data are available (possibly in same cycle)
    logic aw_fire, w_fire;
    assign aw_fire = (s_axil_awvalid && s_axil_awready) || aw_latched;
    assign w_fire  = (s_axil_wvalid  && s_axil_wready)  || w_latched;
    assign wr_en   = aw_fire && w_fire && !s_axil_bvalid;

    // Select between newly arriving data and latched data
    assign wr_addr = aw_latched ? aw_addr_r : s_axil_awaddr;
    assign wr_data = w_latched  ? w_data_r  : s_axil_wdata;
    assign wr_strb = w_latched  ? w_strb_r  : s_axil_wstrb;

    // BVALID management
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            s_axil_bvalid <= 1'b0;
        else if (wr_en)
            s_axil_bvalid <= 1'b1;
        else if (s_axil_bvalid && s_axil_bready)
            s_axil_bvalid <= 1'b0;
    end

    // =======================================================================
    //  Byte-lane write helper: apply strobes to produce new register value
    // =======================================================================
    function automatic logic [31:0] strb_write(
        input logic [31:0] old_val,
        input logic [31:0] new_val,
        input logic [3:0]  strb
    );
        logic [31:0] result;
        result[ 7: 0] = strb[0] ? new_val[ 7: 0] : old_val[ 7: 0];
        result[15: 8] = strb[1] ? new_val[15: 8] : old_val[15: 8];
        result[23:16] = strb[2] ? new_val[23:16] : old_val[23:16];
        result[31:24] = strb[3] ? new_val[31:24] : old_val[31:24];
        return result;
    endfunction

    // =======================================================================
    //  Register write logic
    // =======================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_ctrl           <= 32'd0;
            reg_ucode_base     <= 32'd0;
            reg_ucode_len      <= 32'd0;
            reg_ddr_base_act   <= 32'd0;
            reg_ddr_base_wgt   <= 32'd0;
            reg_ddr_base_kv    <= 32'd0;
            reg_ddr_base_out   <= 32'd0;
            reg_model_hidden   <= 32'd0;
            reg_model_heads    <= 32'd0;
            reg_model_head_dim <= 32'd0;
            reg_seq_len        <= 32'd0;
            reg_token_idx      <= 32'd0;
            reg_debug_ctrl     <= 32'd0;
            reg_exec_mode      <= 32'd0;
            soft_reset_cnt     <= 3'd0;
        end else begin
            // ----------------------------------------------------------
            //  START bit auto-clear: pulse for exactly 1 cycle
            // ----------------------------------------------------------
            if (reg_ctrl[CTRL_START])
                reg_ctrl[CTRL_START] <= 1'b0;

            // ----------------------------------------------------------
            //  SOFT_RESET auto-clear after 4 cycles
            // ----------------------------------------------------------
            if (reg_ctrl[CTRL_SOFT_RESET]) begin
                if (soft_reset_cnt == 3'd3) begin
                    reg_ctrl[CTRL_SOFT_RESET] <= 1'b0;
                    soft_reset_cnt            <= 3'd0;
                end else begin
                    soft_reset_cnt <= soft_reset_cnt + 3'd1;
                end
            end

            // ----------------------------------------------------------
            //  Register write decode
            // ----------------------------------------------------------
            if (wr_en &&!is_ucode_win_wr_addr &&
                        !is_graph_prog_win_wr_addr &&
                        !is_tdesc_win_wr_addr) begin
                case (wr_addr[7:0])
                    REG_CTRL[7:0]: begin
                        reg_ctrl <= strb_write(reg_ctrl, wr_data, wr_strb);
                        if (wr_strb[0] && wr_data[CTRL_SOFT_RESET])
                            soft_reset_cnt <= 3'd0;
                    end

                    REG_UCODE_BASE[7:0]:     reg_ucode_base     <= strb_write(reg_ucode_base,     wr_data, wr_strb);
                    REG_UCODE_LEN[7:0]:      reg_ucode_len      <= strb_write(reg_ucode_len,      wr_data, wr_strb);
                    REG_DDR_BASE_ACT[7:0]:   reg_ddr_base_act   <= strb_write(reg_ddr_base_act,   wr_data, wr_strb);
                    REG_DDR_BASE_WGT[7:0]:   reg_ddr_base_wgt   <= strb_write(reg_ddr_base_wgt,   wr_data, wr_strb);
                    REG_DDR_BASE_KV[7:0]:    reg_ddr_base_kv    <= strb_write(reg_ddr_base_kv,    wr_data, wr_strb);
                    REG_DDR_BASE_OUT[7:0]:   reg_ddr_base_out   <= strb_write(reg_ddr_base_out,   wr_data, wr_strb);
                    REG_MODEL_HIDDEN[7:0]:   reg_model_hidden   <= strb_write(reg_model_hidden,   wr_data, wr_strb);
                    REG_MODEL_HEADS[7:0]:    reg_model_heads    <= strb_write(reg_model_heads,    wr_data, wr_strb);
                    REG_MODEL_HEAD_DIM[7:0]: reg_model_head_dim <= strb_write(reg_model_head_dim, wr_data, wr_strb);
                    REG_SEQ_LEN[7:0]:        reg_seq_len        <= strb_write(reg_seq_len,        wr_data, wr_strb);
                    REG_TOKEN_IDX[7:0]:      reg_token_idx      <= strb_write(reg_token_idx,      wr_data, wr_strb);
                    REG_DEBUG_CTRL[7:0]:     reg_debug_ctrl     <= strb_write(reg_debug_ctrl,     wr_data, wr_strb);
                    REG_EXEC_MODE[7:0]:      reg_exec_mode      <= strb_write(reg_exec_mode,      wr_data, wr_strb);

                    default: ;
                endcase
            end
        end
    end

    // =======================================================================
    //  AXI4-Lite Read FSM
    // =======================================================================
    //  Accept AR, decode address, drive R data.
    // =======================================================================
    logic        rd_pending;
    logic [31:0] rd_addr_r;

    wire is_graph_prog_win_rd_addr =
        (rd_addr_r[15:0] >= GRAPH_PROG_WIN_BASE) &&
        (rd_addr_r[15:0] <  GRAPH_PROG_WIN_END);

    wire [15:0] rd_graph_prog_win_off =
        rd_addr_r[15:0] - GRAPH_PROG_WIN_BASE;

    wire [GRAPH_PROG_AW-1:0] rd_graph_prog_win_idx =
        rd_graph_prog_win_off[GRAPH_PROG_AW+3:4];

    wire [1:0] rd_graph_prog_word_sel =
        rd_graph_prog_win_off[3:2];

    wire is_tdesc_win_rd_addr =
        (rd_addr_r[15:0] >= TDESC_WIN_BASE) &&
        (rd_addr_r[15:0] <  TDESC_WIN_END);

    wire [15:0] rd_tdesc_win_off =
        rd_addr_r[15:0] - TDESC_WIN_BASE;

    wire [TDESC_AW-1:0] rd_tdesc_win_idx =
        rd_tdesc_win_off[TDESC_AW+4:5];

    wire [2:0] rd_tdesc_word_sel =
        rd_tdesc_win_off[4:2];

    assign s_axil_arready = !rd_pending && !s_axil_rvalid;
    assign s_axil_rresp   = AXI_RESP_OKAY;

    // Assemble the read-only STATUS register from external inputs
    logic [31:0] status_reg;
    assign status_reg = {29'd0, error_i, busy_i, done_i};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axil_rvalid <= 1'b0;
            s_axil_rdata  <= 32'd0;
            rd_pending    <= 1'b0;
            rd_addr_r     <= 32'd0;
        end else begin
            // Latch read address
            if (s_axil_arvalid && s_axil_arready) begin
                rd_pending <= 1'b1;
                rd_addr_r  <= s_axil_araddr;
            end

            // Drive read data one cycle after accepting AR
            if (rd_pending && !s_axil_rvalid) begin
                s_axil_rvalid <= 1'b1;
                rd_pending    <= 1'b0;

                if (is_graph_prog_win_rd_addr) begin
                    unique case (rd_graph_prog_word_sel)
                        2'd0: s_axil_rdata <= graph_prog_shadow[rd_graph_prog_win_idx][31:0];
                        2'd1: s_axil_rdata <= graph_prog_shadow[rd_graph_prog_win_idx][63:32];
                        2'd2: s_axil_rdata <= graph_prog_shadow[rd_graph_prog_win_idx][95:64];
                        2'd3: s_axil_rdata <= graph_prog_shadow[rd_graph_prog_win_idx][127:96];
                        default: s_axil_rdata <= 32'hDEAD_BEEF;
                    endcase
                end else if (is_tdesc_win_rd_addr) begin
                    unique case (rd_tdesc_word_sel)
                        3'd0: s_axil_rdata <= tdesc_shadow[rd_tdesc_win_idx][31:0];
                        3'd1: s_axil_rdata <= tdesc_shadow[rd_tdesc_win_idx][63:32];
                        3'd2: s_axil_rdata <= tdesc_shadow[rd_tdesc_win_idx][95:64];
                        3'd3: s_axil_rdata <= tdesc_shadow[rd_tdesc_win_idx][127:96];
                        3'd4: s_axil_rdata <= tdesc_shadow[rd_tdesc_win_idx][159:128];
                        3'd5: s_axil_rdata <= tdesc_shadow[rd_tdesc_win_idx][191:160];
                        3'd6: s_axil_rdata <= tdesc_shadow[rd_tdesc_win_idx][223:192];
                        3'd7: s_axil_rdata <= tdesc_shadow[rd_tdesc_win_idx][255:224];
                        default: s_axil_rdata <= 32'hDEAD_BEEF;
                    endcase
                end else begin
                    case (rd_addr_r[7:0])
                    REG_CTRL[7:0]:           s_axil_rdata <= reg_ctrl;
                    REG_STATUS[7:0]:         s_axil_rdata <= status_reg;
                    REG_UCODE_BASE[7:0]:     s_axil_rdata <= reg_ucode_base;
                    REG_UCODE_LEN[7:0]:      s_axil_rdata <= reg_ucode_len;
                    REG_DDR_BASE_ACT[7:0]:   s_axil_rdata <= reg_ddr_base_act;
                    REG_DDR_BASE_WGT[7:0]:   s_axil_rdata <= reg_ddr_base_wgt;
                    REG_DDR_BASE_KV[7:0]:    s_axil_rdata <= reg_ddr_base_kv;
                    REG_DDR_BASE_OUT[7:0]:   s_axil_rdata <= reg_ddr_base_out;
                    REG_MODEL_HIDDEN[7:0]:   s_axil_rdata <= reg_model_hidden;
                    REG_MODEL_HEADS[7:0]:    s_axil_rdata <= reg_model_heads;
                    REG_MODEL_HEAD_DIM[7:0]: s_axil_rdata <= reg_model_head_dim;
                    REG_SEQ_LEN[7:0]:        s_axil_rdata <= reg_seq_len;
                    REG_TOKEN_IDX[7:0]:      s_axil_rdata <= reg_token_idx;
                    REG_DEBUG_CTRL[7:0]:     s_axil_rdata <= reg_debug_ctrl;
                    REG_EXEC_MODE[7:0]:      s_axil_rdata <= reg_exec_mode;
                    REG_GRAPH_STATUS[7:0]:   s_axil_rdata <= graph_status_i;
                    REG_GRAPH_PC[7:0]:       s_axil_rdata <= graph_pc_i;
                    REG_GRAPH_LAST_OP[7:0]:  s_axil_rdata <= graph_last_op_i;
                    REG_GRAPH_PROG_COMMIT_COUNT: s_axil_rdata <= dbg_graph_prog_commit_count;
                    REG_GRAPH_PROG_LAST_WADDR:   s_axil_rdata <= dbg_graph_prog_last_waddr;
                    REG_GRAPH_PROG_LAST_LOW32:   s_axil_rdata <= dbg_graph_prog_last_low32;
                    REG_GRAPH_PROG_LAST_HIGH32:  s_axil_rdata <= dbg_graph_prog_last_high32;

                    REG_TDESC_COMMIT_COUNT:      s_axil_rdata <= dbg_tdesc_commit_count;
                    REG_TDESC_LAST_WADDR:        s_axil_rdata <= dbg_tdesc_last_waddr;
                    REG_TDESC_LAST_LOW32:        s_axil_rdata <= dbg_tdesc_last_low32;
                    REG_TDESC_LAST_HIGH32:       s_axil_rdata <= dbg_tdesc_last_high32;

                    REG_PERF_TOTAL_CYCLES:       s_axil_rdata <= i_perf_total_cycles;
                    REG_PERF_GEMM_CYCLES:        s_axil_rdata <= i_perf_gemm_cycles;
                    REG_PERF_DMA_CYCLES:         s_axil_rdata <= i_perf_dma_cycles;
                    REG_PERF_EW_CYCLES:          s_axil_rdata <= i_perf_ew_cycles;
                    REG_PERF_STALL_CYCLES:       s_axil_rdata <= i_perf_stall_cycles;
                    default:                 s_axil_rdata <= 32'hDEAD_BEEF;
                    endcase
                end
            end

            // De-assert RVALID when master accepts the data
            if (s_axil_rvalid && s_axil_rready)
                s_axil_rvalid <= 1'b0;
        end
    end

    // =======================================================================
    //  Output assignments
    // =======================================================================
    assign ctrl_o           = reg_ctrl;
    assign status_o         = status_reg;
    assign ucode_base_o     = reg_ucode_base;
    assign ucode_len_o      = reg_ucode_len;
    assign ddr_base_act_o   = reg_ddr_base_act;
    assign ddr_base_wgt_o   = reg_ddr_base_wgt;
    assign ddr_base_kv_o    = reg_ddr_base_kv;
    assign ddr_base_out_o   = reg_ddr_base_out;
    assign model_hidden_o   = reg_model_hidden;
    assign model_heads_o    = reg_model_heads;
    assign model_head_dim_o = reg_model_head_dim;
    assign seq_len_o        = reg_seq_len;
    assign token_idx_o      = reg_token_idx;
    assign debug_ctrl_o     = reg_debug_ctrl;
    assign exec_mode_o      = reg_exec_mode;

    // START is a single-cycle pulse derived from the register bit
    assign start_pulse_o = reg_ctrl[CTRL_START];

    // SOFT_RESET stays high while the counter is active
    assign soft_reset_o  = reg_ctrl[CTRL_SOFT_RESET];

    
always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    ucode_stage       <= 128'd0;
    ucode_stage_idx   <= '0;
    ucode_stage_valid <= 4'b0000;
    ucode_we_o        <= 1'b0;
    ucode_waddr_o     <= '0;
    ucode_wdata_o     <= 128'd0;
  end else begin
    ucode_we_o <= 1'b0;

    if (ucode_win_wr) begin
      if ((ucode_stage_valid == 4'b0000) || (ucode_stage_idx != ucode_win_idx)) begin
        ucode_stage       <= 128'd0;
        ucode_stage_valid <= 4'b0000;
        ucode_stage_idx   <= ucode_win_idx;
      end

      case (ucode_word_sel)
        2'd0: ucode_stage[31:0]   <= wr_data;
        2'd1: ucode_stage[63:32]  <= wr_data;
        2'd2: ucode_stage[95:64]  <= wr_data;
        2'd3: ucode_stage[127:96] <= wr_data;
      endcase

      ucode_stage_valid[ucode_word_sel] <= 1'b1;

      if (ucode_word_sel == 2'd3) begin
        ucode_we_o    <= 1'b1;
        ucode_waddr_o <= ucode_win_idx;
        ucode_wdata_o <= {
          wr_data,
          ucode_stage[95:64],
          ucode_stage[63:32],
          ucode_stage[31:0]
        };
      end
    end
  end
end


always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    graph_prog_stage       <= 128'd0;
    graph_prog_stage_idx   <= '0;
    graph_prog_stage_valid <= 4'b0000;
    graph_prog_we_o        <= 1'b0;
    graph_prog_waddr_o     <= '0;
    graph_prog_wdata_o     <= 128'd0;

    dbg_graph_prog_commit_count <= 32'd0;
    dbg_graph_prog_last_waddr   <= 32'd0;
    dbg_graph_prog_last_low32   <= 32'd0;
    dbg_graph_prog_last_high32  <= 32'd0;
  end else begin
    graph_prog_we_o <= 1'b0;

    if (graph_prog_win_wr) begin
      if ((graph_prog_stage_valid == 4'b0000) ||
          (graph_prog_stage_idx != graph_prog_win_idx)) begin
        graph_prog_stage       <= 128'd0;
        graph_prog_stage_valid <= 4'b0000;
        graph_prog_stage_idx   <= graph_prog_win_idx;
      end

      case (graph_prog_word_sel)
        2'd0: graph_prog_stage[31:0]    <= wr_data;
        2'd1: graph_prog_stage[63:32]   <= wr_data;
        2'd2: graph_prog_stage[95:64]   <= wr_data;
        2'd3: graph_prog_stage[127:96]  <= wr_data;
      endcase

      graph_prog_stage_valid[graph_prog_word_sel] <= 1'b1;

      if (graph_prog_word_sel == 2'd3) begin
        graph_prog_we_o    <= 1'b1;
        graph_prog_waddr_o <= graph_prog_win_idx;
        graph_prog_wdata_o <= {
          wr_data,
          graph_prog_stage[95:64],
          graph_prog_stage[63:32],
          graph_prog_stage[31:0]
        };

        dbg_graph_prog_commit_count <= dbg_graph_prog_commit_count + 32'd1;
        dbg_graph_prog_last_waddr   <= {{(32-GRAPH_PROG_AW){1'b0}}, graph_prog_win_idx};
        dbg_graph_prog_last_low32   <= graph_prog_stage[31:0];
        dbg_graph_prog_last_high32  <= wr_data;
        graph_prog_shadow[graph_prog_win_idx] <= {
          wr_data,
          graph_prog_stage[95:64],
          graph_prog_stage[63:32],
          graph_prog_stage[31:0]
        };
      end
    end
  end
end 

logic [255:0]        tdesc_stage;
logic [TDESC_AW-1:0] tdesc_stage_idx;
logic [7:0]          tdesc_stage_valid;


always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    tdesc_stage       <= 256'd0;
    tdesc_stage_idx   <= '0;
    tdesc_stage_valid <= 8'b0;
    tdesc_we_o        <= 1'b0;
    tdesc_waddr_o     <= '0;
    tdesc_wdata_o     <= 256'd0;

    dbg_tdesc_commit_count <= 32'd0;
    dbg_tdesc_last_waddr   <= 32'd0;
    dbg_tdesc_last_low32   <= 32'd0;
    dbg_tdesc_last_high32  <= 32'd0;
  end else begin
    tdesc_we_o <= 1'b0;

    if (tdesc_win_wr) begin
      if ((tdesc_stage_valid == 8'b0) ||
          (tdesc_stage_idx != tdesc_win_idx)) begin
        tdesc_stage       <= 256'd0;
        tdesc_stage_valid <= 8'b0;
        tdesc_stage_idx   <= tdesc_win_idx;
      end

      case (tdesc_word_sel)
        3'd0: tdesc_stage[31:0]     <= wr_data;
        3'd1: tdesc_stage[63:32]    <= wr_data;
        3'd2: tdesc_stage[95:64]    <= wr_data;
        3'd3: tdesc_stage[127:96]   <= wr_data;
        3'd4: tdesc_stage[159:128]  <= wr_data;
        3'd5: tdesc_stage[191:160]  <= wr_data;
        3'd6: tdesc_stage[223:192]  <= wr_data;
        3'd7: tdesc_stage[255:224]  <= wr_data;
      endcase

      tdesc_stage_valid[tdesc_word_sel] <= 1'b1;

      if (tdesc_word_sel == 3'd7) begin
        tdesc_we_o    <= 1'b1;
        tdesc_waddr_o <= tdesc_win_idx;
        tdesc_wdata_o <= {
          wr_data,
          tdesc_stage[223:192],
          tdesc_stage[191:160],
          tdesc_stage[159:128],
          tdesc_stage[127:96],
          tdesc_stage[95:64],
          tdesc_stage[63:32],
          tdesc_stage[31:0]
        };

        dbg_tdesc_commit_count <= dbg_tdesc_commit_count + 32'd1;
        dbg_tdesc_last_waddr   <= {{(32-TDESC_AW){1'b0}}, tdesc_win_idx};
        dbg_tdesc_last_low32   <= tdesc_stage[31:0];
        dbg_tdesc_last_high32  <= wr_data;
        tdesc_shadow[tdesc_win_idx] <= {
          wr_data,
          tdesc_stage[223:192],
          tdesc_stage[191:160],
          tdesc_stage[159:128],
          tdesc_stage[127:96],
          tdesc_stage[95:64],
          tdesc_stage[63:32],
          tdesc_stage[31:0]
        };
      end
    end
  end
end

endmodule : axi_lite_regs
