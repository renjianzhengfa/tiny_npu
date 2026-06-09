`timescale 1ns/1ps
`default_nettype none

module axi_full64_to_axil32_ctrl_bridge #(
  parameter int unsigned AXI_ADDR_W = 48,
  parameter int unsigned AXI_DATA_W = 64,
  parameter int unsigned AXI_ID_W   = 2,
  parameter int unsigned AXI_USER_W = 2,
  parameter logic [31:0] AXIL_BASE_ADDR32 = 32'h4000_0000,

  parameter type axi_req_t = logic,
  parameter type axi_rsp_t = logic
) (
input  wire clk_i,
input  wire rst_ni,

input  axi_req_t slv_req_i,
output axi_rsp_t slv_rsp_o,

output logic [31:0] m_axil_awaddr_o,
output logic        m_axil_awvalid_o,
input  wire         m_axil_awready_i,

output logic [31:0] m_axil_wdata_o,
output logic [3:0]  m_axil_wstrb_o,
output logic        m_axil_wvalid_o,
input  wire         m_axil_wready_i,

input  wire [1:0]   m_axil_bresp_i,
input  wire         m_axil_bvalid_i,
output logic        m_axil_bready_o,

output logic [31:0] m_axil_araddr_o,
output logic        m_axil_arvalid_o,
input  wire         m_axil_arready_i,

input  wire [31:0]  m_axil_rdata_i,
input  wire [1:0]   m_axil_rresp_i,
input  wire         m_axil_rvalid_i,
output logic        m_axil_rready_o
);
`ifndef SYNTHESIS
  initial begin
    if (AXI_DATA_W != 64) begin
      $error("axi_full64_to_axil32_ctrl_bridge first version expects AXI_DATA_W=64.");
    end
  end
`endif

  typedef enum logic [1:0] {
    WR_IDLE,
    WR_AXIL,
    WR_RESP
  } wr_state_e;

  typedef enum logic [1:0] {
    RD_IDLE,
    RD_AXIL,
    RD_RESP
  } rd_state_e;

  wr_state_e wr_state_q, wr_state_d;
  rd_state_e rd_state_q, rd_state_d;

  logic [AXI_ADDR_W-1:0] wr_addr_q, wr_addr_d;
  logic [AXI_DATA_W-1:0] wr_data_q, wr_data_d;
  logic [AXI_DATA_W/8-1:0] wr_strb_q, wr_strb_d;
  logic [AXI_ID_W-1:0] wr_id_q, wr_id_d;

  logic have_aw_q, have_aw_d;
  logic have_w_q,  have_w_d;
  logic axil_aw_done_q, axil_aw_done_d;
  logic axil_w_done_q,  axil_w_done_d;

  logic [1:0] wr_bresp_q, wr_bresp_d;

  logic [AXI_ADDR_W-1:0] rd_addr_q, rd_addr_d;
  logic [AXI_ID_W-1:0]   rd_id_q, rd_id_d;
  logic [AXI_DATA_W-1:0] rd_data_q, rd_data_d;
  logic [1:0]            rd_resp_q, rd_resp_d;
  logic                  axil_ar_done_q, axil_ar_done_d;

    // ---------------------------------------------------------------------------
  // 64-bit AXI full -> 32-bit AXI-Lite lane adapter
  //
  // Why this is strobe-aware:
  // - Standard 64-bit AXI writes to addr+4 usually place payload in wdata[63:32]
  //   with wstrb[7:4].
  // - The UART master REG_WRITE32 path may always place the 32-bit payload in
  //   wdata[31:0] with wstrb[3:0], even when the address is addr+4.
  //
  // Therefore:
  // - Prefer the lane indicated by WSTRB.
  // - Fall back to addr[2] only if WSTRB is zero or unusual.
  // ---------------------------------------------------------------------------
  function automatic logic [31:0] pick_wdata32(
    input logic [AXI_DATA_W-1:0]   data_i,
    input logic [AXI_DATA_W/8-1:0] strb_i,
    input logic [AXI_ADDR_W-1:0]   addr_i
  );
    begin
      if (|strb_i[3:0]) begin
        pick_wdata32 = data_i[31:0];
      end else if (|strb_i[7:4]) begin
        pick_wdata32 = data_i[63:32];
      end else begin
        pick_wdata32 = addr_i[2] ? data_i[63:32] : data_i[31:0];
      end
    end
  endfunction

  function automatic logic [3:0] pick_wstrb32(
    input logic [AXI_DATA_W/8-1:0] strb_i,
    input logic [AXI_ADDR_W-1:0]   addr_i
  );
    begin
      if (|strb_i[3:0]) begin
        pick_wstrb32 = strb_i[3:0];
      end else if (|strb_i[7:4]) begin
        pick_wstrb32 = strb_i[7:4];
      end else begin
        pick_wstrb32 = 4'hF;
      end
    end
  endfunction

  // For the current Python REG_READ32 model, always return the AXI-Lite 32-bit
  // read data in the low 32 bits of the full AXI read channel.
  //
  // This avoids Python needing to check addr[2] and pick high/low lane.
  function automatic logic [AXI_DATA_W-1:0] place_rdata32(
    input logic [31:0]             data_i,
    input logic [AXI_ADDR_W-1:0]   addr_i
  );
    logic [AXI_DATA_W-1:0] tmp;
    begin
      tmp = '0;
      tmp[31:0] = data_i;
      place_rdata32 = tmp;
    end
  endfunction
  

  // Full AXI response path.
  always_comb begin
    slv_rsp_o = '0;

    slv_rsp_o.aw_ready = (wr_state_q == WR_IDLE) && !have_aw_q;
    slv_rsp_o.w_ready  = (wr_state_q == WR_IDLE) && !have_w_q;

    slv_rsp_o.b_valid  = (wr_state_q == WR_RESP);
    slv_rsp_o.b.id     = wr_id_q;
    slv_rsp_o.b.resp   = wr_bresp_q;
    slv_rsp_o.b.user   = '0;

    slv_rsp_o.ar_ready = (rd_state_q == RD_IDLE);

    slv_rsp_o.r_valid  = (rd_state_q == RD_RESP);
    slv_rsp_o.r.id     = rd_id_q;
    slv_rsp_o.r.data   = rd_data_q;
    slv_rsp_o.r.resp   = rd_resp_q;
    slv_rsp_o.r.last   = 1'b1;
    slv_rsp_o.r.user   = '0;
  end

  // AXI-Lite write side.
  assign m_axil_awaddr_o  = wr_addr_q[31:0] - AXIL_BASE_ADDR32;
  assign m_axil_awvalid_o = (wr_state_q == WR_AXIL) && !axil_aw_done_q;

  assign m_axil_wdata_o   = pick_wdata32(wr_data_q, wr_strb_q, wr_addr_q);

  assign m_axil_wstrb_o   = pick_wstrb32(wr_strb_q, wr_addr_q);
  assign m_axil_wvalid_o  = (wr_state_q == WR_AXIL) && !axil_w_done_q;

  assign m_axil_bready_o  = (wr_state_q == WR_AXIL);

  // AXI-Lite read side.
  assign m_axil_araddr_o  = rd_addr_q[31:0] - AXIL_BASE_ADDR32;
  assign m_axil_arvalid_o = (rd_state_q == RD_AXIL) && !axil_ar_done_q;
  assign m_axil_rready_o  = (rd_state_q == RD_AXIL);

  always_comb begin
    wr_state_d      = wr_state_q;
    wr_addr_d       = wr_addr_q;
    wr_data_d       = wr_data_q;
    wr_strb_d       = wr_strb_q;
    wr_id_d         = wr_id_q;
    have_aw_d       = have_aw_q;
    have_w_d        = have_w_q;
    axil_aw_done_d  = axil_aw_done_q;
    axil_w_done_d   = axil_w_done_q;
    wr_bresp_d      = wr_bresp_q;

    // Accept full AXI AW.
    if ((wr_state_q == WR_IDLE) && !have_aw_q && slv_req_i.aw_valid) begin
      wr_addr_d = slv_req_i.aw.addr;
      wr_id_d   = slv_req_i.aw.id[AXI_ID_W-1:0];
      have_aw_d = 1'b1;
    end

    // Accept full AXI W.
    if ((wr_state_q == WR_IDLE) && !have_w_q && slv_req_i.w_valid) begin
      wr_data_d = slv_req_i.w.data;
      wr_strb_d = slv_req_i.w.strb;
      have_w_d  = 1'b1;
    end

    case (wr_state_q)
      WR_IDLE: begin
        if ((have_aw_d == 1'b1) && (have_w_d == 1'b1)) begin
          wr_state_d     = WR_AXIL;
          axil_aw_done_d = 1'b0;
          axil_w_done_d  = 1'b0;
        end
      end

      WR_AXIL: begin
        if (m_axil_awvalid_o && m_axil_awready_i)
          axil_aw_done_d = 1'b1;

        if (m_axil_wvalid_o && m_axil_wready_i)
          axil_w_done_d = 1'b1;

        if (m_axil_bvalid_i) begin
          wr_bresp_d = m_axil_bresp_i;
          wr_state_d = WR_RESP;
        end
      end

      WR_RESP: begin
        if (slv_req_i.b_ready) begin
          wr_state_d     = WR_IDLE;
          have_aw_d      = 1'b0;
          have_w_d       = 1'b0;
          axil_aw_done_d = 1'b0;
          axil_w_done_d  = 1'b0;
        end
      end

      default: begin
        wr_state_d = WR_IDLE;
      end
    endcase
  end

  always_comb begin
    rd_state_d      = rd_state_q;
    rd_addr_d       = rd_addr_q;
    rd_id_d         = rd_id_q;
    rd_data_d       = rd_data_q;
    rd_resp_d       = rd_resp_q;
    axil_ar_done_d  = axil_ar_done_q;

    case (rd_state_q)
      RD_IDLE: begin
        if (slv_req_i.ar_valid) begin
          rd_addr_d      = slv_req_i.ar.addr;
          rd_id_d        = slv_req_i.ar.id[AXI_ID_W-1:0];
          axil_ar_done_d = 1'b0;
          rd_state_d     = RD_AXIL;
        end
      end

      RD_AXIL: begin
        if (m_axil_arvalid_o && m_axil_arready_i)
          axil_ar_done_d = 1'b1;

        if (m_axil_rvalid_i) begin
          rd_data_d  = place_rdata32(m_axil_rdata_i, rd_addr_q);
          rd_resp_d  = m_axil_rresp_i;
          rd_state_d = RD_RESP;
        end
      end

      RD_RESP: begin
        if (slv_req_i.r_ready) begin
          rd_state_d     = RD_IDLE;
          axil_ar_done_d = 1'b0;
        end
      end

      default: begin
        rd_state_d = RD_IDLE;
      end
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_state_q     <= WR_IDLE;
      wr_addr_q      <= '0;
      wr_data_q      <= '0;
      wr_strb_q      <= '0;
      wr_id_q        <= '0;
      have_aw_q      <= 1'b0;
      have_w_q       <= 1'b0;
      axil_aw_done_q <= 1'b0;
      axil_w_done_q  <= 1'b0;
      wr_bresp_q     <= 2'b00;

      rd_state_q     <= RD_IDLE;
      rd_addr_q      <= '0;
      rd_id_q        <= '0;
      rd_data_q      <= '0;
      rd_resp_q      <= 2'b00;
      axil_ar_done_q <= 1'b0;
    end else begin
      wr_state_q     <= wr_state_d;
      wr_addr_q      <= wr_addr_d;
      wr_data_q      <= wr_data_d;
      wr_strb_q      <= wr_strb_d;
      wr_id_q        <= wr_id_d;
      have_aw_q      <= have_aw_d;
      have_w_q       <= have_w_d;
      axil_aw_done_q <= axil_aw_done_d;
      axil_w_done_q  <= axil_w_done_d;
      wr_bresp_q     <= wr_bresp_d;

      rd_state_q     <= rd_state_d;
      rd_addr_q      <= rd_addr_d;
      rd_id_q        <= rd_id_d;
      rd_data_q      <= rd_data_d;
      rd_resp_q      <= rd_resp_d;
      axil_ar_done_q <= axil_ar_done_d;
    end
  end

endmodule

`default_nettype wire
