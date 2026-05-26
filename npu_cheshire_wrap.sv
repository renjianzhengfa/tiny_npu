`timescale 1ns/1ps
`default_nettype none

module npu_cheshire_wrap #(
  parameter int unsigned CHS_ADDR_W = 48,
  parameter int unsigned CHS_DATA_W = 64,

    // NPU DMA master ID width, used on axi_ext_mst_req[1]
    parameter int unsigned CHS_MST_ID_W  = 2,

    // NPU control slave response ID width, used on axi_ext_slv_req[0]
    // This must match the ID width of ctrl_req_i.aw.id / ctrl_req_i.ar.id.
    parameter int unsigned CHS_CTRL_ID_W = 4,

  parameter int unsigned CHS_USER_W = 2,

  // NPU control window seen by Cheshire.
  parameter logic [31:0] NPU_CTRL_BASE32 = 32'h4000_0000,

  parameter type chs_axi_mst_req_t = logic,
  parameter type chs_axi_mst_rsp_t = logic,
  parameter type chs_axi_slv_req_t = logic,
  parameter type chs_axi_slv_rsp_t = logic
) (
  input  logic clk_i,
  input  logic rst_ni,

  // Cheshire -> NPU control slave path.
  input  chs_axi_slv_req_t ctrl_req_i,
  output chs_axi_slv_rsp_t ctrl_rsp_o,

  // NPU DMA master -> Cheshire path.
  output chs_axi_mst_req_t dma_req_o,
  input  chs_axi_mst_rsp_t dma_rsp_i
);

  // ---------------------------------------------------------------------------
  // NPU AXI-Lite control flat wires, 32-bit.
  // ---------------------------------------------------------------------------
  logic [31:0] s_axil_awaddr;
  logic        s_axil_awvalid;
  logic        s_axil_awready;
  logic [31:0] s_axil_wdata;
  logic [3:0]  s_axil_wstrb;
  logic        s_axil_wvalid;
  logic        s_axil_wready;
  logic [1:0]  s_axil_bresp;
  logic        s_axil_bvalid;
  logic        s_axil_bready;

  logic [31:0] s_axil_araddr;
  logic        s_axil_arvalid;
  logic        s_axil_arready;
  logic [31:0] s_axil_rdata;
  logic [1:0]  s_axil_rresp;
  logic        s_axil_rvalid;
  logic        s_axil_rready;

  // ---------------------------------------------------------------------------
  // NPU DMA AXI master flat wires.
  // First version: make NPU DMA data/id width match Cheshire.
  // Address is kept 32-bit and zero-extended in wrapper.
  // ---------------------------------------------------------------------------
  logic [CHS_CTRL_ID_W-1:0]   npu_m_axi_arid;
  logic [31:0]           npu_m_axi_araddr;
  logic [7:0]            npu_m_axi_arlen;
  logic [2:0]            npu_m_axi_arsize;
  logic [1:0]            npu_m_axi_arburst;
  logic                  npu_m_axi_arvalid;
  logic                  npu_m_axi_arready;

  logic [CHS_CTRL_ID_W-1:0]   npu_m_axi_rid;
  logic [CHS_DATA_W-1:0] npu_m_axi_rdata;
  logic [1:0]            npu_m_axi_rresp;
  logic                  npu_m_axi_rlast;
  logic                  npu_m_axi_rvalid;
  logic                  npu_m_axi_rready;

  logic [CHS_CTRL_ID_W-1:0]   npu_m_axi_awid;
  logic [31:0]           npu_m_axi_awaddr;
  logic [7:0]            npu_m_axi_awlen;
  logic [2:0]            npu_m_axi_awsize;
  logic [1:0]            npu_m_axi_awburst;
  logic                  npu_m_axi_awvalid;
  logic                  npu_m_axi_awready;

  logic [CHS_DATA_W-1:0] npu_m_axi_wdata;
  logic [CHS_DATA_W/8-1:0] npu_m_axi_wstrb;
  logic                  npu_m_axi_wlast;
  logic                  npu_m_axi_wvalid;
  logic                  npu_m_axi_wready;

  logic [CHS_CTRL_ID_W-1:0]   npu_m_axi_bid;
  logic [1:0]            npu_m_axi_bresp;
  logic                  npu_m_axi_bvalid;
  logic                  npu_m_axi_bready;

  // ---------------------------------------------------------------------------
  // Control bridge: Cheshire full AXI slave -> 32-bit AXI-Lite.
  // This is intentionally simple for control-register accesses.
  // ---------------------------------------------------------------------------
  axi_full64_to_axil32_ctrl_bridge #(
    .AXI_ADDR_W       (CHS_ADDR_W),
    .AXI_DATA_W       (CHS_DATA_W),
    .AXI_ID_W         (CHS_CTRL_ID_W),
    .AXI_USER_W       (CHS_USER_W),
    .AXIL_BASE_ADDR32 (NPU_CTRL_BASE32),
    .axi_req_t        (chs_axi_slv_req_t),
    .axi_rsp_t        (chs_axi_slv_rsp_t)
  ) i_ctrl_bridge (
    .clk_i,

    .rst_ni,

    .slv_req_i        (ctrl_req_i),
    .slv_rsp_o        (ctrl_rsp_o),

    .m_axil_awaddr_o  (s_axil_awaddr),
    .m_axil_awvalid_o (s_axil_awvalid),
    .m_axil_awready_i (s_axil_awready),

    .m_axil_wdata_o   (s_axil_wdata),
    .m_axil_wstrb_o   (s_axil_wstrb),
    .m_axil_wvalid_o  (s_axil_wvalid),
    .m_axil_wready_i  (s_axil_wready),

    .m_axil_bresp_i   (s_axil_bresp),
    .m_axil_bvalid_i  (s_axil_bvalid),
    .m_axil_bready_o  (s_axil_bready),

    .m_axil_araddr_o  (s_axil_araddr),
    .m_axil_arvalid_o (s_axil_arvalid),
    .m_axil_arready_i (s_axil_arready),

    .m_axil_rdata_i   (s_axil_rdata),
    .m_axil_rresp_i   (s_axil_rresp),
    .m_axil_rvalid_i  (s_axil_rvalid),
    .m_axil_rready_o  (s_axil_rready)
  );

  // ---------------------------------------------------------------------------
  // Pack tiny-NPU flat DMA AXI master -> Cheshire req/rsp struct.
  // ---------------------------------------------------------------------------
  always_comb begin
    dma_req_o = '0;

    // AW channel
    dma_req_o.aw_valid  = npu_m_axi_awvalid;
    dma_req_o.aw.id     = npu_m_axi_awid;
    dma_req_o.aw.addr   = {{(CHS_ADDR_W-32){1'b0}}, npu_m_axi_awaddr};
    dma_req_o.aw.len    = npu_m_axi_awlen;
    dma_req_o.aw.size   = npu_m_axi_awsize;
    dma_req_o.aw.burst  = npu_m_axi_awburst;
    dma_req_o.aw.lock   = 1'b0;
    dma_req_o.aw.cache  = 4'b0011;
    dma_req_o.aw.prot   = 3'b000;
    dma_req_o.aw.qos    = 4'd0;
    dma_req_o.aw.region = 4'd0;
    dma_req_o.aw.atop   = 6'd0;
    dma_req_o.aw.user   = '0;

    // W channel
    dma_req_o.w_valid   = npu_m_axi_wvalid;
    dma_req_o.w.data    = npu_m_axi_wdata;
    dma_req_o.w.strb    = npu_m_axi_wstrb;
    dma_req_o.w.last    = npu_m_axi_wlast;
    dma_req_o.w.user    = '0;

    // B channel
    dma_req_o.b_ready   = npu_m_axi_bready;

    // AR channel
    dma_req_o.ar_valid  = npu_m_axi_arvalid;
    dma_req_o.ar.id     = npu_m_axi_arid;
    dma_req_o.ar.addr   = {{(CHS_ADDR_W-32){1'b0}}, npu_m_axi_araddr};
    dma_req_o.ar.len    = npu_m_axi_arlen;
    dma_req_o.ar.size   = npu_m_axi_arsize;
    dma_req_o.ar.burst  = npu_m_axi_arburst;
    dma_req_o.ar.lock   = 1'b0;
    dma_req_o.ar.cache  = 4'b0011;
    dma_req_o.ar.prot   = 3'b000;
    dma_req_o.ar.qos    = 4'd0;
    dma_req_o.ar.region = 4'd0;
    dma_req_o.ar.user   = '0;

    // R channel
    dma_req_o.r_ready   = npu_m_axi_rready;
  end

  assign npu_m_axi_awready = dma_rsp_i.aw_ready;
  assign npu_m_axi_wready  = dma_rsp_i.w_ready;
  assign npu_m_axi_bvalid  = dma_rsp_i.b_valid;
  assign npu_m_axi_bresp   = dma_rsp_i.b.resp;
  assign npu_m_axi_bid     = dma_rsp_i.b.id[CHS_CTRL_ID_W-1:0];

  assign npu_m_axi_arready = dma_rsp_i.ar_ready;
  assign npu_m_axi_rvalid  = dma_rsp_i.r_valid;
  assign npu_m_axi_rdata   = dma_rsp_i.r.data;
  assign npu_m_axi_rresp   = dma_rsp_i.r.resp;
  assign npu_m_axi_rlast   = dma_rsp_i.r.last;
  assign npu_m_axi_rid     = dma_rsp_i.r.id[CHS_CTRL_ID_W-1:0];

  // ---------------------------------------------------------------------------
  // tiny-NPU instance.
  // Rename original tiny-NPU module `top` to `tiny_npu_top` before using this.
  // ---------------------------------------------------------------------------
  tiny_npu_top #(
    .P_AXI_DATA_W (CHS_DATA_W),
    .P_AXI_ADDR_W (32),
    .P_AXI_STRB_W (CHS_DATA_W / 8),
    .P_AXI_ID_W   (CHS_CTRL_ID_W)
  ) i_tiny_npu (
    .clk           (clk_i),
    .rst_n         (rst_ni),

    .s_axil_awaddr (s_axil_awaddr),
    .s_axil_awvalid(s_axil_awvalid),
    .s_axil_awready(s_axil_awready),
    .s_axil_wdata  (s_axil_wdata),
    .s_axil_wstrb  (s_axil_wstrb),
    .s_axil_wvalid (s_axil_wvalid),
    .s_axil_wready (s_axil_wready),
    .s_axil_bresp  (s_axil_bresp),
    .s_axil_bvalid (s_axil_bvalid),
    .s_axil_bready (s_axil_bready),

    .s_axil_araddr (s_axil_araddr),
    .s_axil_arvalid(s_axil_arvalid),
    .s_axil_arready(s_axil_arready),
    .s_axil_rdata  (s_axil_rdata),
    .s_axil_rresp  (s_axil_rresp),
    .s_axil_rvalid (s_axil_rvalid),
    .s_axil_rready (s_axil_rready),

    .m_axi_arid    (npu_m_axi_arid),
    .m_axi_araddr  (npu_m_axi_araddr),
    .m_axi_arlen   (npu_m_axi_arlen),
    .m_axi_arsize  (npu_m_axi_arsize),
    .m_axi_arburst (npu_m_axi_arburst),
    .m_axi_arvalid (npu_m_axi_arvalid),
    .m_axi_arready (npu_m_axi_arready),
    .m_axi_rid     (npu_m_axi_rid),
    .m_axi_rdata   (npu_m_axi_rdata),
    .m_axi_rresp   (npu_m_axi_rresp),
    .m_axi_rlast   (npu_m_axi_rlast),
    .m_axi_rvalid  (npu_m_axi_rvalid),
    .m_axi_rready  (npu_m_axi_rready),

    .m_axi_awid    (npu_m_axi_awid),
    .m_axi_awaddr  (npu_m_axi_awaddr),
    .m_axi_awlen   (npu_m_axi_awlen),
    .m_axi_awsize  (npu_m_axi_awsize),
    .m_axi_awburst (npu_m_axi_awburst),
    .m_axi_awvalid (npu_m_axi_awvalid),
    .m_axi_awready (npu_m_axi_awready),
    .m_axi_wdata   (npu_m_axi_wdata),
    .m_axi_wstrb   (npu_m_axi_wstrb),
    .m_axi_wlast   (npu_m_axi_wlast),
    .m_axi_wvalid  (npu_m_axi_wvalid),
    .m_axi_wready  (npu_m_axi_wready),
    .m_axi_bid     (npu_m_axi_bid),
    .m_axi_bresp   (npu_m_axi_bresp),
    .m_axi_bvalid  (npu_m_axi_bvalid),
    .m_axi_bready  (npu_m_axi_bready)
  );

endmodule
`default_nettype none