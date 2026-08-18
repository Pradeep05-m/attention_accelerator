# Create the PYNQ-Z2 hardware wrapper around gqa_attention_wrapper.
#
# The accelerator's AXI interfaces remain inside the PL fabric.  The only
# package connections are the PS DDR/FIXED_IO interfaces, preventing the
# 256-bit AXI-Stream buses from being treated as FPGA user I/O.

set bd_name gqa_pynq_z2_bd
if {[llength [get_bd_designs -quiet $bd_name]] == 0} {
    create_bd_design $bd_name
} else {
    current_bd_design $bd_name
}

set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7_0]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.0}] $ps7

# FCLK_RESET0_N is asynchronous.  Synchronize its deassertion before it
# reaches AXI and RTL reset pins; directly using it causes BD 41-1348
# critical warnings and can produce reset-release timing failures in silicon.
set rst_not [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 fclk_reset_not]
set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}] $rst_not
set rst_sync [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M]
set rst_lock [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 rst_dcm_locked]
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {1}] $rst_lock

set ctrl_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ctrl_ic]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {3}] $ctrl_ic

set mem_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_ic]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $mem_ic

set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
# The PYNQ-Z2 build uses four 16-bit BF16 lanes per beat.  Although a
# two-lane stream reduces the arithmetic lanes, it doubles the HEAD_DIM
# window-mux fabric and requires more slices on XC7Z020.  Four lanes is the
# lowest-area supported stream geometry for this RTL.
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_m_axis_mm2s_tdata_width {64} \
    CONFIG.c_s_axis_s2mm_tdata_width {64}] $dma
# Pin both DMA channels to 64 bits so they match the core TILE_DIM=4.
# Fail at creation time if a Vivado/IP version silently changes this setting,
# rather than allowing a later BD validation error (BD 41-237).
if {[get_property CONFIG.c_m_axis_mm2s_tdata_width $dma] != 64 || \
    [get_property CONFIG.c_s_axis_s2mm_tdata_width $dma] != 64} {
    error "axi_dma_0 AXIS widths must both be 64 bits; check the installed AXI DMA IP configuration."
}

set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_plane]
set_property -dict [list CONFIG.C_GPIO_WIDTH {2} CONFIG.C_ALL_OUTPUTS {1}] $gpio

set tagger [create_bd_cell -type module -reference axis_tuser_tagger axis_tuser_tagger_0]
set core   [create_bd_cell -type module -reference gqa_attention_wrapper gqa_attention_0]

# The full Llama-3 GQA group has four Q heads per KV head.  Instantiating
# four complete BF16 score/softmax/value-accumulation pipelines exceeds the
# XC7Z020's LUT and register budgets.  The wrapper supports GROUP_SIZE=1:
# it time-multiplexes those four Q heads while retaining the same 32:8 GQA
# mapping in gqa_controller.  TILE_DIM=4 makes the core a 64-bit AXI-Stream
# endpoint for the DMA, quartering the original 16-lane BF16 datapath.  This
# adds transfer and compute cycles, but does not change the calculation or
# numerical result. KV_BLOCK_LEN=8 similarly time-multiplexes the softmax and
# P·V block hardware over two eight-key tiles instead of one 16-key tile.
# ACC_TILE_DIM is intentionally
# narrower: it is an internal weighted-V micro-engine and has no impact on
# AXI width.  One lane reduces its BF16 multiplier/adder array by 93.75%
# compared with the original 16-lane implementation.  This deliberately
# trades throughput for a design that can be placed on the XC7Z020.

# PS-facing physical interfaces.  These route through the Zynq PS hard IP,
# not through PL user-I/O banks.
make_bd_intf_pins_external [get_bd_intf_pins $ps7/DDR]
make_bd_intf_pins_external [get_bd_intf_pins $ps7/FIXED_IO]

# AXI-Lite control: PS GP0 -> DMA control, plane selector, accelerator.
connect_bd_intf_net [get_bd_intf_pins $ps7/M_AXI_GP0] [get_bd_intf_pins $ctrl_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $ctrl_ic/M00_AXI] [get_bd_intf_pins $dma/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins $ctrl_ic/M01_AXI] [get_bd_intf_pins $gpio/S_AXI]
connect_bd_intf_net [get_bd_intf_pins $ctrl_ic/M02_AXI] [get_bd_intf_pins $core/S_AXI]

# DMA memory traffic: both DMA master channels share PS HP0 through an AXI
# interconnect, keeping DDR transfers entirely inside the PS/PL system.
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_MM2S] [get_bd_intf_pins $mem_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_S2MM] [get_bd_intf_pins $mem_ic/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $mem_ic/M00_AXI] [get_bd_intf_pins $ps7/S_AXI_HP0]

# Stream path.  The tagger creates the two-bit Q/K/V TUSER plane identifier
# from AXI GPIO; software writes 0, 1, or 2 before each input DMA packet.
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins $tagger/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins $tagger/M_AXIS] [get_bd_intf_pins $core/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins $core/M_AXIS] [get_bd_intf_pins $dma/S_AXIS_S2MM]
connect_bd_net [get_bd_pins $gpio/gpio_io_o] [get_bd_pins $tagger/plane_tag]

# Module-reference parameters controlling AXI port widths must be applied
# after their interfaces are connected.  Otherwise Vivado's connection
# propagation restores the default 256-bit (TILE_DIM=16) values.
# Four keys per streaming softmax block is the largest configuration that
# leaves enough LUT/slice packing margin for the complete PS + DMA design on
# an XC7Z020.  This changes only the hardware scheduling granularity: flash
# softmax still carries its running max/sum between blocks, so it preserves
# the attention computation while time-multiplexing half of the probability
# and P*V lanes.  Keep the RTL defaults unchanged for the existing simulation
# regressions, which exercise the wider 16-key configuration.
set_property -dict [list CONFIG.GROUP_SIZE {1} CONFIG.TILE_DIM {4} CONFIG.ACC_TILE_DIM {1} CONFIG.KV_BLOCK_LEN {4}] $core
set_property -dict [list CONFIG.DATA_WIDTH {64}] $tagger

# One 100-MHz PS-generated fabric clock/reset domain for the whole design.
set fclk [get_bd_pins $ps7/FCLK_CLK0]
set ps_resetn [get_bd_pins $ps7/FCLK_RESET0_N]
connect_bd_net $fclk [get_bd_pins $rst_sync/slowest_sync_clk]
connect_bd_net $ps_resetn [get_bd_pins $rst_not/Op1]
connect_bd_net [get_bd_pins $rst_not/Res] [get_bd_pins $rst_sync/ext_reset_in]
connect_bd_net [get_bd_pins $rst_lock/dout] [get_bd_pins $rst_sync/dcm_locked]
set frst [get_bd_pins $rst_sync/peripheral_aresetn]
foreach pin [list \
    $ps7/M_AXI_GP0_ACLK $ps7/S_AXI_HP0_ACLK \
    $core/clk $tagger/aclk \
    $dma/s_axi_lite_aclk $dma/m_axi_mm2s_aclk $dma/m_axi_s2mm_aclk \
    $gpio/s_axi_aclk \
    $ctrl_ic/ACLK $ctrl_ic/S00_ACLK $ctrl_ic/M00_ACLK $ctrl_ic/M01_ACLK $ctrl_ic/M02_ACLK \
    $mem_ic/ACLK $mem_ic/S00_ACLK $mem_ic/S01_ACLK $mem_ic/M00_ACLK] {
    connect_bd_net $fclk [get_bd_pins $pin]
}
foreach pin [list \
    $core/rst_n $tagger/aresetn $dma/axi_resetn $gpio/s_axi_aresetn \
    $ctrl_ic/ARESETN $ctrl_ic/S00_ARESETN $ctrl_ic/M00_ARESETN $ctrl_ic/M01_ARESETN $ctrl_ic/M02_ARESETN \
    $mem_ic/ARESETN $mem_ic/S00_ARESETN $mem_ic/S01_ARESETN $mem_ic/M00_ARESETN] {
    connect_bd_net $frst [get_bd_pins $pin]
}

# These two reset pins are scalar RTL ports rather than AXI reset interface
# pins.  Connect them explicitly and verify that no input was left tied low.
foreach pin [list $core/rst_n $tagger/aresetn] {
    set reset_pin [get_bd_pins $pin]
    if {[llength [get_bd_nets -quiet -of_objects $reset_pin]] != 1} {
        error "Reset pin $pin is not connected to the PS FCLK_RESET0_N net."
    }
}

assign_bd_address
validate_bd_design
save_bd_design

set wrapper [make_wrapper -files [get_files ${bd_name}.bd] -top]
add_files -norecurse $wrapper
