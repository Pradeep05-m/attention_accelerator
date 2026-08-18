# Run with: vivado -mode batch -source scripts/synth_pynq_z2.tcl
# This performs the reproducible PYNQ-Z2 integration/synthesis check.  The
# accelerator is behind the Zynq PS and AXI DMA, so its wide AXI-Stream buses
# are fabric nets rather than impossible package I/O.  The block design sets
# gqa_attention_wrapper GROUP_SIZE=1 to time-multiplex each four-head GQA
# group; the default parallel GROUP_SIZE=4 configuration does not fit an
# XC7Z020.
set root [file normalize [file join [file dirname [info script]] ..]]
set build [file join $root build pynq_z2_synth]
# A previous interrupted project run can leave an auto-incremental checkpoint
# reference under utils_1/imports without the checkpoint itself.  Reusing
# that tree makes Vivado fail before synthesis starts, so this reproducible
# build always begins from a clean generated-project directory.
if {[file exists $build]} {
    file delete -force $build
}
file mkdir $build
create_project -force gqa_pynq_z2 $build -part xc7z020clg400-1
# The AXI GPIO/reset/PS IP instances are board-aware.  Setting the same board
# part used by the existing PYNQ-Z2 project before creating the block design
# prevents Vivado from treating those generated IPs as locked because their
# former board selection was "unset".
set pynq_z2_board tul.com.tw:pynq-z2:part0:1.0
if {[llength [get_board_parts -quiet $pynq_z2_board]] == 0} {
    puts "WARNING: Board part $pynq_z2_board is not installed; continuing with the FPGA part only."
    puts "WARNING: Board-aware IP may remain unlocked until the PYNQ-Z2 board files are installed."
} else {
    set_property board_part $pynq_z2_board [current_project]
}
set_property target_language Verilog [current_project]
source [file join $root scripts gqa_sources.tcl]
foreach src [gqa_rtl_sources $root] {
    add_files -norecurse $src
}
# rope_unit uses these files through $readmemh.  Make them explicit project
# sources so both synthesis and post-synthesis simulation resolve the same
# coefficients instead of depending on Vivado's launch directory.
set rope_rom [file join $root rope_rom]
set rope_mem [glob -nocomplain [file join $rope_rom *.mem]]
if {[llength $rope_mem] != 2} {
    error "RoPE ROM files are missing. Run: python3 tools/gen_rope_rom.py --out-dir rope_rom"
}
add_files -norecurse $rope_mem
source [file join $root scripts create_pynq_z2_bd.tcl]
generate_target all [get_files gqa_pynq_z2_bd.bd]
# Vivado 2025 can otherwise schedule the generated auto_us interconnect
# runs before their shared clock/data-width-converter HDL has been exported.
# That manifests as Runs 36-287 "File does not exist" failures even though
# the parent block design is valid.
export_ip_user_files -of_objects [get_files gqa_pynq_z2_bd.bd] -no_script -sync -force -quiet
set_property top gqa_pynq_z2_bd_wrapper [current_fileset]
# The design is slice-packing limited. Area/control-set optimized synthesis
# trades spare LUT capacity for a layout that can pack the registers; apply it
# to the top run and every generated OOC run, not just the BD wrapper.
# `*_synth_1` deliberately finds generated OOC runs, but it does not match
# the top-level run named exactly `synth_1`.  Include both; the top run is
# where the wrapper's FF/control-set packing is decided.
set synth_runs [concat [get_runs synth_1] [get_runs -quiet *_synth_1]]
foreach run $synth_runs {
    set_property strategy Flow_AreaOptimized_high $run
    # Set the directive explicitly as some Vivado 2025 OOC runs retain the
    # displayed default strategy name even after the strategy property changes.
    set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AreaOptimized_high $run
    # The design is FF/control-set packing limited (not LUT limited) on the
    # XC7Z020.  Let synthesis absorb small clock-enable/reset control sets
    # into LUT logic so unrelated registers can share slices.  The default
    # threshold of 4 left more than one thousand control sets and caused
    # placement to fail despite available LUT capacity.
    # 32 over-absorbs controls and exceeds LUT capacity on XC7Z020.  20 is
    # the next bounded trial above 16: it reduces control-set fragmentation
    # while retaining adequate LUT headroom for placement and routing.
    set_property STEPS.SYNTH_DESIGN.ARGS.CONTROL_SET_OPT_THRESHOLD 20 $run
}
# Module-reference IP source changes are not automatically considered stale
# by every Vivado release.  Complete each OOC synthesis run first, so the
# parent synthesis and implementation always consume its new checkpoint.
foreach run $synth_runs {
    if {[get_property NAME $run] ne "synth_1"} {
        reset_run $run
        launch_runs $run -jobs 1
        wait_on_run $run
        if {[get_property STATUS $run] ne "synth_design Complete!"} {
            error "Out-of-context synthesis did not complete: [get_property NAME $run]"
        }
    }
}
# Use the project synthesis run rather than synth_design.  It schedules the
# generated interconnect sub-IPs as OOC runs before elaborating the BD top.
# The generated DMA/interconnect OOC runs each use substantial memory.
# Run them serially so this reproducible build also succeeds on 8-16 GB hosts.
launch_runs synth_1 -jobs 1
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    error "PYNQ-Z2 synthesis did not complete successfully."
}
open_run synth_1
report_utilization -file [file join $build utilization.rpt]
report_timing_summary -file [file join $build timing_summary.rpt]
write_checkpoint -force [file join $build post_synth.dcp]
