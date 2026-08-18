# Self-checking GQA wrapper simulation.
# Run from the repository root with:
#   vivado -mode batch -source scripts/run_gqa_wrapper_sim.tcl
set root [file normalize [file join [file dirname [info script]] ..]]
set build [file join $root build gqa_wrapper_sim]
set vectors [file join $root tests data]
if {[info exists ::env(SIM_BUILD_DIR)] && $::env(SIM_BUILD_DIR) ne ""} {
    set build [file normalize $::env(SIM_BUILD_DIR)]
}
if {[info exists ::env(VECTORS_DIR)] && $::env(VECTORS_DIR) ne ""} {
    set vectors [file normalize $::env(VECTORS_DIR)]
}
set project_file [file join $build gqa_wrapper_sim.xpr]
set config_file [file join $vectors gqa_config.txt]

foreach required {gqa_q.mem gqa_k.mem gqa_v.mem gqa_golden.mem} {
    if {![file exists [file join $vectors $required]]} {
        error "Missing [file join $vectors $required]. Run: python3 pyfiles/gen_gqa_wrapper_vectors.py --out-dir tests/data"
    }
}
if {![file exists $config_file]} {
    error "Missing $config_file. Regenerate the vectors with pyfiles/gen_gqa_wrapper_vectors.py."
}
set config_text [read [open $config_file r]]
if {![regexp {SEQ_LEN=([0-9]+)} $config_text -> seq_len]} {
    error "SEQ_LEN missing from $config_file"
}

file mkdir $build
if {[file exists $project_file]} {
    open_project $project_file
} else {
    create_project gqa_wrapper_sim $build -part xc7z020clg400-1
    set_property target_language Verilog [current_project]
    source [file join $root scripts gqa_sources.tcl]
    foreach src [gqa_rtl_sources $root] {
        add_files -norecurse $src
    }
    add_files -fileset sim_1 -norecurse [file join $root tests tb_gqa_attention_wrapper.sv]
    add_files -fileset sim_1 -norecurse [glob -nocomplain [file join $vectors *.mem]]
    add_files -fileset sim_1 -norecurse [glob -nocomplain [file join $root rope_rom *.mem]]
}
set_property top tb_gqa_attention_wrapper_v2 [get_filesets sim_1]
# Elaborate the testbench for the sequence length encoded in the generated
# vectors. This prevents a longer accuracy corpus from being silently
# truncated to the 16-token smoke-test geometry.
set_property -name {xsim.elaborate.xelab.more_options} \
    -value "-generic_top SEQ_LEN=$seq_len -define DEBUG" -objects [get_filesets sim_1]

# The testbench reads its generated vectors by absolute path and writes the
# captured DUT rows under build/, so the result is independent of XSim's CWD.
# +testplusargs are runtime xsim options, not xelab elaboration options.
set xsim_options "-testplusarg MEM_DIR=$vectors -testplusarg ROPE_ROM_DIR=[file join $root rope_rom] -testplusarg RTL_OUT=[file join $build gqa_rtl_out.mem]"
if {[info exists ::env(TRACE)] && $::env(TRACE) ne ""} {
    append xsim_options " -testplusarg TRACE"
}
if {[info exists ::env(OUTPUT_BACKPRESSURE)] && $::env(OUTPUT_BACKPRESSURE) ne ""} {
    append xsim_options " -testplusarg OUTPUT_BACKPRESSURE"
}
if {[info exists ::env(INVALID_CONFIG)] && $::env(INVALID_CONFIG) ne ""} {
    append xsim_options " -testplusarg INVALID_CONFIG"
}
set_property -name {xsim.simulate.xsim.more_options} \
    -value $xsim_options -objects [get_filesets sim_1]

# A persistent Vivado project can otherwise reuse an old XSim snapshot after
# an RTL edit. Accuracy regressions must always elaborate the current sources.
reset_simulation -simset sim_1
launch_simulation
run all
close_sim
close_project
