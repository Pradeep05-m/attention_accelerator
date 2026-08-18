# Repair a Vivado project whose auto-incremental synthesis checkpoint was
# deleted.  This only changes generated Vivado run metadata; RTL and BD
# sources are not removed.
#
# Run from the repository root:
#   vivado -mode batch -source scripts/repair_pynq_z2_incremental.tcl

set root [file normalize [file join [file dirname [info script]] ..]]
set build [file join $root build pynq_z2_synth]
set project [file join $build gqa_pynq_z2.xpr]
if {![file exists $project]} {
    error "PYNQ-Z2 project not found: $project"
}

open_project $project
set synth_run [get_runs synth_1]
set stale_dcp [file join $build gqa_pynq_z2.srcs utils_1 imports synth_1 gqa_pynq_z2_bd_wrapper.dcp]

# Vivado stores this missing file both as an incremental checkpoint property
# and as a project source.  Clear both before reset_run regenerates run Tcl.
catch {set_property AUTO_INCREMENTAL_CHECKPOINT false $synth_run}
catch {set_property INCREMENTAL_CHECKPOINT "" $synth_run}
set stale_file [get_files -quiet $stale_dcp]
if {[llength $stale_file] != 0} {
    remove_files $stale_file
}

# Match the reproducible build settings.  This project is limited by control
# set packing rather than LUT count, so allow synthesis to trade spare LUTs
# for fewer distinct CE/reset control sets.
set synth_runs [concat [get_runs synth_1] [get_runs -quiet *_synth_1]]
foreach run $synth_runs {
    set_property strategy Flow_AreaOptimized_high $run
    set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AreaOptimized_high $run
    set_property STEPS.SYNTH_DESIGN.ARGS.CONTROL_SET_OPT_THRESHOLD 20 $run
}

reset_run synth_1
reset_run impl_1
launch_runs synth_1 -jobs 1
wait_on_run synth_1
if {[get_property STATUS $synth_run] ne "synth_design Complete!"} {
    error "Clean synthesis failed: [get_property STATUS $synth_run]"
}
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
puts "Implementation status: [get_property STATUS [get_runs impl_1]]"
close_project
