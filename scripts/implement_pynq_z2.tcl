# Rebuild and implement the PYNQ-Z2 hardware design from a clean project.
# This avoids reusing an OOC checkpoint generated with obsolete module
# parameters, which otherwise makes the GUI run a different configuration
# from scripts/create_pynq_z2_bd.tcl.
set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir synth_pynq_z2.tcl]

# The remaining reported setup miss is only 0.178 ns and is dominated by
# routing through replicated BF16 adders.  Let Vivado perform the stronger
# placement and post-route physical optimization passes; these do not change
# RTL functionality or its cycle interface.
set impl_run [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore $impl_run
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true $impl_run
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore $impl_run

launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
if {$impl_status ne "write_bitstream Complete!"} {
    error "PYNQ-Z2 implementation did not complete successfully: $impl_status"
}

open_run impl_1
set root [file normalize [file join $script_dir ..]]
set build [file join $root build pynq_z2_synth]
report_utilization -file [file join $build implementation_utilization.rpt]
report_timing_summary -file [file join $build implementation_timing_summary.rpt]
