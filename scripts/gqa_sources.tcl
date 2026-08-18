# Canonical RTL source manifest for simulation and synthesis.
# Keep debug-only variants out of the default build to avoid accidentally
# compiling an experimental implementation into the production wrapper.
proc gqa_rtl_sources {root} {
    return [list \
        [file join $root axi_lite_regs.v] \
        [file join $root axi_stream_egress.v] \
        [file join $root axi_stream_ingest.v] \
        [file join $root axis_tuser_tagger.v] \
        [file join $root axis_row_serializer.v] \
        [file join $root b16_adder.v] \
        [file join $root bf16_dot_product_mac.v] \
        [file join $root bf16_exp.v] \
        [file join $root bf16_mac.v] \
        [file join $root bf16_multiplier.v] \
        [file join $root bf16_subtractor.v] \
        [file join $root flash_softmax.v] \
        [file join $root gqa_attention_wrapper.v] \
        [file join $root gqa_controller.v] \
        [file join $root gqa_kv_broadcast.v] \
        [file join $root gqa_output_collector.v] \
        [file join $root output_accumulator.v] \
        [file join $root q_group_buffer.v] \
        [file join $root reciprocal_unit.v] \
        [file join $root rope_unit.v] \
        [file join $root scale_unit.v] \
        [file join $root score_tag_fifo.v] \
        [file join $root score_tile_buffer.v] \
        [file join $root tile_scheduler.v] \
        [file join $root v_row_assembler.v] \
        [file join $root v_tile_buffer.v]]
}
