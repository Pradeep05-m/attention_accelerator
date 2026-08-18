import subprocess

cmd = "iverilog -g2012 -o sim_softmax tb_flash_softmax_simple.sv flash_softmax.v bf16_multiplier.v b16_adder.v bf16_subtractor.v bf16_exp.v bf16_mac.v scale_unit.v reciprocal_unit.v && vvp sim_softmax"
out = subprocess.check_output(cmd, shell=True).decode()
print(out)
