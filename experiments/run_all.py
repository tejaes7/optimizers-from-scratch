"""Run every experiment in order and regenerate figures/ and results/ (about 2 minutes)."""

from . import adam_bias_correction, adamw_vs_l2, digits, rosenbrock, rotation

for module in (rosenbrock, rotation, digits, adam_bias_correction, adamw_vs_l2):
    print(f"\n=== {module.__name__} ===")
    module.main()
