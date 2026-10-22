# Synthetic source mapping fixture; not an executable kernel.
# gfx942 example
lds_value = read_lds()
value = load_tile()
wait_for_tile()
acc = matrix_multiply(value, lds_value)
