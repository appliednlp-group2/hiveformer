tasks=(basketball_in_hoop slide_block_to_target wipe_desk lamp_off close_drawer turn_tap take_usb_out_of_computer turn_oven_on close_microwave)
case=(instructions instructions_fixed)
for case in ${case[@]}
do
  for task in ${tasks[@]}
  do
    for seed in 0 1 2
    do
      cp xp/hiveformer/$case/$task/$seed/success_rate.pkl /share/shota.takashiro/hiveformer_result/$case/$task/$seed
    done
  done
done
