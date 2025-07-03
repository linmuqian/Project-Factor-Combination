nohup python train.py "0" "1" &
pid1=$!
nohup python train.py "1" "2" &
pid2=$!
nohup python train.py "2" "3" &
pid3=$!
nohup python train.py "3" "4" &
pid4=$!
wait $pid1 $pid2 $pid3 $pid4