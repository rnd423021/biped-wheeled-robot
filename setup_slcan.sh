sudo slcand -o -c -f -s8 /dev/ttyACM0 can0
sudo ifconfig can0 txqueuelen 60000
sudo ip link set up can0
