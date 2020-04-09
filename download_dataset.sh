#!/bin/bash
ID="1-4YEmY3yK6WMkkVw3a4y-M_7U02CftY-"
NAME="/tmp/assets.tar.gz"
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$ID" -O $NAME
rm -rf /tmp/cookies.txt
tar -xzvf $NAME -C .
rm -r $NAME
