#!/bin/bash
ID="1k8IiAu5d8nwiDYGskUdVUXeCkWpUrxKm"
NAME="/tmp/assets.tar.gz"
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$ID" -O $NAME
rm -rf /tmp/cookies.txt
tar -xzvf $NAME -C .
rm -r $NAME
