#!/bin/bash

images=( $(shuf -e *.jpg) )

for image in "${images[@]}"; do
  feh --fullscreen --zoom fill "$image" &
  sleep 3
  killall feh
done
